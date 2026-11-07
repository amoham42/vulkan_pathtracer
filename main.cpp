// Vulkan hardware ray-traced path tracer: interactive window or headless PNG render.

#include "context.h"
#include "options.h"
#include "renderer.h"
#include "scene.h"

#include <GLFW/glfw3.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;

/// Swapchain for the window. The renderer's LDR image is blitted into it each frame.
struct Presenter {
    Presenter(Context& ctx, vk::Extent2D extent) : ctx(ctx), extent(extent) {
        const auto caps = ctx.physicalDevice.getSurfaceCapabilitiesKHR(*ctx.surface);
        const auto formats = ctx.physicalDevice.getSurfaceFormatsKHR(*ctx.surface);
        vk::SurfaceFormatKHR format = formats.front();
        for (const auto& f : formats)
            if ((f.format == vk::Format::eB8G8R8A8Unorm || f.format == vk::Format::eR8G8B8A8Unorm) &&
                f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
                format = f;  // UNORM: tonemap.comp already wrote sRGB-encoded values

        vk::PresentModeKHR mode = vk::PresentModeKHR::eFifo;
        for (auto m : ctx.physicalDevice.getSurfacePresentModesKHR(*ctx.surface))
            if (m == vk::PresentModeKHR::eMailbox) mode = m;  // don't let vsync throttle accumulation

        uint32_t count = std::max(caps.minImageCount, 2u);
        if (caps.maxImageCount) count = std::min(count + 1, caps.maxImageCount);
        vk::SwapchainCreateInfoKHR info;
        info.setSurface(*ctx.surface)
            .setMinImageCount(count)
            .setImageFormat(format.format)
            .setImageColorSpace(format.colorSpace)
            .setImageExtent(extent)
            .setImageArrayLayers(1)
            .setImageUsage(vk::ImageUsageFlagBits::eTransferDst)
            .setPreTransform(caps.currentTransform)
            .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
            .setPresentMode(mode)
            .setClipped(true);
        swapchain = ctx.device->createSwapchainKHRUnique(info);
        images = ctx.device->getSwapchainImagesKHR(*swapchain);
        acquired = ctx.device->createSemaphoreUnique({});
        rendered = ctx.device->createSemaphoreUnique({});
        cmd = std::move(ctx.device->allocateCommandBuffersUnique({*ctx.commandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
    }

    void frame(Renderer& renderer) {
        using PS = vk::PipelineStageFlagBits2;
        using A = vk::AccessFlagBits2;
        const uint32_t index = ctx.device->acquireNextImageKHR(*swapchain, UINT64_MAX, *acquired).value;
        const vk::Image target = images[index];

        cmd->reset();
        cmd->begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        renderer.record(*cmd);
        imageBarrier(*cmd, target, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, PS::eBlit, A::eNone,
                     PS::eBlit, A::eTransferWrite);
        const std::array<vk::Offset3D, 2> corners{vk::Offset3D{0, 0, 0},
                                                  vk::Offset3D{int32_t(extent.width), int32_t(extent.height), 1}};
        const vk::ImageBlit blit({vk::ImageAspectFlagBits::eColor, 0, 0, 1}, corners, {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                                 corners);
        cmd->blitImage(*renderer.ldr.image, vk::ImageLayout::eGeneral, target, vk::ImageLayout::eTransferDstOptimal, blit,
                       vk::Filter::eNearest);
        imageBarrier(*cmd, target, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR, PS::eBlit,
                     A::eTransferWrite, PS::eNone, A::eNone);
        cmd->end();

        const vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eTransfer;
        ctx.queue.submit(vk::SubmitInfo(*acquired, waitStage, *cmd, *rendered));
        const auto result = ctx.queue.presentKHR(vk::PresentInfoKHR(*rendered, *swapchain, index));
        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) throw std::runtime_error("present failed");
        ctx.queue.waitIdle();  // simple and fine for a progressive renderer; also lets us read the timestamps
        renderer.finishFrame();
    }

    Context& ctx;
    vk::Extent2D extent;
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> images;
    vk::UniqueSemaphore acquired, rendered;
    vk::UniqueCommandBuffer cmd;
};

struct Input {
    Renderer* renderer = nullptr;
    float moveScale = 1.0f;
    bool saveRequested = false;
};

void onKey(GLFWwindow* window, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    auto* in = static_cast<Input*>(glfwGetWindowUserPointer(window));
    Renderer& r = *in->renderer;
    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GLFW_TRUE); break;
        case GLFW_KEY_N: r.denoise = !r.denoise; break;
        case GLFW_KEY_EQUAL:
        case GLFW_KEY_KP_ADD: r.exposure *= 1.25f; break;
        case GLFW_KEY_MINUS:
        case GLFW_KEY_KP_SUBTRACT: r.exposure /= 1.25f; break;
        case GLFW_KEY_P: in->saveRequested = true; break;
        default: break;
    }
}

int runWindow(Context& ctx, Renderer& renderer, float sceneSize) {
    Presenter presenter(ctx, renderer.extent());
    Input input{&renderer, sceneSize};
    glfwSetWindowUserPointer(ctx.window, &input);
    glfwSetKeyCallback(ctx.window, onKey);

    auto last = Clock::now(), lastTitle = last;
    double mouseX = 0, mouseY = 0;
    bool dragging = false;
    int shot = 0;
    while (!glfwWindowShouldClose(ctx.window)) {
        glfwPollEvents();
        const auto now = Clock::now();
        const float dt = std::chrono::duration<float>(now - last).count();
        last = now;

        // camera: WASD/QE move, right-drag looks, R/F zoom; any change restarts accumulation
        Camera& cam = renderer.camera;
        const Camera before = cam;
        const glm::vec3 fwd = cam.forward(), right = glm::normalize(glm::cross(fwd, glm::vec3(0, 1, 0)));
        const float speed = 0.35f * input.moveScale * dt;
        auto down = [&](int key) { return glfwGetKey(ctx.window, key) == GLFW_PRESS; };
        if (down(GLFW_KEY_W)) cam.eye += fwd * speed;
        if (down(GLFW_KEY_S)) cam.eye -= fwd * speed;
        if (down(GLFW_KEY_D)) cam.eye += right * speed;
        if (down(GLFW_KEY_A)) cam.eye -= right * speed;
        if (down(GLFW_KEY_E)) cam.eye.y += speed;
        if (down(GLFW_KEY_Q)) cam.eye.y -= speed;
        if (down(GLFW_KEY_R)) cam.fovY = std::max(5.0f, cam.fovY - 30.0f * dt);
        if (down(GLFW_KEY_F)) cam.fovY = std::min(120.0f, cam.fovY + 30.0f * dt);

        double x, y;
        glfwGetCursorPos(ctx.window, &x, &y);
        if (glfwGetMouseButton(ctx.window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            if (dragging) {
                cam.yaw += float(x - mouseX) * 0.004f;
                cam.pitch = glm::clamp(cam.pitch - float(y - mouseY) * 0.004f, -1.5f, 1.5f);
            }
            dragging = true;
        } else {
            dragging = false;
        }
        mouseX = x;
        mouseY = y;
        if (cam.eye != before.eye || cam.yaw != before.yaw || cam.pitch != before.pitch || cam.fovY != before.fovY) renderer.reset();

        presenter.frame(renderer);

        if (input.saveRequested) {
            input.saveRequested = false;
            const std::string name = "screenshot_" + std::to_string(shot++) + ".png";
            renderer.savePng(name);
            std::cout << "saved " << name << "\n";
        }
        if (now - lastTitle > std::chrono::milliseconds(250)) {
            lastTitle = now;
            char title[256];
            std::snprintf(title, sizeof(title), "Vulkan Path Tracer  |  %u spp  |  trace %.1f ms  denoise %s %.1f ms  |  %s",
                          renderer.samples(), renderer.traceMs, renderer.denoise ? "on" : "off", renderer.postMs,
                          ctx.deviceName.c_str());
            glfwSetWindowTitle(ctx.window, title);
        }
    }
    return 0;
}

int runHeadless(Context& ctx, Renderer& renderer, const Options& o) {
    const uint32_t frames = (o.targetSpp + renderer.samplesPerFrame - 1) / renderer.samplesPerFrame;
    auto cmd = std::move(ctx.device->allocateCommandBuffersUnique({*ctx.commandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
    double traceTotal = 0.0;
    const auto start = Clock::now();
    for (uint32_t i = 0; i < frames; i++) {
        cmd->reset();
        cmd->begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        renderer.record(*cmd);
        cmd->end();
        ctx.queue.submit(vk::SubmitInfo().setCommandBuffers(*cmd));
        ctx.queue.waitIdle();
        renderer.finishFrame();
        traceTotal += renderer.traceMs;
        if ((i + 1) % 16 == 0 || i + 1 == frames)
            std::printf("\r  %u / %u spp", renderer.samples(), frames * renderer.samplesPerFrame), std::fflush(stdout);
    }
    const double wall = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    const double pixels = double(renderer.extent().width) * renderer.extent().height;
    std::printf("\n  GPU trace %.0f ms total (%.2f ms per frame of %u spp), %.0f M paths/s, wall %.0f ms\n", traceTotal,
                traceTotal / frames, renderer.samplesPerFrame, pixels * renderer.samples() / (traceTotal * 1e3), wall);

    if (o.compare) {
        const bool keep = renderer.denoise;
        renderer.denoise = false;
        renderer.postProcess();
        std::string noisy = o.output;
        const auto dot = noisy.rfind(".png");
        noisy.insert(dot == std::string::npos ? noisy.size() : dot, "_noisy");
        renderer.savePng(noisy);
        std::cout << "  wrote " << noisy << "\n";
        renderer.denoise = keep;
        renderer.postProcess();
    }
    renderer.savePng(o.output);
    std::printf("  wrote %s (denoise %s, last post pass %.2f ms)\n", o.output.c_str(), renderer.denoise ? "on" : "off", renderer.postMs);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = parseOptions(argc, argv);
        if (options.headless && !options.frameSppGiven) options.samplesPerFrame = 16;

        const auto t0 = Clock::now();
        const Scene scene = loadObj(options.scene);
        Context ctx(!options.headless, options.width, options.height);
        Renderer renderer(ctx, scene, options);
        std::printf("%s\n  %u triangles, %zu emissive, loaded + BVH built in %.0f ms on %s\n", options.scene.c_str(),
                    scene.triangleCount(), scene.lights.size(),
                    std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), ctx.deviceName.c_str());

        const glm::vec3 size = scene.boundsMax - scene.boundsMin;
        return options.headless ? runHeadless(ctx, renderer, options) : runWindow(ctx, renderer, glm::length(size));
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
