// Vulkan hardware ray-traced path tracer: interactive resizable window or headless image render.

#include "context.h"
#include "options.h"
#include "renderer.h"
#include "scene.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using PS = vk::PipelineStageFlagBits2;
using A = vk::AccessFlagBits2;

/// Swapchain presenter using synchronization2 (vkQueueSubmit2) + fence, supporting window resize.
class Presenter {
public:
    Presenter(const Context& ctx, vk::Extent2D extent) : ctx_(ctx), extent_(extent) {
        acquired_ = ctx_.device.createSemaphore({});
        inFlight_ = ctx_.device.createFence({vk::FenceCreateFlagBits::eSignaled});
        cmd_ = std::move(
            ctx_.device.allocateCommandBuffers({*ctx_.commandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
        createSwapchain(extent_);
    }

    ~Presenter() { ctx_.device.waitIdle(); }

    void recreate(vk::Extent2D newExtent) {
        if (newExtent.width == 0 || newExtent.height == 0) return;
        ctx_.device.waitIdle();
        extent_ = newExtent;
        createSwapchain(extent_);
    }

    /// Renders and presents one frame. Returns false if the swapchain is out-of-date and needs recreation.
    bool frame(Renderer& renderer) {
        if (ctx_.device.waitForFences(*inFlight_, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
            throw std::runtime_error("fence wait failed");

        uint32_t index = 0;
        try {
            auto [res, idx] = swapchain_.acquireNextImage(UINT64_MAX, *acquired_, nullptr);
            if (res == vk::Result::eErrorOutOfDateKHR) return false;
            index = idx;
        } catch (const vk::OutOfDateKHRError&) {
            return false;
        }

        ctx_.device.resetFences(*inFlight_);
        const vk::Image target = images_[index];
        const vk::Semaphore renderDone = *rendered_[index];

        cmd_.reset();
        cmd_.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        renderer.recordFrame(cmd_);
        imageBarrier(cmd_, target, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, PS::eBlit,
                     A::eNone, PS::eBlit, A::eTransferWrite);
        const std::array<vk::Offset3D, 2> srcCorners{
            vk::Offset3D{0, 0, 0}, vk::Offset3D{int32_t(renderer.extent().width), int32_t(renderer.extent().height), 1}};
        const std::array<vk::Offset3D, 2> dstCorners{vk::Offset3D{0, 0, 0},
                                                     vk::Offset3D{int32_t(extent_.width), int32_t(extent_.height), 1}};
        const vk::ImageBlit blit({vk::ImageAspectFlagBits::eColor, 0, 0, 1}, srcCorners,
                                 {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, dstCorners);
        cmd_.blitImage(renderer.ldrImage(), vk::ImageLayout::eGeneral, target, vk::ImageLayout::eTransferDstOptimal,
                       blit, vk::Filter::eNearest);
        imageBarrier(cmd_, target, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR, PS::eBlit,
                     A::eTransferWrite, PS::eNone, A::eNone);
        cmd_.end();

        const vk::SemaphoreSubmitInfo waitInfo(*acquired_, 0, PS::eBlit);
        const vk::CommandBufferSubmitInfo cmdInfo(*cmd_);
        const vk::SemaphoreSubmitInfo sigInfo(renderDone, 0, PS::eAllCommands);
        ctx_.queue.submit2(vk::SubmitInfo2({}, waitInfo, cmdInfo, sigInfo), *inFlight_);

        vk::Result presentRes = vk::Result::eSuccess;
        try {
            const vk::SwapchainKHR sc = *swapchain_;
            presentRes = ctx_.queue.presentKHR(vk::PresentInfoKHR(renderDone, sc, index));
        } catch (const vk::OutOfDateKHRError&) {
            return false;
        }

        if (ctx_.device.waitForFences(*inFlight_, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
            throw std::runtime_error("post-present fence wait failed");
        renderer.finishFrame();
        return presentRes == vk::Result::eSuccess;
    }

private:
    void createSwapchain(vk::Extent2D desired) {
        const auto caps = ctx_.physicalDevice.getSurfaceCapabilitiesKHR(*ctx_.surface);
        const auto formats = ctx_.physicalDevice.getSurfaceFormatsKHR(*ctx_.surface);
        vk::SurfaceFormatKHR format = formats.front();
        for (const auto& f : formats) {
            if ((f.format == vk::Format::eB8G8R8A8Unorm || f.format == vk::Format::eR8G8B8A8Unorm) &&
                f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
                format = f;  // UNORM: tonemap.comp writes sRGB-encoded values
        }

        vk::PresentModeKHR mode = vk::PresentModeKHR::eFifo;
        for (auto m : ctx_.physicalDevice.getSurfacePresentModesKHR(*ctx_.surface))
            if (m == vk::PresentModeKHR::eMailbox) mode = m;

        if (caps.currentExtent.width != UINT32_MAX) {
            extent_ = caps.currentExtent;
        } else {
            extent_.width = std::clamp(desired.width, caps.minImageExtent.width, caps.maxImageExtent.width);
            extent_.height = std::clamp(desired.height, caps.minImageExtent.height, caps.maxImageExtent.height);
        }

        uint32_t count = std::max(caps.minImageCount, 2u);
        if (caps.maxImageCount) count = std::min(count + 1, caps.maxImageCount);

        vk::SwapchainCreateInfoKHR info;
        info.setSurface(*ctx_.surface)
            .setMinImageCount(count)
            .setImageFormat(format.format)
            .setImageColorSpace(format.colorSpace)
            .setImageExtent(extent_)
            .setImageArrayLayers(1)
            .setImageUsage(vk::ImageUsageFlagBits::eTransferDst)
            .setPreTransform(caps.currentTransform)
            .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
            .setPresentMode(mode)
            .setClipped(true)
            .setOldSwapchain(*swapchain_);

        auto newSwapchain = ctx_.device.createSwapchainKHR(info);
        swapchain_ = std::move(newSwapchain);
        images_ = swapchain_.getImages();
        rendered_.clear();
        rendered_.reserve(images_.size());
        for (size_t i = 0; i < images_.size(); i++) rendered_.push_back(ctx_.device.createSemaphore({}));
    }

    const Context& ctx_;
    vk::Extent2D extent_;
    vk::raii::SwapchainKHR swapchain_ = nullptr;
    std::vector<vk::Image> images_;
    vk::raii::Semaphore acquired_ = nullptr;
    std::vector<vk::raii::Semaphore> rendered_;
    vk::raii::Fence inFlight_ = nullptr;
    vk::raii::CommandBuffer cmd_ = nullptr;
};

struct Input {
    Renderer* renderer = nullptr;
    float moveScale = 1.0f;
    bool saveRequested = false;
    bool resized = false;
};

void onKey(GLFWwindow* window, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    auto* in = static_cast<Input*>(glfwGetWindowUserPointer(window));
    Renderer& r = *in->renderer;
    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GLFW_TRUE); break;
        case GLFW_KEY_N: r.setDenoise(!r.denoise()); break;
        case GLFW_KEY_EQUAL:
        case GLFW_KEY_KP_ADD: r.setExposure(r.exposure() * 1.25f); break;
        case GLFW_KEY_MINUS:
        case GLFW_KEY_KP_SUBTRACT: r.setExposure(r.exposure() / 1.25f); break;
        case GLFW_KEY_P: in->saveRequested = true; break;
        default: break;
    }
}

void onResize(GLFWwindow* window, int, int) {
    if (auto* in = static_cast<Input*>(glfwGetWindowUserPointer(window))) in->resized = true;
}

int runWindow(const Context& ctx, Renderer& renderer, float sceneSize) {
    Presenter presenter(ctx, renderer.extent());
    Input input{&renderer, sceneSize};
    glfwSetWindowUserPointer(ctx.window, &input);
    glfwSetKeyCallback(ctx.window, onKey);
    glfwSetFramebufferSizeCallback(ctx.window, onResize);

    auto last = Clock::now(), lastTitle = last;
    double mouseX = 0.0, mouseY = 0.0;
    bool dragging = false;
    int shot = 0;

    while (!glfwWindowShouldClose(ctx.window)) {
        glfwPollEvents();

        int fbW = 0, fbH = 0;
        glfwGetFramebufferSize(ctx.window, &fbW, &fbH);
        if (fbW == 0 || fbH == 0) {
            glfwWaitEvents();
            last = Clock::now();
            continue;
        }
        if (input.resized) {
            input.resized = false;
            const vk::Extent2D next{uint32_t(fbW), uint32_t(fbH)};
            presenter.recreate(next);
            renderer.resize(next);
        }

        const auto now = Clock::now();
        const float dt = std::chrono::duration<float>(now - last).count();
        last = now;

        Camera cam = renderer.camera();
        const auto [fwd, right, up] = cam.basis();
        (void)up;
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

        double x = 0.0, y = 0.0;
        glfwGetCursorPos(ctx.window, &x, &y);
        if (glfwGetMouseButton(ctx.window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            if (dragging) {
                cam.yaw += float(x - mouseX) * 0.004f;
                cam.pitch = glm::clamp(cam.pitch - float(y - mouseY) * 0.004f, -1.55f, 1.55f);
            }
            dragging = true;
        } else {
            dragging = false;
        }
        mouseX = x;
        mouseY = y;
        renderer.setCamera(cam);

        if (!presenter.frame(renderer)) {
            glfwGetFramebufferSize(ctx.window, &fbW, &fbH);
            if (fbW > 0 && fbH > 0) {
                const vk::Extent2D next{uint32_t(fbW), uint32_t(fbH)};
                presenter.recreate(next);
                renderer.resize(next);
            }
            continue;
        }

        if (input.saveRequested) {
            input.saveRequested = false;
            const std::string name = std::format("screenshot_{}.png", shot++);
            renderer.saveImage(name);
            std::println("saved {}", name);
        }
        if (now - lastTitle > std::chrono::milliseconds(250)) {
            lastTitle = now;
            const auto title = std::format("Vulkan Path Tracer  |  {} spp  |  trace {:.1f} ms  denoise {} {:.1f} ms  |  {}",
                                           renderer.samples(), renderer.traceMs(), renderer.denoise() ? "on" : "off",
                                           renderer.postMs(), ctx.deviceName);
            glfwSetWindowTitle(ctx.window, title.c_str());
        }
    }
    return 0;
}

std::string makeNoisyPath(const std::string& output) {
    const std::filesystem::path p(output);
    const auto ext = p.extension().string();
    const auto stem = (p.parent_path() / p.stem()).string();
    return ext.empty() ? (output + "_noisy") : (stem + "_noisy" + ext);
}

int runHeadless(const Context& ctx, Renderer& renderer, const Options& o) {
    // Double-buffered command buffers + fences keep the GPU queue fed with zero inter-launch bubbles.
    auto cmds = ctx.device.allocateCommandBuffers({*ctx.commandPool, vk::CommandBufferLevel::ePrimary, 2});
    std::array<vk::raii::Fence, 2> fences{ctx.device.createFence({}), ctx.device.createFence({})};
    std::array<bool, 2> inFlight{false, false};
    std::array<bool, 2> isFinal{false, false};

    auto retireSlot = [&](uint32_t slot, double& traceTotal) {
        if (!inFlight[slot]) return;
        if (ctx.device.waitForFences(*fences[slot], VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
            throw std::runtime_error("waiting for headless trace failed");
        inFlight[slot] = false;
        if (isFinal[slot]) {
            renderer.finishFrame(slot);
        } else {
            renderer.finishTrace(slot);
        }
        traceTotal += renderer.traceMs();
    };

    uint32_t remaining = o.targetSpp;
    uint32_t launches = 0;
    double traceTotal = 0.0;
    const auto start = Clock::now();
    const auto traceStage = renderer.usingRayQuery() ? PS::eComputeShader : PS::eRayTracingShaderKHR;

    while (remaining > 0) {
        const uint32_t slot = launches & 1u;
        retireSlot(slot, traceTotal);

        const uint32_t batchSpp = std::min(remaining, renderer.samplesPerFrame());
        const bool lastLaunch = (remaining == batchSpp);
        auto& cmd = cmds[slot];
        cmd.reset();
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        renderer.recordTrace(cmd, batchSpp, slot);
        if (lastLaunch) {
            memoryBarrier(cmd, traceStage, A::eShaderStorageWrite, PS::eComputeShader, A::eShaderStorageRead);
            renderer.recordPost(cmd, false, slot);
        }
        cmd.end();

        ctx.device.resetFences(*fences[slot]);
        const vk::CommandBufferSubmitInfo cmdInfo(*cmd);
        ctx.queue.submit2(vk::SubmitInfo2({}, {}, cmdInfo, {}), *fences[slot]);
        inFlight[slot] = true;
        isFinal[slot] = lastLaunch;

        remaining -= batchSpp;
        launches++;
        if (launches % 16 == 0 || remaining == 0) {
            std::print("\r  {} / {} spp", renderer.samples(), o.targetSpp);
            std::fflush(stdout);
        }
    }

    // Drain both slots in submission order.
    const uint32_t lastSlot = (launches - 1u) & 1u;
    retireSlot(lastSlot ^ 1u, traceTotal);
    retireSlot(lastSlot, traceTotal);

    const double wall = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    const double pixels = double(renderer.extent().width) * renderer.extent().height;
    std::println("\n  GPU trace {:.0f} ms total ({:.2f} ms per frame of {} spp), {:.0f} M paths/s, wall {:.0f} ms",
                 traceTotal, traceTotal / launches, renderer.samplesPerFrame(),
                 pixels * renderer.samples() / (traceTotal * 1e3), wall);

    const double finalPostMs = renderer.postMs();
    renderer.saveImage(o.output);
    std::println("  wrote {} (denoise {}, post pass {:.2f} ms)", o.output, renderer.denoise() ? "on" : "off", finalPostMs);

    if (o.compare) {
        const bool keep = renderer.denoise();
        renderer.setDenoise(false);
        renderer.postProcess();
        const std::string noisy = makeNoisyPath(o.output);
        renderer.saveImage(noisy);
        renderer.setDenoise(keep);
        std::println("  wrote {}", noisy);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto parsed = parseOptions(std::span<const char* const>(argv, static_cast<size_t>(argc)));
        if (!parsed) {
            if (parsed.error().showHelp) {
                std::print("{}", usageText());
                return 0;
            }
            std::println(stderr, "error: {}", parsed.error().message);
            std::print(stderr, "{}", usageText());
            return 1;
        }
        const Options& options = *parsed;

        const auto t0 = Clock::now();
        const Scene scene = loadObj(options.scene);
        Context ctx(!options.headless, options.width, options.height, options.validation);
        Renderer renderer(ctx, scene, options);

        const auto vkMajor = VK_API_VERSION_MAJOR(ctx.apiVersion);
        const auto vkMinor = VK_API_VERSION_MINOR(ctx.apiVersion);
        std::println("{}  (Vulkan {}.{}, {}{}{})", options.scene, vkMajor, vkMinor,
                     renderer.usingRayQuery() ? "ray_query" : "rt_pipeline",
                     ctx.hasMaintenance5 ? ", maintenance5" : "", ctx.validationEnabled ? ", validation" : "");
        std::println("  {} triangles ({} unique vertices), {} emissive, loaded + BVH built in {:.0f} ms on {}",
                     scene.triangleCount(), scene.positions.size(), scene.lights.size(),
                     std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), ctx.deviceName);

        const glm::vec3 size = scene.boundsMax - scene.boundsMin;
        return options.headless ? runHeadless(ctx, renderer, options) : runWindow(ctx, renderer, glm::length(size));
    } catch (const std::exception& e) {
        std::println(stderr, "error: {}", e.what());
        return 1;
    }
}
