#include "renderer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {

using Stage = vk::ShaderStageFlagBits;
using DT = vk::DescriptorType;

struct FramePush {
    glm::vec4 eye, right, up, forward;
    uint32_t frame, spp, maxDepth, lightCount;
    float lightArea, clampLuminance;
    uint32_t seed, pad;
};
static_assert(sizeof(FramePush) == 96);

struct PostPush {
    int32_t stepWidth, demodulate;
    float sigmaColor, normalPower, sigmaDepth, exposure;
    int32_t pad0, pad1;
};

std::filesystem::path shaderDir() {
    // next to the executable when installed, else the build tree
    std::error_code ec;
    const auto local = std::filesystem::current_path(ec) / "shaders";
    if (std::filesystem::exists(local / "raygen.rgen.spv", ec)) return local;
    return VPT_SHADER_DIR;
}

vk::UniqueShaderModule loadShader(const Context& ctx, const char* name) {
    const auto path = shaderDir() / (std::string(name) + ".spv");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("missing shader " + path.string() + " (build the 'shaders' target)");
    std::vector<char> code(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    file.read(code.data(), static_cast<std::streamsize>(code.size()));
    return ctx.device->createShaderModuleUnique({{}, code.size(), reinterpret_cast<const uint32_t*>(code.data())});
}

uint32_t alignUp(uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); }

constexpr int kAtrousPasses = 5;

}  // namespace

glm::vec3 Camera::forward() const {
    return {std::cos(pitch) * std::sin(yaw), std::sin(pitch), -std::cos(pitch) * std::cos(yaw)};
}

void Camera::lookAt(glm::vec3 target) {
    const glm::vec3 d = glm::normalize(target - eye);
    pitch = std::asin(glm::clamp(d.y, -1.0f, 1.0f));
    yaw = std::atan2(d.x, -d.z);
}

Renderer::Renderer(Context& context, const Scene& scene, const Options& o)
    : ctx(context), extent_{o.width, o.height} {
    denoise = o.denoise;
    exposure = o.exposure;
    maxDepth = o.maxDepth;
    samplesPerFrame = o.samplesPerFrame;

    // frame the scene unless told where to stand
    const glm::vec3 lo = scene.boundsMin, hi = scene.boundsMax, center = 0.5f * (lo + hi), size = hi - lo;
    sceneScale = glm::length(size);
    camera.fovY = o.fov;
    if (o.eye) {
        camera.eye = *o.eye;
        camera.lookAt(o.target.value_or(center));
    } else {
        const float halfFov = glm::radians(o.fov) * 0.5f;
        // fill the frame with the scene's front face (for a Cornell box: the opening)
        const float fit = 0.5f * std::min(size.y, size.x * float(o.height) / float(o.width)) / std::tan(halfFov);
        camera.eye = center + glm::vec3(0.0f, 0.0f, 0.5f * size.z + fit);
        camera.lookAt(o.target.value_or(center));
    }

    createImages();
    createSceneBuffers(scene);
    createRayTracingPipeline();
    createPostPipelines();
    createShaderBindingTable();
    writeDescriptors();
    timestamps = ctx.device->createQueryPoolUnique({{}, vk::QueryType::eTimestamp, 3});
}

void Renderer::createImages() {
    using U = vk::ImageUsageFlagBits;
    const auto usage = U::eStorage | U::eTransferDst | U::eTransferSrc;
    accum = Image(ctx, extent_, vk::Format::eR32G32B32A32Sfloat, usage);
    pingA = Image(ctx, extent_, vk::Format::eR32G32B32A32Sfloat, usage);
    pingB = Image(ctx, extent_, vk::Format::eR32G32B32A32Sfloat, usage);
    albedo = Image(ctx, extent_, vk::Format::eR16G16B16A16Sfloat, usage);
    normalDepth = Image(ctx, extent_, vk::Format::eR32G32B32A32Sfloat, usage);
    ldr = Image(ctx, extent_, vk::Format::eR8G8B8A8Unorm, usage);
}

void Renderer::createSceneBuffers(const Scene& scene) {
    using K = Buffer::Kind;
    vertexBuffer = Buffer(ctx, K::GeometryInput, sizeof(GpuVertex) * scene.vertices.size(), scene.vertices.data());
    indexBuffer = Buffer(ctx, K::GeometryInput, sizeof(uint32_t) * scene.indices.size(), scene.indices.data());
    materialBuffer = Buffer(ctx, K::Storage, sizeof(GpuMaterial) * scene.materials.size(), scene.materials.data());
    materialIdBuffer = Buffer(ctx, K::Storage, sizeof(uint32_t) * scene.materialIds.size(), scene.materialIds.data());
    // never bind an empty buffer
    const GpuLight none{0, 0.0f};
    lightBuffer = scene.lights.empty() ? Buffer(ctx, K::Storage, sizeof(GpuLight), &none)
                                       : Buffer(ctx, K::Storage, sizeof(GpuLight) * scene.lights.size(), scene.lights.data());
    lightCount = static_cast<uint32_t>(scene.lights.size());
    lightArea = scene.lightArea;

    // bottom level: the whole mesh as one opaque triangle geometry
    vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
    triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat)
        .setVertexData(vertexBuffer.address)
        .setVertexStride(sizeof(GpuVertex))
        .setMaxVertex(static_cast<uint32_t>(scene.vertices.size() - 1))
        .setIndexType(vk::IndexType::eUint32)
        .setIndexData(indexBuffer.address);
    vk::AccelerationStructureGeometryKHR blasGeometry(vk::GeometryTypeKHR::eTriangles, triangles, vk::GeometryFlagBitsKHR::eOpaque);
    blas = Accel(ctx, blasGeometry, scene.triangleCount(), vk::AccelerationStructureTypeKHR::eBottomLevel);

    // top level: one identity instance
    vk::AccelerationStructureInstanceKHR instance;
    const vk::TransformMatrixKHR identity(std::array<std::array<float, 4>, 3>{
        std::array{1.0f, 0.0f, 0.0f, 0.0f}, std::array{0.0f, 1.0f, 0.0f, 0.0f}, std::array{0.0f, 0.0f, 1.0f, 0.0f}});
    instance.setTransform(identity)
        .setMask(0xFF)
        .setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable)
        .setAccelerationStructureReference(blas.address);
    instanceBuffer = Buffer(ctx, K::GeometryInput, sizeof(instance), &instance);
    vk::AccelerationStructureGeometryInstancesDataKHR instances(false, instanceBuffer.address);
    vk::AccelerationStructureGeometryKHR tlasGeometry(vk::GeometryTypeKHR::eInstances, instances, vk::GeometryFlagBitsKHR::eOpaque);
    tlas = Accel(ctx, tlasGeometry, 1, vk::AccelerationStructureTypeKHR::eTopLevel);
}

void Renderer::createRayTracingPipeline() {
    const auto rgen = Stage::eRaygenKHR, chit = Stage::eClosestHitKHR;
    const std::vector<vk::DescriptorSetLayoutBinding> bindings{
        {0, DT::eAccelerationStructureKHR, 1, rgen},  // TLAS
        {1, DT::eStorageImage, 1, rgen},              // accumulated radiance
        {2, DT::eStorageBuffer, 1, rgen | chit},      // vertices
        {3, DT::eStorageBuffer, 1, rgen | chit},      // indices
        {4, DT::eStorageBuffer, 1, rgen},             // materials
        {5, DT::eStorageBuffer, 1, rgen | chit},      // per-triangle material ids
        {6, DT::eStorageBuffer, 1, rgen},             // emissive triangles
        {7, DT::eStorageImage, 1, rgen},              // albedo guide
        {8, DT::eStorageImage, 1, rgen},              // normal + depth guide
    };
    rtSetLayout = ctx.device->createDescriptorSetLayoutUnique({{}, bindings});
    const vk::PushConstantRange push(rgen, 0, sizeof(FramePush));
    rtLayout = ctx.device->createPipelineLayoutUnique({{}, *rtSetLayout, push});

    const auto raygen = loadShader(ctx, "raygen.rgen");
    const auto miss = loadShader(ctx, "miss.rmiss");
    const auto shadowMiss = loadShader(ctx, "shadow.rmiss");
    const auto closestHit = loadShader(ctx, "closesthit.rchit");
    const std::array<vk::PipelineShaderStageCreateInfo, 4> stages{{
        {{}, Stage::eRaygenKHR, *raygen, "main"},
        {{}, Stage::eMissKHR, *miss, "main"},
        {{}, Stage::eMissKHR, *shadowMiss, "main"},
        {{}, Stage::eClosestHitKHR, *closestHit, "main"},
    }};
    using G = vk::RayTracingShaderGroupTypeKHR;
    const std::array<vk::RayTracingShaderGroupCreateInfoKHR, 4> groups{{
        {G::eGeneral, 0, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
        {G::eGeneral, 1, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
        {G::eGeneral, 2, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
        {G::eTrianglesHitGroup, VK_SHADER_UNUSED_KHR, 3, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
    }};
    vk::RayTracingPipelineCreateInfoKHR info;
    info.setStages(stages).setGroups(groups).setMaxPipelineRayRecursionDepth(1).setLayout(*rtLayout);
    auto result = ctx.device->createRayTracingPipelineKHRUnique(nullptr, nullptr, info);
    if (result.result != vk::Result::eSuccess) throw std::runtime_error("ray-tracing pipeline creation failed");
    rtPipeline = std::move(result.value);
}

void Renderer::createPostPipelines() {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    for (uint32_t b = 0; b < 5; b++) bindings.push_back({b, DT::eStorageImage, 1, Stage::eCompute});
    postSetLayout = ctx.device->createDescriptorSetLayoutUnique({{}, bindings});
    const vk::PushConstantRange push(Stage::eCompute, 0, sizeof(PostPush));
    postLayout = ctx.device->createPipelineLayoutUnique({{}, *postSetLayout, push});

    auto make = [&](const char* name) {
        const auto module = loadShader(ctx, name);
        vk::ComputePipelineCreateInfo info({}, {{}, Stage::eCompute, *module, "main"}, *postLayout);
        auto result = ctx.device->createComputePipelineUnique(nullptr, info);
        if (result.result != vk::Result::eSuccess) throw std::runtime_error(std::string("failed to build ") + name);
        return std::move(result.value);
    };
    atrousPipeline = make("atrous.comp");
    tonemapPipeline = make("tonemap.comp");
}

// One buffer holding [raygen][miss, shadow miss][hit], each region aligned for the hardware.
void Renderer::createShaderBindingTable() {
    const auto& rt = ctx.rtProperties;
    const uint32_t handleSize = rt.shaderGroupHandleSize;
    const uint32_t entry = alignUp(handleSize, rt.shaderGroupHandleAlignment);
    const uint32_t base = rt.shaderGroupBaseAlignment;

    raygenRegion = vk::StridedDeviceAddressRegionKHR{0, alignUp(entry, base), alignUp(entry, base)};
    missRegion = vk::StridedDeviceAddressRegionKHR{0, entry, alignUp(2 * entry, base)};
    hitRegion = vk::StridedDeviceAddressRegionKHR{0, entry, alignUp(entry, base)};

    constexpr uint32_t groupCount = 4;
    std::vector<uint8_t> handles(groupCount * handleSize);
    if (ctx.device->getRayTracingShaderGroupHandlesKHR(*rtPipeline, 0, groupCount, handles.size(), handles.data()) !=
        vk::Result::eSuccess)
        throw std::runtime_error("could not read shader group handles");

    const vk::DeviceSize total = raygenRegion.size + missRegion.size + hitRegion.size + base;
    sbt = Buffer(ctx, Buffer::Kind::ShaderBindingTable, total);
    const vk::DeviceAddress start = (sbt.address + base - 1) & ~vk::DeviceAddress(base - 1);
    auto* dst = static_cast<uint8_t*>(sbt.map()) + (start - sbt.address);
    std::memcpy(dst, handles.data() + 0 * handleSize, handleSize);
    std::memcpy(dst + raygenRegion.size, handles.data() + 1 * handleSize, handleSize);
    std::memcpy(dst + raygenRegion.size + entry, handles.data() + 2 * handleSize, handleSize);
    std::memcpy(dst + raygenRegion.size + missRegion.size, handles.data() + 3 * handleSize, handleSize);
    sbt.unmap();

    raygenRegion.deviceAddress = start;
    missRegion.deviceAddress = start + raygenRegion.size;
    hitRegion.deviceAddress = start + raygenRegion.size + missRegion.size;
}

void Renderer::writeDescriptors() {
    const std::array<vk::DescriptorPoolSize, 3> sizes{{
        {DT::eAccelerationStructureKHR, 1},
        {DT::eStorageImage, 3 + 3 * 5},
        {DT::eStorageBuffer, 5},
    }};
    descriptorPool = ctx.device->createDescriptorPoolUnique({vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 4, sizes});

    rtSet = std::move(ctx.device->allocateDescriptorSetsUnique({*descriptorPool, *rtSetLayout}).front());
    const std::array postLayouts{*postSetLayout, *postSetLayout, *postSetLayout};
    postSets = ctx.device->allocateDescriptorSetsUnique({*descriptorPool, postLayouts});

    vk::WriteDescriptorSetAccelerationStructureKHR tlasInfo(*tlas.handle);
    const auto accumInfo = accum.descriptor(), albedoInfo = albedo.descriptor(), ndInfo = normalDepth.descriptor();
    const auto vInfo = vertexBuffer.descriptor(), iInfo = indexBuffer.descriptor(), mInfo = materialBuffer.descriptor(),
               midInfo = materialIdBuffer.descriptor(), lInfo = lightBuffer.descriptor();

    std::vector<vk::WriteDescriptorSet> writes{
        vk::WriteDescriptorSet(*rtSet, 0, 0, 1, DT::eAccelerationStructureKHR).setPNext(&tlasInfo),
        vk::WriteDescriptorSet(*rtSet, 1, 0, DT::eStorageImage, accumInfo),
        vk::WriteDescriptorSet(*rtSet, 2, 0, DT::eStorageBuffer, {}, vInfo),
        vk::WriteDescriptorSet(*rtSet, 3, 0, DT::eStorageBuffer, {}, iInfo),
        vk::WriteDescriptorSet(*rtSet, 4, 0, DT::eStorageBuffer, {}, mInfo),
        vk::WriteDescriptorSet(*rtSet, 5, 0, DT::eStorageBuffer, {}, midInfo),
        vk::WriteDescriptorSet(*rtSet, 6, 0, DT::eStorageBuffer, {}, lInfo),
        vk::WriteDescriptorSet(*rtSet, 7, 0, DT::eStorageImage, albedoInfo),
        vk::WriteDescriptorSet(*rtSet, 8, 0, DT::eStorageImage, ndInfo),
    };

    // ping-pong: accum -> A, A -> B, B -> A
    const std::array<std::pair<const Image*, const Image*>, 3> chain{{{&accum, &pingA}, {&pingA, &pingB}, {&pingB, &pingA}}};
    std::vector<std::array<vk::DescriptorImageInfo, 5>> infos(3);
    for (size_t s = 0; s < 3; s++) {
        infos[s] = {chain[s].first->descriptor(), chain[s].second->descriptor(), albedoInfo, ndInfo, ldr.descriptor()};
        for (uint32_t b = 0; b < 5; b++) writes.emplace_back(*postSets[s], b, 0, DT::eStorageImage, infos[s][b]);
    }
    ctx.device->updateDescriptorSets(writes, nullptr);
}

void Renderer::record(vk::CommandBuffer cmd) {
    using PS = vk::PipelineStageFlagBits2;
    using A = vk::AccessFlagBits2;

    cmd.resetQueryPool(*timestamps, 0, 3);
    cmd.writeTimestamp2(PS::eTopOfPipe, *timestamps, 0);

    // ---- trace ----
    const glm::vec3 f = camera.forward();
    const glm::vec3 r = glm::normalize(glm::cross(f, glm::vec3(0.0f, 1.0f, 0.0f)));
    const glm::vec3 u = glm::cross(r, f);
    FramePush push{};
    push.eye = glm::vec4(camera.eye, std::tan(glm::radians(camera.fovY) * 0.5f));
    push.right = glm::vec4(r, float(extent_.width) / float(extent_.height));
    push.up = glm::vec4(u, 0.0f);
    push.forward = glm::vec4(f, 0.0f);
    push.frame = frame;
    push.spp = samplesPerFrame;
    push.maxDepth = maxDepth;
    push.lightCount = lightCount;
    push.lightArea = lightArea;
    push.clampLuminance = 64.0f;
    push.seed = 0x9E3779B9u;

    memoryBarrier(cmd, PS::eComputeShader | PS::eTransfer, A::eShaderStorageRead | A::eTransferRead, PS::eRayTracingShaderKHR,
                  A::eShaderStorageWrite | A::eShaderStorageRead);
    cmd.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *rtPipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *rtLayout, 0, *rtSet, nullptr);
    cmd.pushConstants(*rtLayout, Stage::eRaygenKHR, 0, sizeof(push), &push);
    cmd.traceRaysKHR(raygenRegion, missRegion, hitRegion, callableRegion, extent_.width, extent_.height, 1);
    cmd.writeTimestamp2(PS::eRayTracingShaderKHR, *timestamps, 1);

    memoryBarrier(cmd, PS::eRayTracingShaderKHR, A::eShaderStorageWrite, PS::eComputeShader, A::eShaderStorageRead);
    recordPost(cmd);
    cmd.writeTimestamp2(PS::eComputeShader, *timestamps, 2);
    timestampsWritten = true;
}

void Renderer::postProcess() {
    ctx.oneTimeSubmit([&](vk::CommandBuffer cmd) { recordPost(cmd); });
}

// denoise (optional) + tonemap into `ldr`
void Renderer::recordPost(vk::CommandBuffer cmd) {
    using PS = vk::PipelineStageFlagBits2;
    using A = vk::AccessFlagBits2;
    const uint32_t gx = (extent_.width + 15) / 16, gy = (extent_.height + 15) / 16;
    PostPush post{};
    post.normalPower = 64.0f;
    post.sigmaDepth = 0.05f;
    post.exposure = exposure;

    uint32_t tonemapSet = 0;
    if (denoise) {
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *atrousPipeline);
        const std::array<uint32_t, kAtrousPasses> sets{0, 1, 2, 1, 2};  // ends in A
        for (int pass = 0; pass < kAtrousPasses; pass++) {
            post.stepWidth = 1 << pass;
            post.demodulate = pass == 0 ? 1 : 0;
            post.sigmaColor = 4.0f;  // in local standard deviations; the variance itself drops pass by pass
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *postLayout, 0, *postSets[sets[pass]], nullptr);
            cmd.pushConstants(*postLayout, Stage::eCompute, 0, sizeof(post), &post);
            cmd.dispatch(gx, gy, 1);
            memoryBarrier(cmd, PS::eComputeShader, A::eShaderStorageWrite, PS::eComputeShader, A::eShaderStorageRead);
        }
        tonemapSet = 1;  // reads A
    }
    post.demodulate = denoise ? 1 : 0;
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *tonemapPipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *postLayout, 0, *postSets[tonemapSet], nullptr);
    cmd.pushConstants(*postLayout, Stage::eCompute, 0, sizeof(post), &post);
    cmd.dispatch(gx, gy, 1);
    memoryBarrier(cmd, PS::eComputeShader, A::eShaderStorageWrite, PS::eTransfer | PS::eComputeShader,
                  A::eTransferRead | A::eShaderStorageRead);
}

void Renderer::finishFrame() {
    if (timestampsWritten) {
        std::array<uint64_t, 3> t{};
        if (ctx.device->getQueryPoolResults(*timestamps, 0, 3, sizeof(t), t.data(), sizeof(uint64_t),
                                            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait) ==
            vk::Result::eSuccess) {
            const double period = ctx.properties.limits.timestampPeriod * 1e-6;  // ns -> ms
            traceMs = double(t[1] - t[0]) * period;
            postMs = double(t[2] - t[1]) * period;
        }
    }
    frame++;
}

std::vector<uint8_t> Renderer::readLdr() {
    const vk::DeviceSize bytes = vk::DeviceSize(extent_.width) * extent_.height * 4;
    Buffer readback(ctx, Buffer::Kind::Readback, bytes);
    ctx.oneTimeSubmit([&](vk::CommandBuffer cmd) {
        vk::BufferImageCopy region(0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0}, {extent_.width, extent_.height, 1});
        cmd.copyImageToBuffer(*ldr.image, vk::ImageLayout::eGeneral, *readback.buffer, region);
    });
    std::vector<uint8_t> pixels(bytes);
    std::memcpy(pixels.data(), readback.map(), bytes);
    readback.unmap();
    return pixels;
}

void Renderer::savePng(const std::string& path) {
    const auto pixels = readLdr();
    if (!stbi_write_png(path.c_str(), int(extent_.width), int(extent_.height), 4, pixels.data(), int(extent_.width * 4)))
        throw std::runtime_error("could not write " + path);
}
