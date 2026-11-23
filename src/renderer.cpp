#include "renderer.h"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Stage = vk::ShaderStageFlagBits;
using DT = vk::DescriptorType;
using PS = vk::PipelineStageFlagBits2;
using A = vk::AccessFlagBits2;

struct FramePush {
    glm::vec4 eye, right, up, forward;
    uint32_t prevSamples, spp, maxDepth, lightCount;
    float lightArea, clampLuminance;
    uint32_t seed, frameIndex;
};
static_assert(sizeof(FramePush) == 96);
static_assert(offsetof(FramePush, prevSamples) == 64);
static_assert(offsetof(FramePush, lightArea) == 80);

struct PostPush {
    int32_t stepWidth, demodulate;
    float sigmaColor, normalPower, sigmaDepth, exposure;
    uint32_t totalSamples, pad;
};
static_assert(sizeof(PostPush) == 32);
static_assert(offsetof(PostPush, totalSamples) == 24);

std::filesystem::path shaderDir() {
    std::error_code ec;
    const auto exe = executableDir();
    if (!exe.empty()) {
        const auto besideExe = exe / "shaders";
        if (std::filesystem::exists(besideExe / "raygen.rgen.spv", ec)) return besideExe;
    }
    const auto local = std::filesystem::current_path(ec) / "shaders";
    if (std::filesystem::exists(local / "raygen.rgen.spv", ec)) return local;
    return VPT_SHADER_DIR;
}

std::vector<uint32_t> readSpirv(const char* name) {
    const auto path = shaderDir() / (std::string(name) + ".spv");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("missing shader " + path.string() + " (build the 'shaders' target)");
    const auto bytes = static_cast<size_t>(file.tellg());
    if (bytes == 0 || (bytes % 4) != 0) throw std::runtime_error("corrupt SPIR-V file " + path.string());
    std::vector<uint32_t> words(bytes / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(bytes));
    return words;
}

uint32_t alignUp(uint32_t v, uint32_t a) noexcept { return (v + a - 1) & ~(a - 1); }

constexpr int kAtrousPasses = 5;

std::string lowercaseExt(const std::string& path) {
    auto ext = std::filesystem::path(path).extension().string();
    std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

}  // namespace

glm::vec3 Camera::forward() const noexcept {
    return {std::cos(pitch) * std::sin(yaw), std::sin(pitch), -std::cos(pitch) * std::cos(yaw)};
}

Camera::Basis Camera::basis() const noexcept {
    const glm::vec3 f = forward();
    // Avoid degenerate cross product when looking straight up or down.
    const glm::vec3 worldUp = std::abs(f.y) > 0.999f ? glm::vec3(0.0f, 0.0f, -1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::vec3 r = glm::normalize(glm::cross(f, worldUp));
    const glm::vec3 u = glm::cross(r, f);
    return {f, r, u};
}

void Camera::lookAt(glm::vec3 target) noexcept {
    const glm::vec3 diff = target - eye;
    const float len = glm::length(diff);
    if (len < 1e-6f) return;
    const glm::vec3 d = diff / len;
    pitch = std::asin(glm::clamp(d.y, -1.0f, 1.0f));
    yaw = std::atan2(d.x, -d.z);
}

Renderer::Renderer(const Context& context, const Scene& scene, const Options& o)
    : ctx_(context),
      extent_{o.width, o.height},
      useRayQuery_(o.backend == TraceBackend::RayQuery ? true
                   : o.backend == TraceBackend::RayPipeline ? false
                   : context.hasRayQuery),
      denoise_(o.denoise),
      exposure_(o.exposure),
      clampLuminance_(o.clampLuminance.value_or(o.denoise ? 64.0f : 0.0f)),
      maxDepth_(o.maxDepth),
      samplesPerFrame_(o.samplesPerFrame) {
    if (useRayQuery_ && !ctx_.hasRayQuery)
        throw std::runtime_error("--backend rq requested, but this GPU does not support VK_KHR_ray_query");

    // Frame the scene unless told where to stand.
    const glm::vec3 lo = scene.boundsMin, hi = scene.boundsMax, center = 0.5f * (lo + hi), size = hi - lo;
    camera_.fovY = o.fov;
    if (o.eye) {
        camera_.eye = *o.eye;
        camera_.lookAt(o.target.value_or(center));
    } else {
        const float halfFov = glm::radians(o.fov) * 0.5f;
        const float fit = 0.5f * std::min(size.y, size.x * float(o.height) / float(o.width)) / std::tan(halfFov);
        camera_.eye = center + glm::vec3(0.0f, 0.0f, 0.5f * size.z + fit);
        camera_.lookAt(o.target.value_or(center));
    }

    // Batch all image clears, geometry staging uploads, and BLAS/TLAS builds into ONE GPU submission.
    UploadBatch batch(ctx_);
    createImages(batch);
    createSceneBuffers(batch, scene);
    batch.flush();

    createTracePipeline();
    createPostPipelines();
    if (!useRayQuery_) createShaderBindingTable();
    createDescriptors();
    // Two banks of 3 timestamps (slots 0 and 1) for double-buffered GPU submissions.
    timestamps_ = ctx_.device.createQueryPool({{}, vk::QueryType::eTimestamp, 6});
}

void Renderer::createImages(UploadBatch& batch) {
    using U = vk::ImageUsageFlagBits;
    const auto usage = U::eStorage | U::eTransferDst | U::eTransferSrc;
    accum_ = batch.createImage(extent_, vk::Format::eR32G32B32A32Sfloat, usage);
    pingA_ = batch.createImage(extent_, vk::Format::eR32G32B32A32Sfloat, usage);
    pingB_ = batch.createImage(extent_, vk::Format::eR32G32B32A32Sfloat, usage);
    albedo_ = batch.createImage(extent_, vk::Format::eR16G16B16A16Sfloat, usage);
    normalDepth_ = batch.createImage(extent_, vk::Format::eR32G32B32A32Sfloat, usage);
    ldr_ = batch.createImage(extent_, vk::Format::eR8G8B8A8Unorm, usage);
}

void Renderer::createSceneBuffers(UploadBatch& batch, const Scene& scene) {
    using K = Buffer::Kind;
    vertexBuffer_ = batch.createBuffer(K::GeometryInput, std::span<const GpuVertex>(scene.vertices));
    indexBuffer_ = batch.createBuffer(K::GeometryInput, std::span<const uint32_t>(scene.indices));
    materialBuffer_ = batch.createBuffer(K::Storage, std::span<const GpuMaterial>(scene.materials));
    materialIdBuffer_ = batch.createBuffer(K::Storage, std::span<const uint32_t>(scene.materialIds));
    const std::array<GpuLight, 1> fallbackLight{{}};
    lightBuffer_ = scene.lights.empty() ? batch.createBuffer(K::Storage, std::span<const GpuLight>(fallbackLight))
                                        : batch.createBuffer(K::Storage, std::span<const GpuLight>(scene.lights));
    lightCount_ = static_cast<uint32_t>(scene.lights.size());
    lightArea_ = scene.lightArea;

    // Ensure geometry transfers finish before BLAS build reads them.
    batch.memoryBarrier(PS::eCopy, A::eTransferWrite, PS::eAccelerationStructureBuildKHR, A::eShaderRead);

    vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
    triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat)
        .setVertexData(vertexBuffer_.address())
        .setVertexStride(sizeof(GpuVertex))
        .setMaxVertex(static_cast<uint32_t>(scene.vertices.size() - 1))
        .setIndexType(vk::IndexType::eUint32)
        .setIndexData(indexBuffer_.address());
    const vk::AccelerationStructureGeometryKHR blasGeometry(vk::GeometryTypeKHR::eTriangles, triangles,
                                                            vk::GeometryFlagBitsKHR::eOpaque);
    blas_ = batch.buildAccel(blasGeometry, scene.triangleCount(), vk::AccelerationStructureTypeKHR::eBottomLevel);

    // Prepare TLAS instance buffer and synchronize BLAS build + instance upload before TLAS build.
    vk::AccelerationStructureInstanceKHR instance{};
    const vk::TransformMatrixKHR identity(std::array<std::array<float, 4>, 3>{
        std::array{1.0f, 0.0f, 0.0f, 0.0f}, std::array{0.0f, 1.0f, 0.0f, 0.0f}, std::array{0.0f, 0.0f, 1.0f, 0.0f}});
    instance.setTransform(identity)
        .setMask(0xFF)
        .setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable)
        .setAccelerationStructureReference(blas_.address());
    instanceBuffer_ = batch.createBuffer(K::GeometryInput, sizeof(instance), &instance);

    batch.memoryBarrier(PS::eAccelerationStructureBuildKHR | PS::eCopy,
                        A::eAccelerationStructureWriteKHR | A::eTransferWrite, PS::eAccelerationStructureBuildKHR,
                        A::eAccelerationStructureReadKHR | A::eShaderRead);

    const vk::AccelerationStructureGeometryInstancesDataKHR instances(false, instanceBuffer_.address());
    const vk::AccelerationStructureGeometryKHR tlasGeometry(vk::GeometryTypeKHR::eInstances, instances,
                                                            vk::GeometryFlagBitsKHR::eOpaque);
    tlas_ = batch.buildAccel(tlasGeometry, 1, vk::AccelerationStructureTypeKHR::eTopLevel);
}

void Renderer::createTracePipeline() {
    if (useRayQuery_) {
        const auto comp = Stage::eCompute;
        const std::array<vk::DescriptorSetLayoutBinding, 9> bindings{{
            {0, DT::eAccelerationStructureKHR, 1, comp},
            {1, DT::eStorageImage, 1, comp},
            {2, DT::eStorageBuffer, 1, comp},
            {3, DT::eStorageBuffer, 1, comp},
            {4, DT::eStorageBuffer, 1, comp},
            {5, DT::eStorageBuffer, 1, comp},
            {6, DT::eStorageBuffer, 1, comp},
            {7, DT::eStorageImage, 1, comp},
            {8, DT::eStorageImage, 1, comp},
        }};
        rtSetLayout_ = ctx_.device.createDescriptorSetLayout({{}, bindings});
        const vk::PushConstantRange push(comp, 0, sizeof(FramePush));
        rtLayout_ = ctx_.device.createPipelineLayout({{}, *rtSetLayout_, push});

        const auto code = readSpirv("pathtrace.comp");
        if (ctx_.hasMaintenance5) {
            const vk::ShaderModuleCreateInfo modInfo({}, code);
            const auto stage = vk::PipelineShaderStageCreateInfo({}, comp, {}, "main").setPNext(&modInfo);
            rtPipeline_ = ctx_.device.createComputePipeline(nullptr, vk::ComputePipelineCreateInfo({}, stage, *rtLayout_));
        } else {
            const auto mod = ctx_.device.createShaderModule({{}, code});
            const vk::PipelineShaderStageCreateInfo stage({}, comp, *mod, "main");
            rtPipeline_ = ctx_.device.createComputePipeline(nullptr, vk::ComputePipelineCreateInfo({}, stage, *rtLayout_));
        }
        return;
    }

    const auto rgen = Stage::eRaygenKHR, chit = Stage::eClosestHitKHR;
    const std::array<vk::DescriptorSetLayoutBinding, 9> bindings{{
        {0, DT::eAccelerationStructureKHR, 1, rgen},  // TLAS
        {1, DT::eStorageImage, 1, rgen},              // accumulated radiance + E[L^2]
        {2, DT::eStorageBuffer, 1, rgen | chit},      // vertices
        {3, DT::eStorageBuffer, 1, rgen | chit},      // indices
        {4, DT::eStorageBuffer, 1, rgen},             // materials
        {5, DT::eStorageBuffer, 1, rgen | chit},      // per-triangle material ids
        {6, DT::eStorageBuffer, 1, rgen},             // emissive triangles
        {7, DT::eStorageImage, 1, rgen},              // albedo guide
        {8, DT::eStorageImage, 1, rgen},              // normal + depth guide
    }};
    rtSetLayout_ = ctx_.device.createDescriptorSetLayout({{}, bindings});
    const vk::PushConstantRange push(rgen, 0, sizeof(FramePush));
    rtLayout_ = ctx_.device.createPipelineLayout({{}, *rtSetLayout_, push});

    const std::array<std::vector<uint32_t>, 4> code{
        readSpirv("raygen.rgen"),
        readSpirv("miss.rmiss"),
        readSpirv("shadow.rmiss"),
        readSpirv("closesthit.rchit"),
    };
    const std::array<Stage, 4> stageFlags{Stage::eRaygenKHR, Stage::eMissKHR, Stage::eMissKHR, Stage::eClosestHitKHR};

    using G = vk::RayTracingShaderGroupTypeKHR;
    const std::array<vk::RayTracingShaderGroupCreateInfoKHR, 4> groups{{
        {G::eGeneral, 0, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
        {G::eGeneral, 1, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
        {G::eGeneral, 2, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
        {G::eTrianglesHitGroup, VK_SHADER_UNUSED_KHR, 3, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
    }};

    if (ctx_.hasMaintenance5) {
        std::array<vk::ShaderModuleCreateInfo, 4> modInfos;
        std::array<vk::PipelineShaderStageCreateInfo, 4> stages;
        for (size_t i = 0; i < 4; i++) {
            modInfos[i] = vk::ShaderModuleCreateInfo({}, code[i]);
            stages[i] = vk::PipelineShaderStageCreateInfo({}, stageFlags[i], {}, "main").setPNext(&modInfos[i]);
        }
        vk::RayTracingPipelineCreateInfoKHR info;
        info.setStages(stages).setGroups(groups).setMaxPipelineRayRecursionDepth(1).setLayout(*rtLayout_);
        rtPipeline_ = ctx_.device.createRayTracingPipelineKHR(nullptr, nullptr, info);
    } else {
        std::vector<vk::raii::ShaderModule> modules;
        modules.reserve(4);
        std::array<vk::PipelineShaderStageCreateInfo, 4> stages;
        for (size_t i = 0; i < 4; i++) {
            modules.push_back(ctx_.device.createShaderModule({{}, code[i]}));
            stages[i] = vk::PipelineShaderStageCreateInfo({}, stageFlags[i], *modules.back(), "main");
        }
        vk::RayTracingPipelineCreateInfoKHR info;
        info.setStages(stages).setGroups(groups).setMaxPipelineRayRecursionDepth(1).setLayout(*rtLayout_);
        rtPipeline_ = ctx_.device.createRayTracingPipelineKHR(nullptr, nullptr, info);
    }
}

void Renderer::createPostPipelines() {
    std::array<vk::DescriptorSetLayoutBinding, 5> bindings{};
    for (uint32_t b = 0; b < 5; b++) bindings[b] = {b, DT::eStorageImage, 1, Stage::eCompute};
    // Vulkan 1.4-promoted push descriptors: bind ping-pong images inline without pre-allocating descriptor sets.
    postSetLayout_ =
        ctx_.device.createDescriptorSetLayout({vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR, bindings});
    const vk::PushConstantRange push(Stage::eCompute, 0, sizeof(PostPush));
    postLayout_ = ctx_.device.createPipelineLayout({{}, *postSetLayout_, push});

    auto make = [&](const char* name) {
        const auto code = readSpirv(name);
        if (ctx_.hasMaintenance5) {
            const vk::ShaderModuleCreateInfo modInfo({}, code);
            const auto stage = vk::PipelineShaderStageCreateInfo({}, Stage::eCompute, {}, "main").setPNext(&modInfo);
            return ctx_.device.createComputePipeline(nullptr, vk::ComputePipelineCreateInfo({}, stage, *postLayout_));
        }
        const auto mod = ctx_.device.createShaderModule({{}, code});
        const vk::PipelineShaderStageCreateInfo stage({}, Stage::eCompute, *mod, "main");
        return ctx_.device.createComputePipeline(nullptr, vk::ComputePipelineCreateInfo({}, stage, *postLayout_));
    };
    atrousPipeline_ = make("atrous.comp");
    tonemapPipeline_ = make("tonemap.comp");
}

void Renderer::createShaderBindingTable() {
    const auto& rt = ctx_.rtProperties;
    const uint32_t handleSize = rt.shaderGroupHandleSize;
    const uint32_t entry = alignUp(handleSize, rt.shaderGroupHandleAlignment);
    const uint32_t base = rt.shaderGroupBaseAlignment;

    raygenRegion_ = vk::StridedDeviceAddressRegionKHR{0, alignUp(entry, base), alignUp(entry, base)};
    missRegion_ = vk::StridedDeviceAddressRegionKHR{0, entry, alignUp(2 * entry, base)};
    hitRegion_ = vk::StridedDeviceAddressRegionKHR{0, entry, alignUp(entry, base)};

    constexpr uint32_t groupCount = 4;
    const auto handles = rtPipeline_.getRayTracingShaderGroupHandlesKHR<uint8_t>(0, groupCount, groupCount * handleSize);

    const vk::DeviceSize total = raygenRegion_.size + missRegion_.size + hitRegion_.size + base;
    sbt_ = Buffer(ctx_, Buffer::Kind::ShaderBindingTable, total);
    const vk::DeviceAddress start = (sbt_.address() + base - 1) & ~vk::DeviceAddress(base - 1);
    auto* dst = static_cast<uint8_t*>(sbt_.mapped()) + (start - sbt_.address());
    std::memcpy(dst, handles.data() + 0 * handleSize, handleSize);
    std::memcpy(dst + raygenRegion_.size, handles.data() + 1 * handleSize, handleSize);
    std::memcpy(dst + raygenRegion_.size + entry, handles.data() + 2 * handleSize, handleSize);
    std::memcpy(dst + raygenRegion_.size + missRegion_.size, handles.data() + 3 * handleSize, handleSize);
    sbt_.flush();

    raygenRegion_.deviceAddress = start;
    missRegion_.deviceAddress = start + raygenRegion_.size;
    hitRegion_.deviceAddress = start + raygenRegion_.size + missRegion_.size;
}

void Renderer::createDescriptors() {
    const std::array<vk::DescriptorPoolSize, 3> sizes{{
        {DT::eAccelerationStructureKHR, 1},
        {DT::eStorageImage, 3},
        {DT::eStorageBuffer, 5},
    }};
    descriptorPool_ = ctx_.device.createDescriptorPool({vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, sizes});
    rtSet_ = std::move(ctx_.device.allocateDescriptorSets({*descriptorPool_, *rtSetLayout_}).front());

    const auto tlasHandle = tlas_.handle();
    const vk::WriteDescriptorSetAccelerationStructureKHR tlasInfo(tlasHandle);
    const auto vInfo = vertexBuffer_.descriptor();
    const auto iInfo = indexBuffer_.descriptor();
    const auto mInfo = materialBuffer_.descriptor();
    const auto midInfo = materialIdBuffer_.descriptor();
    const auto lInfo = lightBuffer_.descriptor();
    const auto accumInfo = accum_.descriptor();
    const auto albedoInfo = albedo_.descriptor();
    const auto ndInfo = normalDepth_.descriptor();

    const std::array<vk::WriteDescriptorSet, 9> writes{{
        vk::WriteDescriptorSet(*rtSet_, 0, 0, 1, DT::eAccelerationStructureKHR).setPNext(&tlasInfo),
        vk::WriteDescriptorSet(*rtSet_, 1, 0, DT::eStorageImage, accumInfo),
        vk::WriteDescriptorSet(*rtSet_, 2, 0, DT::eStorageBuffer, {}, vInfo),
        vk::WriteDescriptorSet(*rtSet_, 3, 0, DT::eStorageBuffer, {}, iInfo),
        vk::WriteDescriptorSet(*rtSet_, 4, 0, DT::eStorageBuffer, {}, mInfo),
        vk::WriteDescriptorSet(*rtSet_, 5, 0, DT::eStorageBuffer, {}, midInfo),
        vk::WriteDescriptorSet(*rtSet_, 6, 0, DT::eStorageBuffer, {}, lInfo),
        vk::WriteDescriptorSet(*rtSet_, 7, 0, DT::eStorageImage, albedoInfo),
        vk::WriteDescriptorSet(*rtSet_, 8, 0, DT::eStorageImage, ndInfo),
    }};
    ctx_.device.updateDescriptorSets(writes, nullptr);
}

void Renderer::updateRtImageDescriptors() {
    const auto accumInfo = accum_.descriptor();
    const auto albedoInfo = albedo_.descriptor();
    const auto ndInfo = normalDepth_.descriptor();
    const std::array<vk::WriteDescriptorSet, 3> writes{{
        vk::WriteDescriptorSet(*rtSet_, 1, 0, DT::eStorageImage, accumInfo),
        vk::WriteDescriptorSet(*rtSet_, 7, 0, DT::eStorageImage, albedoInfo),
        vk::WriteDescriptorSet(*rtSet_, 8, 0, DT::eStorageImage, ndInfo),
    }};
    ctx_.device.updateDescriptorSets(writes, nullptr);
}

void Renderer::resize(vk::Extent2D newExtent) {
    if (newExtent.width == 0 || newExtent.height == 0 || newExtent == extent_) return;
    ctx_.device.waitIdle();
    extent_ = newExtent;
    UploadBatch batch(ctx_);
    createImages(batch);
    batch.flush();
    updateRtImageDescriptors();
    reset();
}

void Renderer::recordTrace(const vk::raii::CommandBuffer& cmd, uint32_t spp, uint32_t slot) {
    const uint32_t launchSpp = spp > 0 ? spp : samplesPerFrame_;
    const uint32_t qBase = (slot & 1u) * 3u;
    cmd.resetQueryPool(*timestamps_, qBase, 3);
    cmd.writeTimestamp2(PS::eTopOfPipe, *timestamps_, qBase + 0);

    const auto [f, r, u] = camera_.basis();
    const FramePush push{
        .eye = glm::vec4(camera_.eye, std::tan(glm::radians(camera_.fovY) * 0.5f)),
        .right = glm::vec4(r, float(extent_.width) / float(extent_.height)),
        .up = glm::vec4(u, 0.0f),
        .forward = glm::vec4(f, 0.0f),
        .prevSamples = accumulatedSamples_,
        .spp = launchSpp,
        .maxDepth = maxDepth_,
        .lightCount = lightCount_,
        .lightArea = lightArea_,
        .clampLuminance = clampLuminance_,
        .seed = 0x9E3779B9u,
        .frameIndex = frameIndex_,
    };

    accumulatedSamples_ += launchSpp;
    frameIndex_++;

    {
        const ScopedGpuLabel label(ctx_, cmd, useRayQuery_ ? "PathTrace (RayQuery)" : "PathTrace (RTPipeline)",
                                   {0.2f, 0.8f, 0.3f, 1.0f});
        const auto traceStage = useRayQuery_ ? PS::eComputeShader : PS::eRayTracingShaderKHR;
        memoryBarrier(cmd, PS::eRayTracingShaderKHR | PS::eComputeShader | PS::eTransfer,
                      A::eShaderStorageWrite | A::eShaderStorageRead | A::eTransferRead, traceStage,
                      A::eShaderStorageWrite | A::eShaderStorageRead);
        if (useRayQuery_) {
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *rtPipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *rtLayout_, 0, *rtSet_, nullptr);
            cmd.pushConstants<FramePush>(*rtLayout_, Stage::eCompute, 0, push);
            cmd.dispatch((extent_.width + 7) / 8, (extent_.height + 7) / 8, 1);
        } else {
            cmd.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *rtPipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *rtLayout_, 0, *rtSet_, nullptr);
            cmd.pushConstants<FramePush>(*rtLayout_, Stage::eRaygenKHR, 0, push);
            cmd.traceRaysKHR(raygenRegion_, missRegion_, hitRegion_, callableRegion_, extent_.width, extent_.height, 1);
        }
    }
    cmd.writeTimestamp2(useRayQuery_ ? PS::eComputeShader : PS::eRayTracingShaderKHR, *timestamps_, qBase + 1);
}

void Renderer::recordPost(const vk::raii::CommandBuffer& cmd, bool writeTimestamps, uint32_t slot) {
    const uint32_t qBase = (slot & 1u) * 3u;
    const uint32_t gx = (extent_.width + 15) / 16, gy = (extent_.height + 15) / 16;
    const auto albInfo = albedo_.descriptor();
    const auto ndInfo = normalDepth_.descriptor();
    const auto ldrInfo = ldr_.descriptor();

    auto pushBindings = [&](const Image& srcImg, const Image& dstImg) {
        const auto srcInfo = srcImg.descriptor();
        const auto dstInfo = dstImg.descriptor();
        const std::array<vk::WriteDescriptorSet, 5> writes{{
            vk::WriteDescriptorSet({}, 0, 0, DT::eStorageImage, srcInfo),
            vk::WriteDescriptorSet({}, 1, 0, DT::eStorageImage, dstInfo),
            vk::WriteDescriptorSet({}, 2, 0, DT::eStorageImage, albInfo),
            vk::WriteDescriptorSet({}, 3, 0, DT::eStorageImage, ndInfo),
            vk::WriteDescriptorSet({}, 4, 0, DT::eStorageImage, ldrInfo),
        }};
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *postLayout_, 0, writes);
    };

    if (writeTimestamps) {
        cmd.resetQueryPool(*timestamps_, qBase + 1, 2);
        cmd.writeTimestamp2(PS::eTopOfPipe, *timestamps_, qBase + 1);
    }

    PostPush post{
        .stepWidth = 1,
        .demodulate = 0,
        .sigmaColor = 4.0f,
        .normalPower = 64.0f,
        .sigmaDepth = 0.05f,
        .exposure = exposure_,
        .totalSamples = std::max(1u, accumulatedSamples_),
        .pad = 0,
    };

    const Image* tonemapSrc = &accum_;
    if (denoise_) {
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *atrousPipeline_);
        const std::array<std::pair<const Image*, const Image*>, kAtrousPasses> passes{{
            {&accum_, &pingA_},
            {&pingA_, &pingB_},
            {&pingB_, &pingA_},
            {&pingA_, &pingB_},
            {&pingB_, &pingA_},
        }};
        static constexpr std::array<const char*, kAtrousPasses> kPassNames{
            "Atrous Pass 0 (step 1)", "Atrous Pass 1 (step 2)", "Atrous Pass 2 (step 4)", "Atrous Pass 3 (step 8)",
            "Atrous Pass 4 (step 16)"};
        for (int pass = 0; pass < kAtrousPasses; pass++) {
            const ScopedGpuLabel label(ctx_, cmd, kPassNames[pass], {0.9f, 0.6f, 0.1f, 1.0f});
            post.stepWidth = 1 << pass;
            post.demodulate = pass == 0 ? 1 : 0;
            pushBindings(*passes[pass].first, *passes[pass].second);
            cmd.pushConstants<PostPush>(*postLayout_, Stage::eCompute, 0, post);
            cmd.dispatch(gx, gy, 1);
            memoryBarrier(cmd, PS::eComputeShader, A::eShaderStorageWrite, PS::eComputeShader, A::eShaderStorageRead);
        }
        tonemapSrc = &pingA_;
    }

    {
        const ScopedGpuLabel label(ctx_, cmd, "Tonemap", {0.7f, 0.3f, 0.9f, 1.0f});
        post.demodulate = denoise_ ? 1 : 0;
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *tonemapPipeline_);
        pushBindings(*tonemapSrc, pingB_);
        cmd.pushConstants<PostPush>(*postLayout_, Stage::eCompute, 0, post);
        cmd.dispatch(gx, gy, 1);
        memoryBarrier(cmd, PS::eComputeShader, A::eShaderStorageWrite, PS::eTransfer | PS::eComputeShader,
                      A::eTransferRead | A::eShaderStorageRead);
    }

    cmd.writeTimestamp2(PS::eComputeShader, *timestamps_, qBase + 2);
}

void Renderer::recordFrame(const vk::raii::CommandBuffer& cmd) {
    recordTrace(cmd, 0, 0);
    memoryBarrier(cmd, useRayQuery_ ? PS::eComputeShader : PS::eRayTracingShaderKHR, A::eShaderStorageWrite,
                  PS::eComputeShader, A::eShaderStorageRead);
    recordPost(cmd, false, 0);
}

void Renderer::postProcess() {
    ctx_.oneTimeSubmit([&](const vk::raii::CommandBuffer& cmd) {
        memoryBarrier(cmd, PS::eRayTracingShaderKHR | PS::eComputeShader, A::eShaderStorageWrite, PS::eComputeShader,
                      A::eShaderStorageRead);
        recordPost(cmd, true, 0);
    });
    const auto [res, t] = timestamps_.getResults<uint64_t>(
        1, 2, 2 * sizeof(uint64_t), sizeof(uint64_t), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    if (res == vk::Result::eSuccess) {
        const double period = ctx_.properties.limits.timestampPeriod * 1e-6;
        postMs_ = double(t[1] - t[0]) * period;
    }
}

void Renderer::finishTrace(uint32_t slot) {
    const uint32_t qBase = (slot & 1u) * 3u;
    const auto [res, t] = timestamps_.getResults<uint64_t>(
        qBase, 2, 2 * sizeof(uint64_t), sizeof(uint64_t), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    if (res == vk::Result::eSuccess) {
        const double period = ctx_.properties.limits.timestampPeriod * 1e-6;
        traceMs_ = double(t[1] - t[0]) * period;
    }
}

void Renderer::finishFrame(uint32_t slot) {
    const uint32_t qBase = (slot & 1u) * 3u;
    const auto [res, t] = timestamps_.getResults<uint64_t>(
        qBase, 3, 3 * sizeof(uint64_t), sizeof(uint64_t), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    if (res == vk::Result::eSuccess) {
        const double period = ctx_.properties.limits.timestampPeriod * 1e-6;
        traceMs_ = double(t[1] - t[0]) * period;
        postMs_ = double(t[2] - t[1]) * period;
    }
}

std::vector<uint8_t> Renderer::readLdr() const {
    const vk::DeviceSize bytes = vk::DeviceSize(extent_.width) * extent_.height * 4;
    Buffer readback(ctx_, Buffer::Kind::Readback, bytes);
    ctx_.oneTimeSubmit([&](const vk::raii::CommandBuffer& cmd) {
        const ScopedGpuLabel label(ctx_, cmd, "Readback LDR", {0.4f, 0.7f, 0.9f, 1.0f});
        const vk::BufferImageCopy region(0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0},
                                         {extent_.width, extent_.height, 1});
        cmd.copyImageToBuffer(ldr_.handle(), vk::ImageLayout::eGeneral, readback.handle(), region);
    });
    readback.invalidate();
    std::vector<uint8_t> pixels(static_cast<size_t>(bytes));
    std::memcpy(pixels.data(), readback.mapped(), static_cast<size_t>(bytes));
    return pixels;
}

std::vector<float> Renderer::readHdrRgb() const {
    const size_t pixelCount = size_t(extent_.width) * extent_.height;
    const vk::DeviceSize hdrBytes = vk::DeviceSize(pixelCount) * 4 * sizeof(float);
    const Image& src = denoise_ ? pingA_ : accum_;
    Buffer hdrBuf(ctx_, Buffer::Kind::Readback, hdrBytes);
    Buffer albBuf;
    if (denoise_) albBuf = Buffer(ctx_, Buffer::Kind::Readback, vk::DeviceSize(pixelCount) * 4 * sizeof(uint16_t));

    ctx_.oneTimeSubmit([&](const vk::raii::CommandBuffer& cmd) {
        const vk::BufferImageCopy region(0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0},
                                         {extent_.width, extent_.height, 1});
        cmd.copyImageToBuffer(src.handle(), vk::ImageLayout::eGeneral, hdrBuf.handle(), region);
        if (denoise_) cmd.copyImageToBuffer(albedo_.handle(), vk::ImageLayout::eGeneral, albBuf.handle(), region);
    });
    hdrBuf.invalidate();
    const auto* rgba = static_cast<const float*>(hdrBuf.mapped());

    auto f16ToF32 = [](uint16_t h) -> float {
        const uint32_t sign = (uint32_t(h & 0x8000u)) << 16;
        uint32_t exp = (h >> 10) & 0x1Fu;
        uint32_t mant = h & 0x03FFu;
        uint32_t bits = 0;
        if (exp == 0) {
            if (mant == 0) bits = sign;
            else {
                exp = 127 - 14;
                while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
                mant &= 0x03FFu;
                bits = sign | (exp << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7F800000u | (mant << 13);
        } else {
            bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
        return std::bit_cast<float>(bits);
    };

    const uint16_t* alb = denoise_ ? static_cast<const uint16_t*>(albBuf.mapped()) : nullptr;
    if (denoise_) albBuf.invalidate();

    std::vector<float> rgb(pixelCount * 3);
    for (size_t i = 0; i < pixelCount; i++) {
        for (size_t c = 0; c < 3; c++) {
            float v = rgba[4 * i + c] * exposure_;
            if (alb) v *= std::max(f16ToF32(alb[4 * i + c]), 1e-3f);
            rgb[3 * i + c] = std::max(v, 0.0f);
        }
    }
    return rgb;
}

void Renderer::saveImage(const std::string& path) const {
    const auto ext = lowercaseExt(path);
    int ok = 0;
    if (ext == ".hdr") {
        const auto rgb = readHdrRgb();
        ok = stbi_write_hdr(path.c_str(), int(extent_.width), int(extent_.height), 3, rgb.data());
    } else if (ext == ".jpg" || ext == ".jpeg") {
        const auto pixels = readLdr();
        ok = stbi_write_jpg(path.c_str(), int(extent_.width), int(extent_.height), 4, pixels.data(), 95);
    } else {
        const auto pixels = readLdr();
        ok = stbi_write_png(path.c_str(), int(extent_.width), int(extent_.height), 4, pixels.data(),
                            int(extent_.width * 4));
    }
    if (!ok) throw std::runtime_error("could not write " + path);
}
