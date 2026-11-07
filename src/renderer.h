#pragma once

#include "options.h"
#include "resources.h"
#include "scene.h"

#include <glm/glm.hpp>

#include <string>
#include <vector>

struct Camera {
    glm::vec3 eye{0.0f};
    float yaw = 0.0f;    // radians, 0 looks down -Z
    float pitch = 0.0f;  // radians
    float fovY = 39.0f;  // degrees

    glm::vec3 forward() const;
    void lookAt(glm::vec3 target);
};

/// Owns the scene on the GPU, the ray-tracing and post-processing pipelines and every image they
/// touch. One call to record() traces one frame of samples, then denoises and tonemaps into `ldr`.
class Renderer {
public:
    Renderer(Context& ctx, const Scene& scene, const Options& options);

    void record(vk::CommandBuffer cmd);
    /// Re-runs only denoise + tonemap (after changing `denoise` or `exposure`), blocking.
    void postProcess();
    void reset() { frame = 0; }
    /// After the frame's command buffer has completed: read GPU timings and advance the frame count.
    void finishFrame();

    std::vector<uint8_t> readLdr();  // RGBA8, blocking
    void savePng(const std::string& path);

    Camera camera;
    bool denoise = true;
    float exposure = 1.0f;
    uint32_t frame = 0;  // frames accumulated
    uint32_t samplesPerFrame = 4;
    uint32_t maxDepth = 8;
    double traceMs = 0.0, postMs = 0.0;  // GPU time of the last frame

    uint32_t samples() const { return frame * samplesPerFrame; }
    vk::Extent2D extent() const { return extent_; }
    Image ldr;

private:
    void recordPost(vk::CommandBuffer cmd);
    void createImages();
    void createSceneBuffers(const Scene& scene);
    void createRayTracingPipeline();
    void createPostPipelines();
    void createShaderBindingTable();
    void writeDescriptors();

    Context& ctx;
    vk::Extent2D extent_;
    uint32_t lightCount = 0;
    float lightArea = 0.0f;
    float sceneScale = 1.0f;

    Buffer vertexBuffer, indexBuffer, materialBuffer, materialIdBuffer, lightBuffer, instanceBuffer;
    Accel blas, tlas;
    Image accum, pingA, pingB, albedo, normalDepth;

    vk::UniqueDescriptorPool descriptorPool;
    vk::UniqueDescriptorSetLayout rtSetLayout, postSetLayout;
    vk::UniqueDescriptorSet rtSet;
    std::vector<vk::UniqueDescriptorSet> postSets;  // 0: accum->A, 1: A->B, 2: B->A
    vk::UniquePipelineLayout rtLayout, postLayout;
    vk::UniquePipeline rtPipeline, atrousPipeline, tonemapPipeline;

    Buffer sbt;
    vk::StridedDeviceAddressRegionKHR raygenRegion, missRegion, hitRegion, callableRegion;

    vk::UniqueQueryPool timestamps;
    bool timestampsWritten = false;
};
