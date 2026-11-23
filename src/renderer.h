#pragma once

#include "options.h"
#include "resources.h"
#include "scene.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

struct Camera {
    glm::vec3 eye{0.0f};
    float yaw = 0.0f;    // radians, 0 looks down -Z
    float pitch = 0.0f;  // radians
    float fovY = 39.0f;  // degrees

    struct Basis {
        glm::vec3 forward;
        glm::vec3 right;
        glm::vec3 up;
    };

    glm::vec3 forward() const noexcept;
    Basis basis() const noexcept;
    void lookAt(glm::vec3 target) noexcept;

    bool operator==(const Camera&) const = default;
};

/// Owns the scene on the GPU, the ray-tracing and post-processing pipelines, and all render targets.
class Renderer {
public:
    Renderer(const Context& ctx, const Scene& scene, const Options& options);

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    /// Records a single ray-tracing pass of `spp` samples (0 = use samplesPerFrame).
    /// `slot` selects timestamp query bank (0 or 1) for double-buffered submissions.
    void recordTrace(const vk::raii::CommandBuffer& cmd, uint32_t spp = 0, uint32_t slot = 0);

    /// Records denoise (if enabled) + tonemap into the LDR target.
    void recordPost(const vk::raii::CommandBuffer& cmd, bool writeTimestamps = false, uint32_t slot = 0);

    /// Interactive convenience: records trace + barrier + post in one command buffer.
    void recordFrame(const vk::raii::CommandBuffer& cmd);

    /// Runs denoise + tonemap in a blocking one-shot submission and updates postMs.
    void postProcess();

    /// Reads GPU timestamps for `slot` after command completion.
    void finishTrace(uint32_t slot = 0);
    void finishFrame(uint32_t slot = 0);

    /// Recreates resolution-dependent render targets and resets accumulation.
    void resize(vk::Extent2D newExtent);

    void reset() noexcept {
        accumulatedSamples_ = 0;
        frameIndex_ = 0;
    }

    const Camera& camera() const noexcept { return camera_; }
    void setCamera(const Camera& cam) noexcept {
        if (cam != camera_) {
            camera_ = cam;
            reset();
        }
    }

    bool denoise() const noexcept { return denoise_; }
    void setDenoise(bool enabled) noexcept { denoise_ = enabled; }
    float exposure() const noexcept { return exposure_; }
    void setExposure(float value) noexcept { exposure_ = value; }
    bool usingRayQuery() const noexcept { return useRayQuery_; }
    uint32_t samplesPerFrame() const noexcept { return samplesPerFrame_; }
    uint32_t samples() const noexcept { return accumulatedSamples_; }
    vk::Extent2D extent() const noexcept { return extent_; }
    vk::Image ldrImage() const noexcept { return ldr_.handle(); }

    double traceMs() const noexcept { return traceMs_; }
    double postMs() const noexcept { return postMs_; }

    /// Writes PNG, JPG, or linear float32 HDR (selected by file extension).
    void saveImage(const std::string& path) const;

private:
    void createImages(UploadBatch& batch);
    void createSceneBuffers(UploadBatch& batch, const Scene& scene);
    void createTracePipeline();
    void createPostPipelines();
    void createShaderBindingTable();
    void createDescriptors();
    void updateRtImageDescriptors();

    std::vector<uint8_t> readLdr() const;
    std::vector<float> readHdrRgb() const;

    const Context& ctx_;
    vk::Extent2D extent_;
    Camera camera_;
    bool useRayQuery_ = false;
    bool denoise_ = true;
    float exposure_ = 1.0f;
    float clampLuminance_ = 64.0f;
    uint32_t maxDepth_ = 8;
    uint32_t samplesPerFrame_ = 4;
    uint32_t accumulatedSamples_ = 0;
    uint32_t frameIndex_ = 0;
    uint32_t lightCount_ = 0;
    float lightArea_ = 0.0f;
    double traceMs_ = 0.0;
    double postMs_ = 0.0;

    Buffer vertexBuffer_, normalBuffer_, indexBuffer_, materialBuffer_, materialIdBuffer_, lightBuffer_, instanceBuffer_;
    Accel blas_, tlas_;
    Image accum_, pingA_, pingB_, albedo_, normalDepth_, ldr_;

    vk::raii::DescriptorPool descriptorPool_ = nullptr;
    vk::raii::DescriptorSetLayout rtSetLayout_ = nullptr;
    vk::raii::DescriptorSetLayout postSetLayout_ = nullptr;
    vk::raii::DescriptorSet rtSet_ = nullptr;
    vk::raii::PipelineLayout rtLayout_ = nullptr;
    vk::raii::PipelineLayout postLayout_ = nullptr;
    vk::raii::Pipeline rtPipeline_ = nullptr;
    vk::raii::Pipeline atrousPass0Pipeline_ = nullptr;
    vk::raii::Pipeline atrousPipeline_ = nullptr;
    vk::raii::Pipeline tonemapPipeline_ = nullptr;
    vk::raii::Pipeline tonemapFp16Pipeline_ = nullptr;

    Buffer sbt_;
    vk::StridedDeviceAddressRegionKHR raygenRegion_{}, missRegion_{}, hitRegion_{}, callableRegion_{};

    vk::raii::QueryPool timestamps_ = nullptr;
};
