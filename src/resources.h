#pragma once

#include "context.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

/// Move-only VMA-backed GPU buffer. Host-visible kinds (Staging, ShaderBindingTable, Readback)
/// are persistently mapped by VMA.
class Buffer {
public:
    enum class Kind {
        Scratch,             // acceleration-structure build scratch (device-local)
        AccelStorage,        // backing store of a BLAS/TLAS (device-local)
        GeometryInput,       // vertices/indices: read by AS build and shaders (device-local)
        Storage,             // shader-readable storage buffer (device-local)
        Staging,             // host-visible sequential write, CPU -> GPU
        ShaderBindingTable,  // host-visible sequential write, shader binding table
        Readback,            // host-visible cached random read, GPU -> CPU
    };

    Buffer() = default;
    Buffer(const Context& ctx, Kind kind, vk::DeviceSize size, const void* data = nullptr);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    void reset() noexcept;

    vk::Buffer handle() const noexcept { return buffer_; }
    vk::DeviceSize size() const noexcept { return size_; }
    vk::DeviceAddress address() const noexcept { return address_; }
    void* mapped() const noexcept { return mapped_; }
    vk::DescriptorBufferInfo descriptor() const noexcept { return {buffer_, 0, size_}; }

    /// Ensures host writes to a mapped buffer are visible to the GPU, or GPU writes are visible on host.
    void flush() const;
    void invalidate() const;

private:
    friend class UploadBatch;

    VmaAllocator allocator_ = nullptr;
    vk::Buffer buffer_{};
    VmaAllocation allocation_ = nullptr;
    vk::DeviceSize size_ = 0;
    vk::DeviceAddress address_ = 0;
    void* mapped_ = nullptr;
};

/// Move-only VMA-backed 2D storage image, kept in GENERAL layout for its lifetime.
class Image {
public:
    Image() = default;
    Image(const Context& ctx, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    void reset() noexcept;

    vk::Image handle() const noexcept { return image_; }
    vk::ImageView view() const noexcept { return *view_; }
    vk::Extent2D extent() const noexcept { return extent_; }
    vk::Format format() const noexcept { return format_; }
    vk::DescriptorImageInfo descriptor() const noexcept { return {{}, *view_, vk::ImageLayout::eGeneral}; }

private:
    friend class UploadBatch;
    Image(const Context& ctx, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage,
          const vk::raii::CommandBuffer& cmd);

    VmaAllocator allocator_ = nullptr;
    vk::Image image_{};
    VmaAllocation allocation_ = nullptr;
    vk::raii::ImageView view_ = nullptr;
    vk::Extent2D extent_{};
    vk::Format format_ = vk::Format::eUndefined;
};

/// Move-only bottom- or top-level acceleration structure.
class Accel {
public:
    Accel() = default;
    Accel(const Context& ctx, const vk::AccelerationStructureGeometryKHR& geometry, uint32_t primitiveCount,
          vk::AccelerationStructureTypeKHR type);

    vk::AccelerationStructureKHR handle() const noexcept { return *handle_; }
    vk::DeviceAddress address() const noexcept { return address_; }

private:
    friend class UploadBatch;

    // Declaration order: handle destroyed before backing storage.
    Buffer storage_;
    vk::raii::AccelerationStructureKHR handle_ = nullptr;
    vk::DeviceAddress address_ = 0;
};

/// Batches buffer staging copies, image clears/transitions, and acceleration-structure builds into
/// a single command buffer submission + single fence wait.
class UploadBatch {
public:
    explicit UploadBatch(const Context& ctx);

    Buffer createBuffer(Buffer::Kind kind, vk::DeviceSize bytes, const void* data);
    template <typename T>
    Buffer createBuffer(Buffer::Kind kind, std::span<const T> items) {
        return createBuffer(kind, items.size_bytes(), items.data());
    }

    Image createImage(vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage);

    Accel buildAccel(const vk::AccelerationStructureGeometryKHR& geometry, uint32_t primitiveCount,
                     vk::AccelerationStructureTypeKHR type);

    void memoryBarrier(vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                       vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess);

    /// Submits all recorded work via vkQueueSubmit2, waits on a fence, and frees transient staging/scratch buffers.
    void flush();

private:
    const Context& ctx_;
    vk::raii::CommandBuffer cmd_ = nullptr;
    std::vector<Buffer> transient_;
    bool flushed_ = false;
};

/// Barrier helpers (synchronization2).
void memoryBarrier(const vk::raii::CommandBuffer& cmd, vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                   vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess);
void imageBarrier(const vk::raii::CommandBuffer& cmd, vk::Image image, vk::ImageLayout from, vk::ImageLayout to,
                  vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess, vk::PipelineStageFlags2 dstStage,
                  vk::AccessFlags2 dstAccess);

/// Scoped VK_EXT_debug_utils GPU command-buffer marker for Nsight Systems / RenderDoc.
class ScopedGpuLabel {
public:
    ScopedGpuLabel(const Context& ctx, const vk::raii::CommandBuffer& cmd, const char* name,
                   std::array<float, 4> color = {0.2f, 0.6f, 1.0f, 1.0f})
        : cmd_(ctx.hasDebugUtils ? &cmd : nullptr) {
        if (cmd_) cmd_->beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT(name, color));
    }
    ~ScopedGpuLabel() {
        if (cmd_) cmd_->endDebugUtilsLabelEXT();
    }
    ScopedGpuLabel(const ScopedGpuLabel&) = delete;
    ScopedGpuLabel& operator=(const ScopedGpuLabel&) = delete;

private:
    const vk::raii::CommandBuffer* cmd_ = nullptr;
};
