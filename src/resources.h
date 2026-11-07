#pragma once

#include "context.h"

/// A buffer and its memory. Device-local buffers are filled through a staging copy, so big meshes
/// live in VRAM instead of being read across PCIe on every ray.
class Buffer {
public:
    enum class Kind {
        Scratch,             // acceleration-structure build scratch
        AccelStorage,        // backing store of a BLAS/TLAS
        GeometryInput,       // vertices/indices: read by the AS build and by shaders
        Storage,             // other shader-readable data (materials, lights)
        ShaderBindingTable,  // host-visible, written once
        Readback,            // host-visible, GPU -> CPU
    };

    Buffer() = default;
    Buffer(const Context& ctx, Kind kind, vk::DeviceSize size, const void* data = nullptr);

    vk::DescriptorBufferInfo descriptor() const { return {*buffer, 0, size}; }
    void* map() const;
    void unmap() const;

    const Context* ctx = nullptr;
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    vk::DeviceSize size = 0;
    vk::DeviceAddress address = 0;
};

/// A 2D storage image, kept in GENERAL layout for its whole life.
class Image {
public:
    Image() = default;
    Image(const Context& ctx, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage);

    vk::DescriptorImageInfo descriptor() const { return {{}, *view, vk::ImageLayout::eGeneral}; }

    vk::UniqueImage image;
    vk::UniqueDeviceMemory memory;
    vk::UniqueImageView view;
    vk::Extent2D extent;
    vk::Format format = vk::Format::eUndefined;
};

/// A bottom- or top-level acceleration structure.
class Accel {
public:
    Accel() = default;
    Accel(const Context& ctx, const vk::AccelerationStructureGeometryKHR& geometry, uint32_t primitiveCount,
          vk::AccelerationStructureTypeKHR type);

    Buffer storage;
    vk::UniqueAccelerationStructureKHR handle;
    vk::DeviceAddress address = 0;
};

/// Barrier helpers (synchronization2).
void memoryBarrier(vk::CommandBuffer cmd, vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                   vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess);
void imageBarrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout from, vk::ImageLayout to,
                  vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess, vk::PipelineStageFlags2 dstStage,
                  vk::AccessFlags2 dstAccess);
