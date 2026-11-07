#include "resources.h"

#include <cstring>

namespace {
using Usage = vk::BufferUsageFlagBits;
using Mem = vk::MemoryPropertyFlagBits;

struct RawBuffer {
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
};

RawBuffer allocate(const Context& ctx, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags props) {
    RawBuffer out;
    out.buffer = ctx.device->createBufferUnique({{}, size, usage});
    const auto req = ctx.device->getBufferMemoryRequirements(*out.buffer);
    vk::MemoryAllocateFlagsInfo flags;
    if (usage & Usage::eShaderDeviceAddress) flags.setFlags(vk::MemoryAllocateFlagBits::eDeviceAddress);
    vk::MemoryAllocateInfo info(req.size, ctx.findMemoryType(req.memoryTypeBits, props), &flags);
    out.memory = ctx.device->allocateMemoryUnique(info);
    ctx.device->bindBufferMemory(*out.buffer, *out.memory, 0);
    return out;
}
}  // namespace

Buffer::Buffer(const Context& context, Kind kind, vk::DeviceSize bytes, const void* data) : ctx(&context), size(bytes) {
    vk::BufferUsageFlags usage = Usage::eShaderDeviceAddress;
    vk::MemoryPropertyFlags props = Mem::eDeviceLocal;
    switch (kind) {
        case Kind::Scratch: usage |= Usage::eStorageBuffer; break;
        case Kind::AccelStorage: usage |= Usage::eAccelerationStructureStorageKHR; break;
        case Kind::GeometryInput:
            usage |= Usage::eAccelerationStructureBuildInputReadOnlyKHR | Usage::eStorageBuffer | Usage::eTransferDst;
            break;
        case Kind::Storage: usage |= Usage::eStorageBuffer | Usage::eTransferDst; break;
        case Kind::ShaderBindingTable:
            usage |= Usage::eShaderBindingTableKHR;
            props = Mem::eHostVisible | Mem::eHostCoherent;
            break;
        case Kind::Readback:
            usage = Usage::eTransferDst;
            props = Mem::eHostVisible | Mem::eHostCoherent | Mem::eHostCached;
            break;
    }

    RawBuffer raw;
    try {
        raw = allocate(context, size, usage, props);
    } catch (const std::runtime_error&) {
        if (kind != Kind::Readback) throw;
        raw = allocate(context, size, usage, Mem::eHostVisible | Mem::eHostCoherent);  // no cached heap
    }
    buffer = std::move(raw.buffer);
    memory = std::move(raw.memory);
    if (usage & Usage::eShaderDeviceAddress) address = context.device->getBufferAddress({*buffer});

    if (!data) return;
    if (props & Mem::eHostVisible) {
        std::memcpy(map(), data, size);
        unmap();
        return;
    }
    // device-local: go through a staging buffer
    RawBuffer staging = allocate(context, size, Usage::eTransferSrc, Mem::eHostVisible | Mem::eHostCoherent);
    void* dst = context.device->mapMemory(*staging.memory, 0, size);
    std::memcpy(dst, data, size);
    context.device->unmapMemory(*staging.memory);
    context.oneTimeSubmit([&](vk::CommandBuffer cmd) { cmd.copyBuffer(*staging.buffer, *buffer, vk::BufferCopy(0, 0, size)); });
}

void* Buffer::map() const { return ctx->device->mapMemory(*memory, 0, size); }
void Buffer::unmap() const { ctx->device->unmapMemory(*memory); }

Image::Image(const Context& ctx, vk::Extent2D size, vk::Format fmt, vk::ImageUsageFlags usage) : extent(size), format(fmt) {
    vk::ImageCreateInfo info;
    info.setImageType(vk::ImageType::e2D)
        .setFormat(format)
        .setExtent({extent.width, extent.height, 1})
        .setMipLevels(1)
        .setArrayLayers(1)
        .setUsage(usage);
    image = ctx.device->createImageUnique(info);

    const auto req = ctx.device->getImageMemoryRequirements(*image);
    memory = ctx.device->allocateMemoryUnique({req.size, ctx.findMemoryType(req.memoryTypeBits, Mem::eDeviceLocal)});
    ctx.device->bindImageMemory(*image, *memory, 0);

    view = ctx.device->createImageViewUnique(
        {{}, *image, vk::ImageViewType::e2D, format, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});

    ctx.oneTimeSubmit([&](vk::CommandBuffer cmd) {
        imageBarrier(cmd, *image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eNone,
                     vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eAllCommands,
                     vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
        // start from black: the accumulation buffer is read before it is first written
        cmd.clearColorImage(*image, vk::ImageLayout::eGeneral, vk::ClearColorValue(std::array{0.f, 0.f, 0.f, 0.f}),
                            vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    });
}

Accel::Accel(const Context& ctx, const vk::AccelerationStructureGeometryKHR& geometry, uint32_t primitiveCount,
             vk::AccelerationStructureTypeKHR type) {
    vk::AccelerationStructureBuildGeometryInfoKHR build;
    build.setType(type)
        .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace)
        .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
        .setGeometries(geometry);

    const auto sizes =
        ctx.device->getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, build, primitiveCount);
    storage = Buffer(ctx, Buffer::Kind::AccelStorage, sizes.accelerationStructureSize);
    handle = ctx.device->createAccelerationStructureKHRUnique({{}, *storage.buffer, 0, sizes.accelerationStructureSize, type});

    const vk::DeviceSize scratchAlign = 256;  // >= minAccelerationStructureScratchOffsetAlignment on current GPUs
    Buffer scratch(ctx, Buffer::Kind::Scratch, sizes.buildScratchSize + scratchAlign);
    build.setDstAccelerationStructure(*handle).setScratchData((scratch.address + scratchAlign - 1) & ~(scratchAlign - 1));

    vk::AccelerationStructureBuildRangeInfoKHR range(primitiveCount, 0, 0, 0);
    ctx.oneTimeSubmit([&](vk::CommandBuffer cmd) { cmd.buildAccelerationStructuresKHR(build, &range); });

    address = ctx.device->getAccelerationStructureAddressKHR({*handle});
}

void memoryBarrier(vk::CommandBuffer cmd, vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                   vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess) {
    vk::MemoryBarrier2 barrier(srcStage, srcAccess, dstStage, dstAccess);
    cmd.pipelineBarrier2(vk::DependencyInfo().setMemoryBarriers(barrier));
}

void imageBarrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout from, vk::ImageLayout to,
                  vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess, vk::PipelineStageFlags2 dstStage,
                  vk::AccessFlags2 dstAccess) {
    vk::ImageMemoryBarrier2 barrier(srcStage, srcAccess, dstStage, dstAccess, from, to, VK_QUEUE_FAMILY_IGNORED,
                                    VK_QUEUE_FAMILY_IGNORED, image, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    cmd.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(barrier));
}
