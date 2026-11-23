#include "resources.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace {

using Usage = vk::BufferUsageFlagBits;
using PS = vk::PipelineStageFlagBits2;
using A = vk::AccessFlagBits2;

struct BufferSpec {
    vk::BufferUsageFlags usage;
    VmaAllocationCreateFlags vmaFlags = 0;
};

BufferSpec specFor(Buffer::Kind kind) noexcept {
    switch (kind) {
        case Buffer::Kind::Scratch:
            return {Usage::eShaderDeviceAddress | Usage::eStorageBuffer, 0};
        case Buffer::Kind::AccelStorage:
            return {Usage::eShaderDeviceAddress | Usage::eAccelerationStructureStorageKHR, 0};
        case Buffer::Kind::GeometryInput:
            return {Usage::eShaderDeviceAddress | Usage::eAccelerationStructureBuildInputReadOnlyKHR |
                        Usage::eStorageBuffer | Usage::eTransferDst,
                    0};
        case Buffer::Kind::Storage:
            return {Usage::eShaderDeviceAddress | Usage::eStorageBuffer | Usage::eTransferDst, 0};
        case Buffer::Kind::Staging:
            return {Usage::eTransferSrc,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        case Buffer::Kind::ShaderBindingTable:
            return {Usage::eShaderDeviceAddress | Usage::eShaderBindingTableKHR,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        case Buffer::Kind::Readback:
            return {Usage::eTransferDst,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    }
    return {};
}

vk::DeviceSize alignUp(vk::DeviceSize value, vk::DeviceSize alignment) noexcept {
    return (value + alignment - 1) & ~(alignment - 1);
}

void recordImageInit(const vk::raii::CommandBuffer& cmd, vk::Image img) {
    imageBarrier(cmd, img, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, PS::eNone, A::eNone,
                 PS::eClear, A::eTransferWrite);
    cmd.clearColorImage(img, vk::ImageLayout::eGeneral, vk::ClearColorValue(std::array{0.f, 0.f, 0.f, 0.f}),
                        vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    imageBarrier(cmd, img, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral, PS::eClear, A::eTransferWrite,
                 PS::eAllCommands, A::eMemoryRead | A::eMemoryWrite);
}

}  // namespace

// ---- Buffer ----

Buffer::Buffer(const Context& ctx, Kind kind, vk::DeviceSize bytes, const void* data)
    : allocator_(ctx.allocator.get()), size_(bytes) {
    const auto spec = specFor(kind);
    vk::BufferCreateInfo bufInfo({}, size_, spec.usage);

    VmaAllocationCreateInfo allocCreate{};
    allocCreate.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreate.flags = spec.vmaFlags;

    VkBuffer rawBuf = VK_NULL_HANDLE;
    VmaAllocationInfo allocInfo{};
    const auto rawCreate = static_cast<VkBufferCreateInfo>(bufInfo);
    if (vmaCreateBuffer(allocator_, &rawCreate, &allocCreate, &rawBuf, &allocation_, &allocInfo) != VK_SUCCESS)
        throw std::runtime_error("vmaCreateBuffer failed");

    buffer_ = rawBuf;
    mapped_ = allocInfo.pMappedData;
    if (spec.usage & Usage::eShaderDeviceAddress) address_ = ctx.device.getBufferAddress({buffer_});

    if (!data) return;
    if (mapped_) {
        std::memcpy(mapped_, data, static_cast<size_t>(size_));
        vmaFlushAllocation(allocator_, allocation_, 0, VK_WHOLE_SIZE);
        return;
    }

    Buffer staging(ctx, Kind::Staging, size_, data);
    ctx.oneTimeSubmit([&](const vk::raii::CommandBuffer& cmd) {
        cmd.copyBuffer(staging.handle(), buffer_, vk::BufferCopy(0, 0, size_));
    });
}

Buffer::~Buffer() { reset(); }

Buffer::Buffer(Buffer&& other) noexcept
    : allocator_(std::exchange(other.allocator_, nullptr)),
      buffer_(std::exchange(other.buffer_, vk::Buffer{})),
      allocation_(std::exchange(other.allocation_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      address_(std::exchange(other.address_, 0)),
      mapped_(std::exchange(other.mapped_, nullptr)) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        reset();
        allocator_ = std::exchange(other.allocator_, nullptr);
        buffer_ = std::exchange(other.buffer_, vk::Buffer{});
        allocation_ = std::exchange(other.allocation_, nullptr);
        size_ = std::exchange(other.size_, 0);
        address_ = std::exchange(other.address_, 0);
        mapped_ = std::exchange(other.mapped_, nullptr);
    }
    return *this;
}

void Buffer::reset() noexcept {
    if (allocator_ && buffer_) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
    }
    allocator_ = nullptr;
    buffer_ = vk::Buffer{};
    allocation_ = nullptr;
    size_ = 0;
    address_ = 0;
    mapped_ = nullptr;
}

void Buffer::flush() const {
    if (allocator_ && allocation_) vmaFlushAllocation(allocator_, allocation_, 0, VK_WHOLE_SIZE);
}

void Buffer::invalidate() const {
    if (allocator_ && allocation_) vmaInvalidateAllocation(allocator_, allocation_, 0, VK_WHOLE_SIZE);
}

// ---- Image ----

Image::Image(const Context& ctx, vk::Extent2D ext, vk::Format fmt, vk::ImageUsageFlags usage,
             const vk::raii::CommandBuffer& cmd)
    : allocator_(ctx.allocator.get()), extent_(ext), format_(fmt) {
    vk::ImageCreateInfo info;
    info.setImageType(vk::ImageType::e2D)
        .setFormat(format_)
        .setExtent({extent_.width, extent_.height, 1})
        .setMipLevels(1)
        .setArrayLayers(1)
        .setUsage(usage);

    VmaAllocationCreateInfo allocCreate{};
    allocCreate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocCreate.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VkImage rawImg = VK_NULL_HANDLE;
    const auto rawCreate = static_cast<VkImageCreateInfo>(info);
    if (vmaCreateImage(allocator_, &rawCreate, &allocCreate, &rawImg, &allocation_, nullptr) != VK_SUCCESS)
        throw std::runtime_error("vmaCreateImage failed");
    image_ = rawImg;

    view_ = ctx.device.createImageView(
        {{}, image_, vk::ImageViewType::e2D, format_, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});
    recordImageInit(cmd, image_);
}

Image::Image(const Context& ctx, vk::Extent2D ext, vk::Format fmt, vk::ImageUsageFlags usage)
    : allocator_(ctx.allocator.get()), extent_(ext), format_(fmt) {
    vk::ImageCreateInfo info;
    info.setImageType(vk::ImageType::e2D)
        .setFormat(format_)
        .setExtent({extent_.width, extent_.height, 1})
        .setMipLevels(1)
        .setArrayLayers(1)
        .setUsage(usage);

    VmaAllocationCreateInfo allocCreate{};
    allocCreate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocCreate.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VkImage rawImg = VK_NULL_HANDLE;
    const auto rawCreate = static_cast<VkImageCreateInfo>(info);
    if (vmaCreateImage(allocator_, &rawCreate, &allocCreate, &rawImg, &allocation_, nullptr) != VK_SUCCESS)
        throw std::runtime_error("vmaCreateImage failed");
    image_ = rawImg;

    view_ = ctx.device.createImageView(
        {{}, image_, vk::ImageViewType::e2D, format_, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});
    ctx.oneTimeSubmit([&](const vk::raii::CommandBuffer& cmd) { recordImageInit(cmd, image_); });
}

Image::~Image() { reset(); }

Image::Image(Image&& other) noexcept
    : allocator_(std::exchange(other.allocator_, nullptr)),
      image_(std::exchange(other.image_, vk::Image{})),
      allocation_(std::exchange(other.allocation_, nullptr)),
      view_(std::move(other.view_)),
      extent_(std::exchange(other.extent_, vk::Extent2D{})),
      format_(std::exchange(other.format_, vk::Format::eUndefined)) {}

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        reset();
        allocator_ = std::exchange(other.allocator_, nullptr);
        image_ = std::exchange(other.image_, vk::Image{});
        allocation_ = std::exchange(other.allocation_, nullptr);
        view_ = std::move(other.view_);
        extent_ = std::exchange(other.extent_, vk::Extent2D{});
        format_ = std::exchange(other.format_, vk::Format::eUndefined);
    }
    return *this;
}

void Image::reset() noexcept {
    // Destroy image view before the underlying VkImage & VmaAllocation.
    view_ = nullptr;
    if (allocator_ && image_) {
        vmaDestroyImage(allocator_, image_, allocation_);
    }
    allocator_ = nullptr;
    image_ = vk::Image{};
    allocation_ = nullptr;
    extent_ = vk::Extent2D{};
    format_ = vk::Format::eUndefined;
}

// ---- Accel ----

Accel::Accel(const Context& ctx, const vk::AccelerationStructureGeometryKHR& geometry, uint32_t primitiveCount,
             vk::AccelerationStructureTypeKHR type) {
    UploadBatch batch(ctx);
    *this = batch.buildAccel(geometry, primitiveCount, type);
    batch.flush();
}

// ---- UploadBatch ----

UploadBatch::UploadBatch(const Context& ctx) : ctx_(ctx) {
    auto cmds = ctx_.device.allocateCommandBuffers({*ctx_.commandPool, vk::CommandBufferLevel::ePrimary, 1});
    cmd_ = std::move(cmds.front());
    cmd_.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
}

Buffer UploadBatch::createBuffer(Buffer::Kind kind, vk::DeviceSize bytes, const void* data) {
    Buffer buf(ctx_, kind, bytes, nullptr);
    if (!data) return buf;
    if (buf.mapped()) {
        std::memcpy(buf.mapped(), data, static_cast<size_t>(bytes));
        vmaFlushAllocation(buf.allocator_, buf.allocation_, 0, VK_WHOLE_SIZE);
        return buf;
    }
    Buffer staging(ctx_, Buffer::Kind::Staging, bytes, data);
    cmd_.copyBuffer(staging.handle(), buf.handle(), vk::BufferCopy(0, 0, bytes));
    transient_.push_back(std::move(staging));
    return buf;
}

Image UploadBatch::createImage(vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage) {
    return Image(ctx_, extent, format, usage, cmd_);
}

Accel UploadBatch::buildAccel(const vk::AccelerationStructureGeometryKHR& geometry, uint32_t primitiveCount,
                              vk::AccelerationStructureTypeKHR type) {
    vk::AccelerationStructureBuildGeometryInfoKHR build;
    build.setType(type)
        .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace)
        .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
        .setGeometries(geometry);

    const auto sizes =
        ctx_.device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, build, primitiveCount);

    Accel accel;
    accel.storage_ = Buffer(ctx_, Buffer::Kind::AccelStorage, sizes.accelerationStructureSize);
    accel.handle_ = ctx_.device.createAccelerationStructureKHR(
        {{}, accel.storage_.handle(), 0, sizes.accelerationStructureSize, type});

    const vk::DeviceSize scratchAlign =
        std::max<vk::DeviceSize>(ctx_.asProperties.minAccelerationStructureScratchOffsetAlignment, 16);
    Buffer scratch(ctx_, Buffer::Kind::Scratch, sizes.buildScratchSize + scratchAlign);
    build.setDstAccelerationStructure(*accel.handle_).setScratchData(alignUp(scratch.address(), scratchAlign));

    const vk::AccelerationStructureBuildRangeInfoKHR range(primitiveCount, 0, 0, 0);
    cmd_.buildAccelerationStructuresKHR(build, &range);
    transient_.push_back(std::move(scratch));

    accel.address_ = ctx_.device.getAccelerationStructureAddressKHR({*accel.handle_});
    return accel;
}

void UploadBatch::memoryBarrier(vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                                vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess) {
    ::memoryBarrier(cmd_, srcStage, srcAccess, dstStage, dstAccess);
}

void UploadBatch::flush() {
    if (flushed_) return;
    flushed_ = true;
    cmd_.end();
    const vk::raii::Fence fence = ctx_.device.createFence({});
    const vk::CommandBufferSubmitInfo cmdInfo(*cmd_);
    const vk::SubmitInfo2 submit({}, {}, cmdInfo, {});
    ctx_.queue.submit2(submit, *fence);
    if (ctx_.device.waitForFences(*fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
        throw std::runtime_error("waiting for UploadBatch failed");
    transient_.clear();
}

// ---- Barriers ----

void memoryBarrier(const vk::raii::CommandBuffer& cmd, vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                   vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess) {
    const vk::MemoryBarrier2 barrier(srcStage, srcAccess, dstStage, dstAccess);
    cmd.pipelineBarrier2(vk::DependencyInfo().setMemoryBarriers(barrier));
}

void imageBarrier(const vk::raii::CommandBuffer& cmd, vk::Image image, vk::ImageLayout from, vk::ImageLayout to,
                  vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess, vk::PipelineStageFlags2 dstStage,
                  vk::AccessFlags2 dstAccess) {
    const vk::ImageMemoryBarrier2 barrier(srcStage, srcAccess, dstStage, dstAccess, from, to, VK_QUEUE_FAMILY_IGNORED,
                                          VK_QUEUE_FAMILY_IGNORED, image, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    cmd.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(barrier));
}
