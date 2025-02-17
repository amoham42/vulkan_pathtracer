#include "buffer.h"


Buffer::Buffer(Context& context, BufferType type, vk::DeviceSize size, const void* data) {
    vk::BufferUsageFlags usage;
    vk::MemoryPropertyFlags memoryProps;
    using Usage = vk::BufferUsageFlagBits;
    using Memory = vk::MemoryPropertyFlagBits;
    switch (type) {
        case BufferType::AccelInput: {
            usage = Usage::eAccelerationStructureBuildInputReadOnlyKHR |
                    Usage::eStorageBuffer |
                    Usage::eShaderDeviceAddress;

            memoryProps = Memory::eHostVisible | Memory::eHostCoherent;
        } break;
        case BufferType::Scratch: {
            usage = Usage::eStorageBuffer |
                    Usage::eShaderDeviceAddress;

            memoryProps = Memory::eDeviceLocal;
        } break;
        case BufferType::ShaderBindingTable: {
            usage = Usage::eShaderBindingTableKHR |
                    Usage::eShaderDeviceAddress;

            memoryProps = Memory::eHostVisible | Memory::eHostCoherent;
        } break;
        case BufferType::AccelStorage: {
            usage = Usage::eAccelerationStructureStorageKHR |
                    Usage::eShaderDeviceAddress;

            memoryProps = Memory::eDeviceLocal;
        }
    }

    buffer = context.device->createBufferUnique({{}, size, usage});

    // Allocate memory
    vk::MemoryRequirements requirements = context.device->getBufferMemoryRequirements(*buffer);
    uint32_t memoryTypeIndex = context.findMemoryType(requirements.memoryTypeBits, memoryProps);

    vk::MemoryAllocateFlagsInfo flagsInfo{vk::MemoryAllocateFlagBits::eDeviceAddress};

    vk::MemoryAllocateInfo memoryInfo;
    memoryInfo.setAllocationSize(requirements.size);
    memoryInfo.setMemoryTypeIndex(memoryTypeIndex);
    memoryInfo.setPNext(&flagsInfo);
    memory = context.device->allocateMemoryUnique(memoryInfo);

    context.device->bindBufferMemory(*buffer, *memory, 0);

    // Get device address
    vk::BufferDeviceAddressInfoKHR bufferDeviceAI{*buffer};
    deviceAddress = context.device->getBufferAddressKHR(&bufferDeviceAI);

    descBufferInfo.setBuffer(*buffer);
    descBufferInfo.setOffset(0);
    descBufferInfo.setRange(size);

    if (data) {
        void* mapped = context.device->mapMemory(*memory, 0, size);
        memcpy(mapped, data, size);
        context.device->unmapMemory(*memory);
    }
}

vk::UniqueBuffer& Buffer::get_buffer() {
    return buffer;
}

uint64_t Buffer::get_address() const {
    return deviceAddress;
}

vk::DescriptorBufferInfo& Buffer::get_bufferInfo() {
    return descBufferInfo;
}