#pragma once

#include <vulkan/vulkan.hpp>
#include "buffer.h"
#include "context.h"

class AccelerationStructure {
public:
    AccelerationStructure() = default;
    AccelerationStructure(vkContext& context, vk::AccelerationStructureGeometryKHR geometry,
        uint32_t primitiveCount, vk::AccelerationStructureTypeKHR type);

    vk::WriteDescriptorSetAccelerationStructureKHR& get_accelInfo();
    Buffer& get_buffer();
private:
    Buffer buffer;
    vk::UniqueAccelerationStructureKHR accel;
    vk::WriteDescriptorSetAccelerationStructureKHR descAccelInfo;
};
