#include "acceleration_structure.h"

AccelerationStructure::AccelerationStructure(vkContext& context, vk::AccelerationStructureGeometryKHR geometry, uint32_t primitiveCount, vk::AccelerationStructureTypeKHR type) {
    vk::AccelerationStructureBuildGeometryInfoKHR buildGeometryInfo;
    buildGeometryInfo.setType(type);
    buildGeometryInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
    buildGeometryInfo.setGeometries(geometry);

    // Create buffer
    vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo = context.device->getAccelerationStructureBuildSizesKHR(  //
        vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount);
    vk::DeviceSize size = buildSizesInfo.accelerationStructureSize;
    buffer = Buffer(context, Type::AccelStorage, size);

    // Create accel
    vk::AccelerationStructureCreateInfoKHR accelInfo;
    accelInfo.setBuffer(*buffer.get_buffer());
    accelInfo.setSize(size);
    accelInfo.setType(type);
    accel = context.device->createAccelerationStructureKHRUnique(accelInfo);

    // Build
    Buffer scratchBuffer(context, Type::Scratch, buildSizesInfo.buildScratchSize);
    buildGeometryInfo.setScratchData(scratchBuffer.get_address());
    buildGeometryInfo.setDstAccelerationStructure(*accel);

    context.oneTimeSubmit([&](vk::CommandBuffer commandBuffer) {  //
        vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo;
        buildRangeInfo.setPrimitiveCount(primitiveCount);
        buildRangeInfo.setFirstVertex(0);
        buildRangeInfo.setPrimitiveOffset(0);
        buildRangeInfo.setTransformOffset(0);
        commandBuffer.buildAccelerationStructuresKHR(buildGeometryInfo, &buildRangeInfo);
    });

    descAccelInfo.setAccelerationStructures(*accel);
}

vk::WriteDescriptorSetAccelerationStructureKHR& AccelerationStructure::get_accelInfo() {
    return descAccelInfo;
}

Buffer& AccelerationStructure::get_buffer() {
    return buffer;
}