
#include <string>
#include <fstream>
#include <iostream>

#include "src/mesh_loader.h"
#include "src/context.h"
#include "src/wavelet_denoise.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb/stb_image_write.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./main <file name> \n";
        return 0;
    }
    Context context;

    //  ==================== SWAPCHAIN & COMMAND BUFFER ====================
    vk::SwapchainCreateInfoKHR scInfo;
    scInfo.setSurface(*context.surface);
    scInfo.setMinImageCount(3);
    scInfo.setImageFormat(vk::Format::eR8G8B8A8Unorm);
    scInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
    scInfo.setImageExtent({WIDTH, HEIGHT});
    scInfo.setImageArrayLayers(1);
    scInfo.setImageUsage(vk::ImageUsageFlagBits::eTransferDst);
    scInfo.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity);
    scInfo.setPresentMode(vk::PresentModeKHR::eFifo);
    scInfo.setClipped(true);
    scInfo.setQueueFamilyIndices(context.queueFamilyIndex);
    vk::UniqueSwapchainKHR sc = context.device->createSwapchainKHRUnique(scInfo);

    std::vector<vk::Image> scImages = context.device->getSwapchainImagesKHR(*sc);

    vk::CommandBufferAllocateInfo commandBufferInfo;
    commandBufferInfo.setCommandPool(*context.commandPool);
    commandBufferInfo.setCommandBufferCount(static_cast<uint32_t>(scImages.size()));
    std::vector<vk::UniqueCommandBuffer> commandBuffers = context.device->allocateCommandBuffersUnique(commandBufferInfo);

    //  ==================== LOADING IMAGE & OBJECT DATA ====================
    Image outputImage{context,
                      {WIDTH, HEIGHT},
                      vk::Format::eR8G8B8A8Unorm,
                      vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst};

    // Load mesh
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    loadFromFile(vertices, indices, faces, argv[1]);

    Buffer vertexBuffer{context, Buffer::Type::AccelInput, sizeof(Vertex) * vertices.size(), vertices.data()};
    Buffer indexBuffer{context, Buffer::Type::AccelInput, sizeof(uint32_t) * indices.size(), indices.data()};
    Buffer faceBuffer{context, Buffer::Type::AccelInput, sizeof(Face) * faces.size(), faces.data()};

    //  ==================== CREATE TLAS & BLAS ====================
    vk::AccelerationStructureGeometryTrianglesDataKHR triangleData;
    triangleData.setVertexFormat(vk::Format::eR32G32B32Sfloat);
    triangleData.setVertexData(vertexBuffer.deviceAddress);
    triangleData.setVertexStride(sizeof(Vertex));
    triangleData.setMaxVertex(static_cast<uint32_t>(vertices.size()));
    triangleData.setIndexType(vk::IndexType::eUint32);
    triangleData.setIndexData(indexBuffer.deviceAddress);

    vk::AccelerationStructureGeometryKHR triangleGeometry;
    triangleGeometry.setGeometryType(vk::GeometryTypeKHR::eTriangles);
    triangleGeometry.setGeometry({triangleData});
    triangleGeometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    const auto primitiveCount = static_cast<uint32_t>(indices.size() / 3);

    Accel bottomAccel{context, triangleGeometry, primitiveCount, vk::AccelerationStructureTypeKHR::eBottomLevel};

    // Create top level accel struct
    vk::TransformMatrixKHR transformMatrix = std::array{
        std::array{1.0f, 0.0f, 0.0f, 0.0f},
        std::array{0.0f, 1.0f, 0.0f, 0.0f},
        std::array{0.0f, 0.0f, 1.0f, 0.0f},
    };

    vk::AccelerationStructureInstanceKHR accelInstance;
    accelInstance.setTransform(transformMatrix);
    accelInstance.setMask(0xFF);
    accelInstance.setAccelerationStructureReference(bottomAccel.buffer.deviceAddress);
    accelInstance.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

    Buffer instancesBuffer{context, Buffer::Type::AccelInput, sizeof(vk::AccelerationStructureInstanceKHR), &accelInstance};

    vk::AccelerationStructureGeometryInstancesDataKHR instancesData;
    instancesData.setArrayOfPointers(false);
    instancesData.setData(instancesBuffer.deviceAddress);

    vk::AccelerationStructureGeometryKHR instanceGeometry;
    instanceGeometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
    instanceGeometry.setGeometry({instancesData});
    instanceGeometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    Accel topAccel{context, instanceGeometry, 1, vk::AccelerationStructureTypeKHR::eTopLevel};

    //  ==================== SHADERS ====================
    const std::vector<char> raygenCode = readFile("../shaders/raygen.rgen.spv");
    const std::vector<char> missCode = readFile("../shaders/miss.rmiss.spv");
    const std::vector<char> chitCode = readFile("../shaders/closesthit.rchit.spv");

    std::vector<vk::UniqueShaderModule> shaderModules(3);
    shaderModules[0] = context.device->createShaderModuleUnique({{}, raygenCode.size(), reinterpret_cast<const uint32_t*>(raygenCode.data())});
    shaderModules[1] = context.device->createShaderModuleUnique({{}, missCode.size(), reinterpret_cast<const uint32_t*>(missCode.data())});
    shaderModules[2] = context.device->createShaderModuleUnique({{}, chitCode.size(), reinterpret_cast<const uint32_t*>(chitCode.data())});

    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages(3);
    shaderStages[0] = {{}, vk::ShaderStageFlagBits::eRaygenKHR, *shaderModules[0], "main"};
    shaderStages[1] = {{}, vk::ShaderStageFlagBits::eMissKHR, *shaderModules[1], "main"};
    shaderStages[2] = {{}, vk::ShaderStageFlagBits::eClosestHitKHR, *shaderModules[2], "main"};

    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups(3);
    shaderGroups[0] = {vk::RayTracingShaderGroupTypeKHR::eGeneral, 0, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
    shaderGroups[1] = {vk::RayTracingShaderGroupTypeKHR::eGeneral, 1, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
    shaderGroups[2] = {vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup, VK_SHADER_UNUSED_KHR, 2, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};

    //  ==================== PIPELINE & DESCRIPTOR SETS ====================
    std::vector<vk::DescriptorSetLayoutBinding> bindings{
        {0, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eRaygenKHR},  // Binding = 0 : TLAS
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR},              // Binding = 1 : Storage image
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},         // Binding = 2 : Vertices
        {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},         // Binding = 3 : Indices
        {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},         // Binding = 4 : Faces
    };

    vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo;
    descSetLayoutInfo.setBindings(bindings);
    vk::UniqueDescriptorSetLayout descSetLayout = context.device->createDescriptorSetLayoutUnique(descSetLayoutInfo);

    vk::PushConstantRange pushRange;
    pushRange.setOffset(0);
    pushRange.setSize(sizeof(Controls));
    pushRange.setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(*descSetLayout);
    pipelineLayoutInfo.setPushConstantRanges(pushRange);
    vk::UniquePipelineLayout pipelineLayout = context.device->createPipelineLayoutUnique(pipelineLayoutInfo);

    vk::RayTracingPipelineCreateInfoKHR rtPipelineInfo;
    rtPipelineInfo.setStages(shaderStages);
    rtPipelineInfo.setGroups(shaderGroups);
    rtPipelineInfo.setMaxPipelineRayRecursionDepth(4);
    rtPipelineInfo.setLayout(*pipelineLayout);

    auto result = context.device->createRayTracingPipelineKHRUnique(nullptr, nullptr, rtPipelineInfo);
    if (result.result != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create RT pipeline!");
    }

    vk::UniquePipeline pipeline = std::move(result.value);

    //  ==================== MAKE RAYTRACING PROPERTIES ====================
    auto properties = context.physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    auto rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

    uint32_t handleSize = rtProperties.shaderGroupHandleSize;
    uint32_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
    auto groupCount = static_cast<uint32_t>(shaderGroups.size());
    uint32_t sbtSize = groupCount * handleSizeAligned;

    std::vector<uint8_t> handleStorage(sbtSize);
    if (context.device->getRayTracingShaderGroupHandlesKHR(*pipeline, 0, groupCount, sbtSize, handleStorage.data()) != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to process RT group handles!");
    }

    //  ==================== BINDING SHADERS FOR RAYTRACING STRUCTURE ====================
    Buffer raygenSBT{context, Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 0 * handleSizeAligned};
    Buffer missSBT{context, Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 1 * handleSizeAligned};
    Buffer hitSBT{context, Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 2 * handleSizeAligned};

    uint32_t stride = rtProperties.shaderGroupHandleAlignment;
    uint32_t size = rtProperties.shaderGroupHandleAlignment;

    vk::StridedDeviceAddressRegionKHR raygenRegion{raygenSBT.deviceAddress, stride, size};
    vk::StridedDeviceAddressRegionKHR missRegion{missSBT.deviceAddress, stride, size};
    vk::StridedDeviceAddressRegionKHR hitRegion{hitSBT.deviceAddress, stride, size};

    //  ==================== CREATE DESCRIPTOR SETS ====================
    vk::UniqueDescriptorSet descSet = context.allocateDescSet(*descSetLayout);
    std::vector<vk::WriteDescriptorSet> writes(bindings.size());
    for (int i = 0; i < bindings.size(); i++) {
        writes[i].setDstSet(*descSet);
        writes[i].setDescriptorType(bindings[i].descriptorType);
        writes[i].setDescriptorCount(bindings[i].descriptorCount);
        writes[i].setDstBinding(bindings[i].binding);
    }
    writes[0].setPNext(&topAccel.descAccelInfo);
    writes[1].setImageInfo(outputImage.descImageInfo);
    writes[2].setBufferInfo(vertexBuffer.descBufferInfo);
    writes[3].setBufferInfo(indexBuffer.descBufferInfo);
    writes[4].setBufferInfo(faceBuffer.descBufferInfo);
    context.device->updateDescriptorSets(writes, nullptr);

    //  ==================== RUN WINDOW ====================
    uint32_t imageIndex = 0;
    vk::UniqueSemaphore imageAcquiredSemaphore = context.device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
    while (!glfwWindowShouldClose(context.window)) {
        glfwPollEvents();

        // Acquire next image
        imageIndex = context.device->acquireNextImageKHR(*sc, UINT64_MAX, *imageAcquiredSemaphore).value;

        // Record commands
        vk::CommandBuffer commandBuffer = *commandBuffers[imageIndex];
        commandBuffer.begin(vk::CommandBufferBeginInfo());
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *pipelineLayout, 0, *descSet, nullptr);
        commandBuffer.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eRaygenKHR, 0, sizeof(Controls), &context.controls);
        commandBuffer.traceRaysKHR(raygenRegion, missRegion, hitRegion, {}, WIDTH, HEIGHT, 1);

        vk::Image srcImage = *outputImage.image;
        vk::Image dstImage = scImages[imageIndex];
        Image::setImageLayout(commandBuffer, srcImage, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal);
        Image::setImageLayout(commandBuffer, dstImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        Image::copyImage(commandBuffer, srcImage, dstImage);
        Image::setImageLayout(commandBuffer, srcImage, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);
        Image::setImageLayout(commandBuffer, dstImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR);

        commandBuffer.end();

        // Submit
        context.queue.submit(vk::SubmitInfo().setCommandBuffers(commandBuffer));

        // Present image
        vk::PresentInfoKHR presentInfo;
        presentInfo.setSwapchains(*sc);
        presentInfo.setImageIndices(imageIndex);
        presentInfo.setWaitSemaphores(*imageAcquiredSemaphore);
        if (auto result1 = context.queue.presentKHR(presentInfo); result1 != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present.");
        }
        context.queue.waitIdle();
        context.controls.frame++;
    }

    context.device->waitIdle();
    constexpr vk::DeviceSize imageSize = WIDTH * HEIGHT * 4;

    vk::BufferCreateInfo stagingBufferInfo(
        vk::BufferCreateFlags(),
        imageSize,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive
    );
    vk::UniqueBuffer stagingBuffer = context.device->createBufferUnique(stagingBufferInfo);
    vk::MemoryRequirements memRequirements = context.device->getBufferMemoryRequirements(*stagingBuffer);

    uint32_t memType = 0;
    vk::PhysicalDeviceMemoryProperties memProperties = context.physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            ((memProperties.memoryTypes[i].propertyFlags &
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent) ==
                (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent))) {
            memType = i;
        }
    }
    if (memType == 0) {throw std::runtime_error("Failed to find suitable memory type!");}

    vk::MemoryAllocateInfo allocInfo(memRequirements.size, memType);
    vk::UniqueDeviceMemory stagingBufferMemory = context.device->allocateMemoryUnique(allocInfo);

    context.device->bindBufferMemory(*stagingBuffer, *stagingBufferMemory, 0);

    vk::CommandBufferAllocateInfo copyAllocInfo(
        *context.commandPool,
        vk::CommandBufferLevel::ePrimary,
        1
    );
    auto copyCmdBuffers = context.device->allocateCommandBuffersUnique(copyAllocInfo);
    vk::CommandBuffer copyCmdBuffer = *copyCmdBuffers.front();

    copyCmdBuffer.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    Image::setImageLayout(
        copyCmdBuffer,
        *outputImage.image,
        vk::ImageLayout::eGeneral,
        vk::ImageLayout::eTransferSrcOptimal);

    vk::BufferImageCopy region{};
    region.setBufferOffset(0);
    region.setBufferRowLength(0);
    region.setBufferImageHeight(0);
    region.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        region.imageOffset = vk::Offset3D{0, 0, 0};
        region.imageExtent = vk::Extent3D{WIDTH, HEIGHT, 1};


    copyCmdBuffer.copyImageToBuffer(
        *outputImage.image,
        vk::ImageLayout::eTransferSrcOptimal,
        *stagingBuffer,
        region);

    Image::setImageLayout(
        copyCmdBuffer,
        *outputImage.image,
        vk::ImageLayout::eTransferSrcOptimal,
        vk::ImageLayout::eGeneral);

    copyCmdBuffer.end();

    vk::SubmitInfo copySubmitInfo;
    copySubmitInfo.setCommandBuffers(copyCmdBuffer);
    context.queue.submit(copySubmitInfo);
    context.queue.waitIdle();

    if (void* data = context.device->mapMemory(*stagingBufferMemory, 0, imageSize)) {
        waveletDenoiseImage(static_cast<unsigned char*>(data), WIDTH, HEIGHT, 4, 5.0f);
        stbi_write_png("output.png", WIDTH, HEIGHT, 4, data, static_cast<int>(WIDTH * 4));
        context.device->unmapMemory(*stagingBufferMemory);
        std::cout << "Image dumped to output.png" << std::endl;
    } else {
        std::cerr << "Failed to map staging buffer memory." << std::endl;
    }
    glfwDestroyWindow(context.window);
    glfwTerminate();
}
