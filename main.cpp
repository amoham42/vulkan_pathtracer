
#include <string>
#include <fstream>
#include <iostream>
#include <functional>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include "src/mesh_loader.h"
#include <glm/glm.hpp>

#include "src/acceleration_structure.h"
#include "src/image.h"
#include "src/buffer.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

constexpr int WIDTH = 1200;
constexpr int HEIGHT = 1200;

struct Controls {
    glm::vec3 cameraPosition = glm::vec3(0, -1, 5);
    float fov = 45.0f;
    float light_intensity = 1.0f;
    int frame = 0;
    int accumulate = 0;
};

// The key callback function
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // Only process key presses or repeats.
    if (action != GLFW_PRESS && action != GLFW_REPEAT)
        return;

    // Retrieve our controls instance from the window's user pointer.
    Controls* controls = reinterpret_cast<Controls*>(glfwGetWindowUserPointer(window));
    if (!controls) return;

    const float moveSpeed = 0.1f;
    const float fovStep = 5.0f;

    switch (key) {
        case GLFW_KEY_W: controls->cameraPosition.z -= moveSpeed; break;
        case GLFW_KEY_S: controls->cameraPosition.z += moveSpeed; break;
        case GLFW_KEY_A: controls->cameraPosition.x -= moveSpeed; break;
        case GLFW_KEY_D: controls->cameraPosition.x += moveSpeed; break;
        case GLFW_KEY_Q: controls->cameraPosition.y -= moveSpeed; break;
        case GLFW_KEY_E: controls->cameraPosition.y += moveSpeed; break;
        case GLFW_KEY_R: controls->fov += fovStep; break;
        case GLFW_KEY_F: controls->fov -= fovStep; break;
        case GLFW_KEY_SPACE: controls->accumulate = 0; break;
        case GLFW_KEY_B: controls->accumulate = 1; break;
        case GLFW_KEY_UP: controls->light_intensity += 0.1f; break;
        case GLFW_KEY_DOWN: controls->light_intensity -= 0.1f; break;
        default: break;
    }

    controls->frame = 0;
    std::cout << "Accumulate: " << controls->accumulate << std::endl;

}

struct Context {
    Context() {
        // Create window
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Tracing", nullptr, nullptr);

        glfwSetWindowUserPointer(window, &controls);
        glfwSetKeyCallback(window, key_callback);

        // Prepase extensions and layers
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        std::vector layers{"VK_LAYER_KHRONOS_validation"};

        auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

        // Create instance
        vk::ApplicationInfo appInfo;
        appInfo.setApiVersion(VK_API_VERSION_1_3);

        vk::InstanceCreateInfo instanceInfo;
        instanceInfo.setPApplicationInfo(&appInfo);
        instanceInfo.setPEnabledLayerNames(layers);
        instanceInfo.setPEnabledExtensionNames(extensions);
        instance = vk::createInstanceUnique(instanceInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

        // Pick first gpu
        physicalDevice = instance->enumeratePhysicalDevices().front();

        // Create debug messenger
        vk::DebugUtilsMessengerCreateInfoEXT messageInfo;
        messageInfo.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        messageInfo.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
        messageInfo.setPfnUserCallback(&debugUtilsMessengerCallback);
        messenger = instance->createDebugUtilsMessengerEXTUnique(messageInfo);

        // Create surface
        VkSurfaceKHR _surface;
        VkResult res = glfwCreateWindowSurface(VkInstance(*instance), window, nullptr, &_surface);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(_surface), {*instance});

        // Find queue family
        std::vector queueFamilies = physicalDevice.getQueueFamilyProperties();
        for (int i = 0; i < queueFamilies.size(); i++) {
            auto supportCompute = queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute;
            auto supportPresent = physicalDevice.getSurfaceSupportKHR(i, *surface);
            if (supportCompute && supportPresent) {
                queueFamilyIndex = i;
            }
        }

        // Create device
        const float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo queueCreateInfo;
        queueCreateInfo.setQueueFamilyIndex(queueFamilyIndex);
        queueCreateInfo.setQueuePriorities(queuePriority);

        const std::vector deviceExtensions{
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
            VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
            VK_KHR_MAINTENANCE3_EXTENSION_NAME,
            VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        };

        if (!checkDeviceExtensionSupport(deviceExtensions)) {
            throw std::runtime_error("Missing extension support!");
        }

        vk::DeviceCreateInfo deviceInfo;
        deviceInfo.setQueueCreateInfos(queueCreateInfo);
        deviceInfo.setPEnabledExtensionNames(deviceExtensions);

        vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{true};
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{true};
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{true};
        vk::StructureChain createInfoChain{
            deviceInfo,
            bufferDeviceAddressFeatures,
            rayTracingPipelineFeatures,
            accelerationStructureFeatures,
        };

        device = physicalDevice.createDeviceUnique(createInfoChain.get<vk::DeviceCreateInfo>());
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);

        queue = device->getQueue(queueFamilyIndex, 0);

        // Create command pool
        vk::CommandPoolCreateInfo commandPoolInfo;
        commandPoolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        commandPoolInfo.setQueueFamilyIndex(queueFamilyIndex);
        commandPool = device->createCommandPoolUnique(commandPoolInfo);

        // Create descriptor pool
        std::vector<vk::DescriptorPoolSize> poolSizes{
            {vk::DescriptorType::eAccelerationStructureKHR, 1},
            {vk::DescriptorType::eStorageImage, 1},
            {vk::DescriptorType::eStorageBuffer, 3},
        };

        vk::DescriptorPoolCreateInfo descPoolInfo;
        descPoolInfo.setPoolSizes(poolSizes);
        descPoolInfo.setMaxSets(1);
        descPoolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
        descPool = device->createDescriptorPoolUnique(descPoolInfo);
    }

    bool checkDeviceExtensionSupport(const std::vector<const char*>& requiredExtensions) const {
        std::vector<vk::ExtensionProperties> availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();
        std::vector<std::string> requiredExtensionNames(requiredExtensions.begin(), requiredExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensionNames.erase(std::remove(requiredExtensionNames.begin(), requiredExtensionNames.end(), extension.extensionName),
                                         requiredExtensionNames.end());
        }

        if (requiredExtensionNames.empty()) {
            std::cout << "Found all extensions" << std::endl;
            return true;
        } else {
            std::cout << "The following extensions are not supported:" << std::endl;
            for (const auto& name : requiredExtensionNames) {
                std::cout << "\t" << name << std::endl;
            }
            return false;
        }
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i != memProperties.memoryTypeCount; ++i) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("Memory type not found");
    }

    void oneTimeSubmit(const std::function<void(vk::CommandBuffer)>& func) const {
        vk::CommandBufferAllocateInfo comBufInfo;
        comBufInfo.setCommandPool(*commandPool);
        comBufInfo.setCommandBufferCount(1);

        vk::UniqueCommandBuffer comBuffer = std::move(device->allocateCommandBuffersUnique(comBufInfo).front());
        comBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        func(*comBuffer);
        comBuffer->end();

        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBuffers(*comBuffer);
        queue.submit(submitInfo);
        queue.waitIdle();
    }

    vk::UniqueDescriptorSet allocateDescSet(vk::DescriptorSetLayout descSetLayout) {
        vk::DescriptorSetAllocateInfo descSetInfo;
        descSetInfo.setDescriptorPool(*descPool);
        descSetInfo.setSetLayouts(descSetLayout);
        return std::move(device->allocateDescriptorSetsUnique(descSetInfo).front());
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                                      VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                                                                      VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
                                                                      void* pUserData) {
        std::cerr << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }

    GLFWwindow* window;
    vk::DynamicLoader dl;
    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT messenger;
    vk::UniqueSurfaceKHR surface;
    vk::UniqueDevice device;
    vk::PhysicalDevice physicalDevice;
    uint32_t queueFamilyIndex;
    vk::Queue queue;
    vk::UniqueCommandPool commandPool;
    vk::UniqueDescriptorPool descPool;
    Controls controls;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./main <file name> \n";
        return 0;
    }
    Context context;

    vk::SwapchainCreateInfoKHR scInfo;
    scInfo.setSurface(*context.surface);
    scInfo.setMinImageCount(3);
    scInfo.setImageFormat(vk::Format::eB8G8R8A8Unorm);
    scInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
    scInfo.setImageExtent({WIDTH, HEIGHT});
    scInfo.setImageArrayLayers(1);
    scInfo.setImageUsage(vk::ImageUsageFlagBits::eTransferDst);
    scInfo.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity);
    scInfo.setPresentMode(vk::PresentModeKHR::eFifo);
    scInfo.setClipped(true);
    scInfo.setQueueFamilyIndices(context.queueFamilyIndex);
    vk::UniqueSwapchainKHR swapchain = context.device->createSwapchainKHRUnique(scInfo);

    std::vector<vk::Image> swapchainImages = context.device->getSwapchainImagesKHR(*swapchain);

    vk::CommandBufferAllocateInfo comBufInfo;
    comBufInfo.setCommandPool(*context.commandPool);
    comBufInfo.setCommandBufferCount(static_cast<uint32_t>(swapchainImages.size()));
    std::vector<vk::UniqueCommandBuffer> comBuffers = context.device->allocateCommandBuffersUnique(comBufInfo);

    Image outputImage{context,
                      {WIDTH, HEIGHT},
                      vk::Format::eB8G8R8A8Unorm,
                      vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst};

    // Load mesh
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    loadFromFile(vertices, indices, faces, argv[1]);

    Buffer vertexBuffer = Buffer(context, BufferType::AccelInput, sizeof(Vertex) * vertices.size(), vertices.data());
    Buffer indexBuffer{context, Type::AccelInput, sizeof(uint32_t) * indices.size(), indices.data()};
    Buffer faceBuffer{context, Type::AccelInput, sizeof(Face) * faces.size(), faces.data()};

    // Create bottom level accel struct
    vk::AccelerationStructureGeometryTrianglesDataKHR triData;
    triData.setVertexFormat(vk::Format::eR32G32B32Sfloat);
    triData.setVertexData(vertexBuffer.get_address());
    triData.setVertexStride(sizeof(Vertex));
    triData.setMaxVertex(static_cast<uint32_t>(vertices.size()));
    triData.setIndexType(vk::IndexType::eUint32);
    triData.setIndexData(indexBuffer.get_address());

    vk::AccelerationStructureGeometryKHR triGeo;
    triGeo.setGeometryType(vk::GeometryTypeKHR::eTriangles);
    triGeo.setGeometry({triData});
    triGeo.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    const auto primitiveCount = static_cast<uint32_t>(indices.size() / 3);

    AccelerationStructure bottomAccel{context, triGeo, primitiveCount, vk::AccelerationStructureTypeKHR::eBottomLevel};

    // Create top level accel struct
    vk::TransformMatrixKHR transformMatrix = std::array{
        std::array{1.0f, 0.0f, 0.0f, 0.0f},
        std::array{0.0f, 1.0f, 0.0f, 0.0f},
        std::array{0.0f, 0.0f, 1.0f, 0.0f},
    };

    vk::AccelerationStructureInstanceKHR accelerationStructure;
    accelerationStructure.setTransform(transformMatrix);
    accelerationStructure.setMask(0xFF);
    accelerationStructure.setAccelerationStructureReference(bottomAccel);
    accelerationStructure.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

    Buffer instBuffer{context, Type::AccelInput, sizeof(vk::AccelerationStructureInstanceKHR), &accelerationStructure};

    vk::AccelerationStructureGeometryInstancesDataKHR instData;
    instData.setArrayOfPointers(false);
    instData.setData(instBuffer.deviceAddress);

    vk::AccelerationStructureGeometryKHR instGeo;
    instGeo.setGeometryType(vk::GeometryTypeKHR::eInstances);
    instGeo.setGeometry({instData});
    instGeo.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    AccelerationStructure topAccel{context, instGeo, 1, vk::AccelerationStructureTypeKHR::eTopLevel};

    // Load shaders
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

    // create ray tracing pipeline
    std::vector<vk::DescriptorSetLayoutBinding> bindings{
        {0, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eRaygenKHR},  // Binding = 0 : TLAS
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR},              // Binding = 1 : Storage image
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},         // Binding = 2 : Vertices
        {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},         // Binding = 3 : Indices
        {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},         // Binding = 4 : Faces
    };

    // Create desc set layout
    vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo;
    descSetLayoutInfo.setBindings(bindings);
    vk::UniqueDescriptorSetLayout descSetLayout = context.device->createDescriptorSetLayoutUnique(descSetLayoutInfo);

    // Create pipeline layout
    vk::PushConstantRange pushRange;
    pushRange.setOffset(0);
    pushRange.setSize(sizeof(Controls));
    pushRange.setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(*descSetLayout);
    pipelineLayoutInfo.setPushConstantRanges(pushRange);
    vk::UniquePipelineLayout pipelineLayout = context.device->createPipelineLayoutUnique(pipelineLayoutInfo);

    // Create pipeline
    vk::RayTracingPipelineCreateInfoKHR rayTracePLInfo;
    rayTracePLInfo.setStages(shaderStages);
    rayTracePLInfo.setGroups(shaderGroups);
    rayTracePLInfo.setMaxPipelineRayRecursionDepth(4);
    rayTracePLInfo.setLayout(*pipelineLayout);

    auto result = context.device->createRayTracingPipelineKHRUnique(nullptr, nullptr, rayTracePLInfo);
    if (result.result != vk::Result::eSuccess) {
        throw std::runtime_error("failed to create ray tracing pipeline.");
    }

    vk::UniquePipeline pipeline = std::move(result.value);

    // Get ray tracing properties
    auto properties = context.physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    auto rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

    // Calculate shader binding table (SBT) size
    uint32_t handleSize = rtProperties.shaderGroupHandleSize;
    uint32_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
    uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
    uint32_t sbtSize = groupCount * handleSizeAligned;

    // Get shader group handles
    std::vector<uint8_t> handleStorage(sbtSize);
    if (context.device->getRayTracingShaderGroupHandlesKHR(*pipeline, 0, groupCount, sbtSize, handleStorage.data()) != vk::Result::eSuccess) {
        throw std::runtime_error("failed to get ray tracing shader group handles.");
    }

    // Create SBT
    Buffer raygenSBT{context, BufferType::ShaderBindingTable, handleSize, handleStorage.data() + 0 * handleSizeAligned};
    Buffer missSBT{context, BufferType::ShaderBindingTable, handleSize, handleStorage.data() + 1 * handleSizeAligned};
    Buffer hitSBT{context, BufferType::ShaderBindingTable, handleSize, handleStorage.data() + 2 * handleSizeAligned};

    uint32_t stride = rtProperties.shaderGroupHandleAlignment;
    uint32_t size = rtProperties.shaderGroupHandleAlignment;

    vk::StridedDeviceAddressRegionKHR raygenRegion{raygenSBT.deviceAddress, stride, size};
    vk::StridedDeviceAddressRegionKHR missRegion{missSBT.deviceAddress, stride, size};
    vk::StridedDeviceAddressRegionKHR hitRegion{hitSBT.deviceAddress, stride, size};

    // Create desc set
    vk::UniqueDescriptorSet descSet = context.allocateDescSet(*descSetLayout);
    std::vector<vk::WriteDescriptorSet> writes(bindings.size());
    for (int i = 0; i < bindings.size(); i++) {
        writes[i].setDstSet(*descSet);
        writes[i].setDescriptorType(bindings[i].descriptorType);
        writes[i].setDescriptorCount(bindings[i].descriptorCount);
        writes[i].setDstBinding(bindings[i].binding);
    }
    writes[0].setPNext(&topAccel.get_accelInfo());
    writes[1].setImageInfo(outputImage.descImageInfo);
    writes[2].setBufferInfo(vertexBuffer.descBufferInfo);
    writes[3].setBufferInfo(indexBuffer.descBufferInfo);
    writes[4].setBufferInfo(faceBuffer.descBufferInfo);
    context.device->updateDescriptorSets(writes, nullptr);

    // Main loop
    uint32_t imageIndex = 0;
    // int frame = 0;
    vk::UniqueSemaphore imageAcquiredSemaphore = context.device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
    while (!glfwWindowShouldClose(context.window)) {
        glfwPollEvents();

        // Acquire next image
        imageIndex = context.device->acquireNextImageKHR(*swapchain, UINT64_MAX, *imageAcquiredSemaphore).value;

        // Record commands
        vk::CommandBuffer comBuffer = *comBuffers[imageIndex];
        comBuffer.begin(vk::CommandBufferBeginInfo());
        comBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipeline);
        comBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *pipelineLayout, 0, *descSet, nullptr);
        comBuffer.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eRaygenKHR, 0, sizeof(Controls), &context.controls);
        comBuffer.traceRaysKHR(raygenRegion, missRegion, hitRegion, {}, WIDTH, HEIGHT, 1);

        vk::Image srcImage = *outputImage.image;
        vk::Image dstImage = swapchainImages[imageIndex];
        Image::setImageLayout(comBuffer, srcImage, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal);
        Image::setImageLayout(comBuffer, dstImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        Image::copyImage(comBuffer, srcImage, dstImage);
        Image::setImageLayout(comBuffer, srcImage, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);
        Image::setImageLayout(comBuffer, dstImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR);

        comBuffer.end();

        // Submit
        context.queue.submit(vk::SubmitInfo().setCommandBuffers(comBuffer));

        // Present image
        vk::PresentInfoKHR screenInfo;
        screenInfo.setSwapchains(*swapchain);
        screenInfo.setImageIndices(imageIndex);
        screenInfo.setWaitSemaphores(*imageAcquiredSemaphore);
        auto result = context.queue.presentKHR(screenInfo);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present.");
        }
        context.queue.waitIdle();
        context.controls.frame++;
    }

    context.device->waitIdle();
    glfwDestroyWindow(context.window);
    glfwTerminate();
}
