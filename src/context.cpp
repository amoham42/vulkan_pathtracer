#include "context.h"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
// The key callback function
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // Only process key presses or repeats.
    if (action != GLFW_PRESS && action != GLFW_REPEAT)
        return;

    // Retrieve our controls instance from the window's user pointer.
    auto* controls = reinterpret_cast<Controls*>(glfwGetWindowUserPointer(window));
    if (!controls)
        return;

    constexpr float moveSpeed = 0.1f;
    constexpr float fovStep = 5.0f;

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

Context::Context() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Pathtracing", nullptr, nullptr);

    glfwSetWindowUserPointer(window, &controls);
    glfwSetKeyCallback(window, key_callback);

    // Prepase extensions and layers
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    // std::vector layers{"VK_LAYER_KHRONOS_validation"};

    auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    // Create instance
    vk::ApplicationInfo appInfo;
    appInfo.setApiVersion(VK_API_VERSION_1_3);

    vk::InstanceCreateInfo instanceInfo;
    instanceInfo.setPApplicationInfo(&appInfo);
    // instanceInfo.setPEnabledLayerNames(layers);
    instanceInfo.setPEnabledExtensionNames(extensions);
    instance = vk::createInstanceUnique(instanceInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

    // Pick first gpu
    physicalDevice = instance->enumeratePhysicalDevices().front();

    // Create debug messenger
    vk::DebugUtilsMessengerCreateInfoEXT messengerInfo;
    messengerInfo.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
    messengerInfo.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
    messengerInfo.setPfnUserCallback(&debugUtilsMessengerCallback);
    messenger = instance->createDebugUtilsMessengerEXTUnique(messengerInfo);

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
        throw std::runtime_error("Some extensions are not supported!");
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

bool Context::checkDeviceExtensionSupport(const std::vector<const char*>& requiredExtensions) const {
    std::vector<vk::ExtensionProperties> availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();
    std::vector<std::string> requiredExtensionNames(requiredExtensions.begin(), requiredExtensions.end());

    for (const auto& extension : availableExtensions) {
        requiredExtensionNames.erase(std::remove(requiredExtensionNames.begin(), requiredExtensionNames.end(), extension.extensionName),
                                     requiredExtensionNames.end());
    }

    if (requiredExtensionNames.empty()) {
        std::cout << "All Extensions are supported!" << std::endl;
        return true;
    } else {
        std::cout << "The following extensions are not supported:" << std::endl;
        for (const auto& name : requiredExtensionNames) {
            std::cout << "\t" << name << std::endl;
        }
        return false;
    }
}

uint32_t Context::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i != memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find correct memory type!");
}

void Context::oneTimeSubmit(const std::function<void(vk::CommandBuffer)>& func) const {
     vk::CommandBufferAllocateInfo commandBufferInfo;
     commandBufferInfo.setCommandPool(*commandPool);
     commandBufferInfo.setCommandBufferCount(1);

     vk::UniqueCommandBuffer commandBuffer = std::move(device->allocateCommandBuffersUnique(commandBufferInfo).front());
     commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
     func(*commandBuffer);
     commandBuffer->end();

     vk::SubmitInfo submitInfo;
     submitInfo.setCommandBuffers(*commandBuffer);
     queue.submit(submitInfo);
     queue.waitIdle();
}

vk::UniqueDescriptorSet Context::allocateDescSet(vk::DescriptorSetLayout descSetLayout) {
     vk::DescriptorSetAllocateInfo descSetInfo;
     descSetInfo.setDescriptorPool(*descPool);
     descSetInfo.setSetLayouts(descSetLayout);
     return std::move(device->allocateDescriptorSetsUnique(descSetInfo).front());
}

Buffer::Buffer(const Context& context, Type type, vk::DeviceSize size, const void* data) {
    vk::BufferUsageFlags usage;
    vk::MemoryPropertyFlags memoryProps;
    using Usage = vk::BufferUsageFlagBits;
    using Memory = vk::MemoryPropertyFlagBits;

    switch (type) {
        case Type::AccelInput:
            usage = Usage::eAccelerationStructureBuildInputReadOnlyKHR | Usage::eStorageBuffer | Usage::eShaderDeviceAddress;
            memoryProps = Memory::eHostVisible | Memory::eHostCoherent;
            break;
        case Type::Scratch:
            usage = Usage::eStorageBuffer | Usage::eShaderDeviceAddress;
            memoryProps = Memory::eDeviceLocal;
            break;
        case Type::AccelStorage:
            usage = Usage::eAccelerationStructureStorageKHR | Usage::eShaderDeviceAddress;
            memoryProps = Memory::eDeviceLocal;
            break;
        case Type::ShaderBindingTable:
            usage = Usage::eShaderBindingTableKHR | Usage::eShaderDeviceAddress;
            memoryProps = Memory::eHostVisible | Memory::eHostCoherent;
            break;
    }

    allocateBuffer(context, size, usage, memoryProps);
    if (data) {
        copyData(context, data, size);
    }
}

void Buffer::allocateBuffer(const Context& context, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memoryProps) {
    buffer = context.device->createBufferUnique({{}, size, usage});

    vk::MemoryRequirements requirements = context.device->getBufferMemoryRequirements(*buffer);
    uint32_t memoryTypeIndex = context.findMemoryType(requirements.memoryTypeBits, memoryProps);

    vk::MemoryAllocateFlagsInfo flagsInfo{vk::MemoryAllocateFlagBits::eDeviceAddress};

    vk::MemoryAllocateInfo memoryInfo;
    memoryInfo.setAllocationSize(requirements.size);
    memoryInfo.setMemoryTypeIndex(memoryTypeIndex);
    memoryInfo.setPNext(&flagsInfo);
    memory = context.device->allocateMemoryUnique(memoryInfo);

    context.device->bindBufferMemory(*buffer, *memory, 0);

    vk::BufferDeviceAddressInfoKHR bufferDeviceAI{*buffer};
    deviceAddress = context.device->getBufferAddressKHR(&bufferDeviceAI);

    descBufferInfo.setBuffer(*buffer);
    descBufferInfo.setOffset(0);
    descBufferInfo.setRange(size);
}

void Buffer::copyData(const Context& context, const void* data, vk::DeviceSize size) {
    void* mapped = context.device->mapMemory(*memory, 0, size);
    memcpy(mapped, data, size);
    context.device->unmapMemory(*memory);
}

Image::Image(const Context& context, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage) {
    // Create image
    vk::ImageCreateInfo imageInfo;
    imageInfo.setImageType(vk::ImageType::e2D);
    imageInfo.setExtent({extent.width, extent.height, 1});
    imageInfo.setMipLevels(1);
    imageInfo.setArrayLayers(1);
    imageInfo.setFormat(format);
    imageInfo.setUsage(usage);
    image = context.device->createImageUnique(imageInfo);

    // Allocate memory
    vk::MemoryRequirements requirements = context.device->getImageMemoryRequirements(*image);
    uint32_t memoryTypeIndex = context.findMemoryType(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::MemoryAllocateInfo memoryInfo;
    memoryInfo.setAllocationSize(requirements.size);
    memoryInfo.setMemoryTypeIndex(memoryTypeIndex);
    memory = context.device->allocateMemoryUnique(memoryInfo);

    // Bind memory and image
    context.device->bindImageMemory(*image, *memory, 0);

    // Create image view
    vk::ImageViewCreateInfo imageViewInfo;
    imageViewInfo.setImage(*image);
    imageViewInfo.setViewType(vk::ImageViewType::e2D);
    imageViewInfo.setFormat(format);
    imageViewInfo.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    view = context.device->createImageViewUnique(imageViewInfo);

    // Set image info
    descImageInfo.setImageView(*view);
    descImageInfo.setImageLayout(vk::ImageLayout::eGeneral);
    context.oneTimeSubmit([&](vk::CommandBuffer commandBuffer) {  //
        setImageLayout(commandBuffer, *image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    });
}

vk::AccessFlags Image::toAccessFlags(vk::ImageLayout layout) {
    switch (layout) {
        case vk::ImageLayout::eTransferSrcOptimal:
            return vk::AccessFlagBits::eTransferRead;
        case vk::ImageLayout::eTransferDstOptimal:
            return vk::AccessFlagBits::eTransferWrite;
        default:
            return {};
    }
}

void Image::setImageLayout(vk::CommandBuffer commandBuffer, vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
    vk::ImageMemoryBarrier barrier;
    barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setImage(image);
    barrier.setOldLayout(oldLayout);
    barrier.setNewLayout(newLayout);
    barrier.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    barrier.setSrcAccessMask(toAccessFlags(oldLayout));
    barrier.setDstAccessMask(toAccessFlags(newLayout));
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,  //
                                  vk::PipelineStageFlagBits::eAllCommands,  //
                                  {}, {}, {}, barrier);
}

void Image::copyImage(vk::CommandBuffer commandBuffer, vk::Image srcImage, vk::Image dstImage) {
    vk::ImageCopy copyRegion;
    copyRegion.setSrcSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
    copyRegion.setDstSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
    copyRegion.setExtent({WIDTH, HEIGHT, 1});
    commandBuffer.copyImage(srcImage, vk::ImageLayout::eTransferSrcOptimal, dstImage, vk::ImageLayout::eTransferDstOptimal, copyRegion);
}

Accel::Accel(const Context& context, vk::AccelerationStructureGeometryKHR geometry, uint32_t primitiveCount, vk::AccelerationStructureTypeKHR type) {
    vk::AccelerationStructureBuildGeometryInfoKHR buildGeometryInfo;
    buildGeometryInfo.setType(type);
    buildGeometryInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
    buildGeometryInfo.setGeometries(geometry);

    // Create buffer
    vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo = context.device->getAccelerationStructureBuildSizesKHR(  //
        vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount);
    vk::DeviceSize size = buildSizesInfo.accelerationStructureSize;
    buffer = Buffer{context, Buffer::Type::AccelStorage, size};

    // Create accel
    vk::AccelerationStructureCreateInfoKHR accelInfo;
    accelInfo.setBuffer(*buffer.buffer);
    accelInfo.setSize(size);
    accelInfo.setType(type);
    accel = context.device->createAccelerationStructureKHRUnique(accelInfo);

    // Build
    Buffer scratchBuffer{context, Buffer::Type::Scratch, buildSizesInfo.buildScratchSize};
    buildGeometryInfo.setScratchData(scratchBuffer.deviceAddress);
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

