#include "context.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string_view>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    std::cerr << "[vulkan] " << data->pMessage << "\n";
    return VK_FALSE;
}

bool hasExtensions(vk::PhysicalDevice gpu, const std::vector<const char*>& required) {
    const auto available = gpu.enumerateDeviceExtensionProperties();
    return std::all_of(required.begin(), required.end(), [&](const char* name) {
        return std::any_of(available.begin(), available.end(),
                           [&](const vk::ExtensionProperties& e) { return std::string_view(e.extensionName) == name; });
    });
}

}  // namespace

Context::Context(bool windowed, uint32_t width, uint32_t height) {
    std::vector<const char*> instanceExtensions;
    if (windowed) {
        if (!glfwInit()) throw std::runtime_error("GLFW failed to initialise");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), "Vulkan Path Tracer", nullptr, nullptr);
        if (!window) throw std::runtime_error("could not open a window");
        uint32_t count = 0;
        const char** names = glfwGetRequiredInstanceExtensions(&count);
        instanceExtensions.assign(names, names + count);
    }
    instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VULKAN_HPP_DEFAULT_DISPATCHER.init(loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));

    vk::ApplicationInfo appInfo("Vulkan Path Tracer", 1, nullptr, 0, VK_API_VERSION_1_3);
    std::vector<const char*> layers;
#ifdef VPT_VALIDATION
    layers.push_back("VK_LAYER_KHRONOS_validation");
#endif
    instance = vk::createInstanceUnique({{}, &appInfo, layers, instanceExtensions});
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

    messenger = instance->createDebugUtilsMessengerEXTUnique(
        {{},
         vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning,
         vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral,
         &debugCallback});

    if (window) {
        VkSurfaceKHR raw;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &raw) != VK_SUCCESS)
            throw std::runtime_error("could not create a window surface");
        surface = vk::UniqueSurfaceKHR(raw, {*instance});
    }

    std::vector<const char*> deviceExtensions{
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    };
    if (window) deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    if (!pickPhysicalDevice(deviceExtensions))
        throw std::runtime_error("no GPU with hardware ray tracing (VK_KHR_ray_tracing_pipeline) was found");

    const float priority = 1.0f;
    vk::DeviceQueueCreateInfo queueInfo({}, queueFamily, 1, &priority);

    vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
                       vk::PhysicalDeviceRayTracingPipelineFeaturesKHR, vk::PhysicalDeviceAccelerationStructureFeaturesKHR>
        chain;
    chain.get<vk::DeviceCreateInfo>().setQueueCreateInfos(queueInfo).setPEnabledExtensionNames(deviceExtensions);
    chain.get<vk::PhysicalDeviceVulkan12Features>().setBufferDeviceAddress(true).setScalarBlockLayout(true);
    chain.get<vk::PhysicalDeviceVulkan13Features>().setSynchronization2(true);
    chain.get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>().setRayTracingPipeline(true);
    chain.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().setAccelerationStructure(true);
    device = physicalDevice.createDeviceUnique(chain.get<vk::DeviceCreateInfo>());
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);

    queue = device->getQueue(queueFamily, 0);
    commandPool = device->createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamily});
}

Context::~Context() {
    if (device) device->waitIdle();
    // the surface must go before the window
    surface.reset();
    if (window) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

// Prefer a discrete GPU: laptops list the integrated one first, and it usually can't ray trace.
bool Context::pickPhysicalDevice(const std::vector<const char*>& extensions) {
    int bestScore = -1;
    for (vk::PhysicalDevice gpu : instance->enumeratePhysicalDevices()) {
        if (!hasExtensions(gpu, extensions)) continue;
        const auto families = gpu.getQueueFamilyProperties();
        for (uint32_t i = 0; i < families.size(); i++) {
            const bool compute = static_cast<bool>(families[i].queueFlags & vk::QueueFlagBits::eCompute);
            const bool present = !surface || gpu.getSurfaceSupportKHR(i, *surface);
            if (!compute || !present) continue;
            const int score = gpu.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu ? 2 : 1;
            if (score > bestScore) {
                bestScore = score;
                physicalDevice = gpu;
                queueFamily = i;
            }
            break;
        }
    }
    if (bestScore < 0) return false;

    auto props = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    properties = props.get<vk::PhysicalDeviceProperties2>().properties;
    rtProperties = props.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    deviceName = properties.deviceName.data();
    return true;
}

uint32_t Context::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags wanted) const {
    const auto memory = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memory.memoryTypeCount; i++)
        if ((typeFilter & (1u << i)) && (memory.memoryTypes[i].propertyFlags & wanted) == wanted) return i;
    throw std::runtime_error("no suitable memory type");
}

void Context::oneTimeSubmit(const std::function<void(vk::CommandBuffer)>& record) const {
    auto cmd = std::move(device->allocateCommandBuffersUnique({*commandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
    cmd->begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    record(*cmd);
    cmd->end();
    queue.submit(vk::SubmitInfo().setCommandBuffers(*cmd));
    queue.waitIdle();
}
