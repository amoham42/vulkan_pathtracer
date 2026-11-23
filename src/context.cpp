#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#include "context.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <print>
#include <ranges>
#include <stdexcept>
#include <string_view>

namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    std::println(stderr, "[vulkan] {}", data->pMessage);
    return VK_FALSE;
}

bool hasAllExtensions(const vk::raii::PhysicalDevice& gpu, const std::vector<const char*>& required) {
    const auto available = gpu.enumerateDeviceExtensionProperties();
    return std::ranges::all_of(required, [&](const char* name) {
        return std::ranges::any_of(available, [&](const vk::ExtensionProperties& e) {
            return std::string_view(e.extensionName.data()) == name;
        });
    });
}

bool hasExtension(const vk::raii::PhysicalDevice& gpu, std::string_view name) {
    const auto available = gpu.enumerateDeviceExtensionProperties();
    return std::ranges::any_of(available, [&](const vk::ExtensionProperties& e) {
        return std::string_view(e.extensionName.data()) == name;
    });
}

bool hasInstanceLayer(const vk::raii::Context& ctx, std::string_view name) {
    const auto layers = ctx.enumerateInstanceLayerProperties();
    return std::ranges::any_of(layers, [&](const vk::LayerProperties& l) {
        return std::string_view(l.layerName.data()) == name;
    });
}

bool hasInstanceExtension(const vk::raii::Context& ctx, std::string_view name) {
    const auto exts = ctx.enumerateInstanceExtensionProperties();
    return std::ranges::any_of(exts, [&](const vk::ExtensionProperties& e) {
        return std::string_view(e.extensionName.data()) == name;
    });
}

uint32_t negotiateApiVersion(uint32_t loaderVersion) {
#if defined(VK_API_VERSION_1_4)
    if (loaderVersion >= VK_API_VERSION_1_4) return VK_API_VERSION_1_4;
#endif
    if (loaderVersion >= VK_API_VERSION_1_3) return VK_API_VERSION_1_3;
    throw std::runtime_error("Vulkan 1.3 or newer is required");
}

}  // namespace

Context::Context(bool windowed, uint32_t width, uint32_t height, bool enableValidation) {
    std::vector<const char*> instanceExtensions;
    if (windowed) {
        if (!glfwInit()) throw std::runtime_error("GLFW failed to initialise");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), "Vulkan Path Tracer", nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            throw std::runtime_error("could not open a window");
        }
        uint32_t count = 0;
        const char** names = glfwGetRequiredInstanceExtensions(&count);
        instanceExtensions.assign(names, names + count);
    }

    const bool hasDebugUtils = hasInstanceExtension(raiiContext, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (hasDebugUtils) instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    apiVersion = negotiateApiVersion(raiiContext.enumerateInstanceVersion());

    std::vector<const char*> layers;
    if (enableValidation) {
        constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";
        if (hasInstanceLayer(raiiContext, kValidationLayer)) {
            layers.push_back(kValidationLayer);
            validationEnabled = true;
        } else {
            std::println(stderr, "[vulkan] warning: {} requested but not installed; continuing without validation",
                         kValidationLayer);
        }
    }

    const vk::ApplicationInfo appInfo("Vulkan Path Tracer", VK_MAKE_VERSION(2, 0, 0), nullptr, 0, apiVersion);
    instance = raiiContext.createInstance({{}, &appInfo, layers, instanceExtensions});

    if (hasDebugUtils && validationEnabled) {
        messenger = instance.createDebugUtilsMessengerEXT(
            {{},
             vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning,
             vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral,
             &debugCallback});
    }

    if (window) {
        VkSurfaceKHR raw = VK_NULL_HANDLE;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &raw) != VK_SUCCESS)
            throw std::runtime_error("could not create a window surface");
        surface = vk::raii::SurfaceKHR(instance, raw);
    }

    std::vector<const char*> deviceExtensions{
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
    };
    if (window) deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    if (!pickPhysicalDevice(deviceExtensions, window != nullptr))
        throw std::runtime_error("no GPU with Vulkan 1.3+ hardware ray tracing (VK_KHR_ray_tracing_pipeline) was found");

    // Cap effective API version to what the physical device itself supports.
    apiVersion = std::min(apiVersion, properties.apiVersion);

    // Enable Vulkan 1.4-promoted maintenance5 when supported for inline ShaderModule creation.
    hasMaintenance5 = hasExtension(physicalDevice, VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    if (hasMaintenance5) deviceExtensions.push_back(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);

    const float priority = 1.0f;
    const vk::DeviceQueueCreateInfo queueInfo({}, queueFamily, 1, &priority);

    vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
                       vk::PhysicalDeviceRayTracingPipelineFeaturesKHR, vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
                       vk::PhysicalDeviceMaintenance5FeaturesKHR>
        chain;
    chain.get<vk::DeviceCreateInfo>().setQueueCreateInfos(queueInfo).setPEnabledExtensionNames(deviceExtensions);
    chain.get<vk::PhysicalDeviceVulkan12Features>().setBufferDeviceAddress(true).setScalarBlockLayout(true);
    chain.get<vk::PhysicalDeviceVulkan13Features>().setSynchronization2(true);
    chain.get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>().setRayTracingPipeline(true);
    chain.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().setAccelerationStructure(true);
    if (hasMaintenance5) {
        chain.get<vk::PhysicalDeviceMaintenance5FeaturesKHR>().setMaintenance5(true);
    } else {
        chain.unlink<vk::PhysicalDeviceMaintenance5FeaturesKHR>();
    }

    device = physicalDevice.createDevice(chain.get<vk::DeviceCreateInfo>());
    queue = device.getQueue(queueFamily, 0);
    commandPool = device.createCommandPool({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamily});

    // Initialise VMA using dynamic Vulkan pointers sourced from the RAII context/instance.
    VmaVulkanFunctions vmaFuncs{};
    vmaFuncs.vkGetInstanceProcAddr = raiiContext.getDispatcher()->vkGetInstanceProcAddr;
    vmaFuncs.vkGetDeviceProcAddr =
        reinterpret_cast<PFN_vkGetDeviceProcAddr>(vmaFuncs.vkGetInstanceProcAddr(*instance, "vkGetDeviceProcAddr"));

    VmaAllocatorCreateInfo allocInfo{};
    allocInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    allocInfo.physicalDevice = *physicalDevice;
    allocInfo.device = *device;
    allocInfo.instance = *instance;
#if defined(VMA_VULKAN_VERSION) && VMA_VULKAN_VERSION >= 1004000
    allocInfo.vulkanApiVersion = apiVersion;
#else
    allocInfo.vulkanApiVersion = std::min(apiVersion, VK_API_VERSION_1_3);
#endif
    allocInfo.pVulkanFunctions = &vmaFuncs;

    VmaAllocator rawAlloc = nullptr;
    if (vmaCreateAllocator(&allocInfo, &rawAlloc) != VK_SUCCESS)
        throw std::runtime_error("vmaCreateAllocator failed");
    allocator = VmaAllocatorHandle(rawAlloc);
}

Context::~Context() {
    if (*device) device.waitIdle();
    // Destroy VMA & device-owned objects before destroying the surface and GLFW window.
    commandPool = nullptr;
    allocator.reset();
    device = nullptr;
    surface = nullptr;
    if (window) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

bool Context::pickPhysicalDevice(const std::vector<const char*>& requiredExtensions, bool requireGraphics) {
    int bestScore = -1;
    for (auto&& gpu : instance.enumeratePhysicalDevices()) {
        const auto props = gpu.getProperties();
        if (props.apiVersion < VK_API_VERSION_1_3) continue;
        if (!hasAllExtensions(gpu, requiredExtensions)) continue;

        const auto families = gpu.getQueueFamilyProperties();
        const vk::QueueFlags needed =
            requireGraphics ? (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute) : vk::QueueFlagBits::eCompute;
        for (uint32_t i = 0; i < families.size(); i++) {
            const bool flagsOk = (families[i].queueFlags & needed) == needed;
            const bool presentOk = !*surface || gpu.getSurfaceSupportKHR(i, *surface) == VK_TRUE;
            if (!flagsOk || !presentOk) continue;

            const int score = props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu ? 2 : 1;
            if (score > bestScore) {
                bestScore = score;
                physicalDevice = std::move(gpu);
                queueFamily = i;
            }
            break;
        }
    }
    if (bestScore < 0) return false;

    const auto props2 = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2,
                                                      vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
                                                      vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();
    properties = props2.get<vk::PhysicalDeviceProperties2>().properties;
    rtProperties = props2.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    asProperties = props2.get<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();
    deviceName = properties.deviceName.data();
    return true;
}

void Context::oneTimeSubmit(const std::function<void(const vk::raii::CommandBuffer&)>& record) const {
    auto cmds = device.allocateCommandBuffers({*commandPool, vk::CommandBufferLevel::ePrimary, 1});
    const auto& cmd = cmds.front();
    cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    record(cmd);
    cmd.end();

    const vk::raii::Fence fence = device.createFence({});
    const vk::CommandBufferSubmitInfo cmdInfo(*cmd);
    const vk::SubmitInfo2 submit({}, {}, cmdInfo, {});
    queue.submit2(submit, *fence);
    if (device.waitForFences(*fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
        throw std::runtime_error("waiting for one-time GPU submission failed");
}
