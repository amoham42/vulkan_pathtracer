#pragma once

#include <vulkan/vulkan_raii.hpp>

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include <vk_mem_alloc.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct GLFWwindow;

/// Move-only RAII wrapper around VmaAllocator.
class VmaAllocatorHandle {
public:
    VmaAllocatorHandle() = default;
    explicit VmaAllocatorHandle(VmaAllocator alloc) noexcept : handle_(alloc) {}
    ~VmaAllocatorHandle() { reset(); }

    VmaAllocatorHandle(const VmaAllocatorHandle&) = delete;
    VmaAllocatorHandle& operator=(const VmaAllocatorHandle&) = delete;

    VmaAllocatorHandle(VmaAllocatorHandle&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    VmaAllocatorHandle& operator=(VmaAllocatorHandle&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    void reset() noexcept {
        if (handle_) {
            vmaDestroyAllocator(handle_);
            handle_ = nullptr;
        }
    }

    VmaAllocator get() const noexcept { return handle_; }
    explicit operator bool() const noexcept { return handle_ != nullptr; }

private:
    VmaAllocator handle_ = nullptr;
};

/// Owns the Vulkan RAII instance, physical/logical device, VMA allocator, queue, and optional GLFW window.
class Context {
public:
    Context(bool windowed, uint32_t width, uint32_t height, bool enableValidation);
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    /// Records and submits a one-shot command buffer using synchronization2 (vkQueueSubmit2) + a fence.
    void oneTimeSubmit(const std::function<void(const vk::raii::CommandBuffer&)>& record) const;

    GLFWwindow* window = nullptr;

    // Declaration order defines strict reverse teardown order:
    // commandPool -> allocator -> device -> surface -> messenger -> instance -> raiiContext.
    vk::raii::Context raiiContext;
    vk::raii::Instance instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT messenger = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    VmaAllocatorHandle allocator;
    vk::raii::Queue queue = nullptr;
    vk::raii::CommandPool commandPool = nullptr;

    uint32_t apiVersion = VK_API_VERSION_1_3;
    uint32_t queueFamily = 0;
    bool hasMaintenance5 = false;
    bool validationEnabled = false;

    vk::PhysicalDeviceProperties properties{};
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rtProperties{};
    vk::PhysicalDeviceAccelerationStructurePropertiesKHR asProperties{};
    std::string deviceName;

private:
    bool pickPhysicalDevice(const std::vector<const char*>& requiredExtensions, bool requireGraphics);
};
