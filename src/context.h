#pragma once

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct GLFWwindow;

/// Instance, device and queue. With a window it also owns the GLFW window and its surface;
/// without one (--headless) it needs no display at all.
class Context {
public:
    Context(bool windowed, uint32_t width, uint32_t height);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;
    void oneTimeSubmit(const std::function<void(vk::CommandBuffer)>& record) const;

    GLFWwindow* window = nullptr;
    vk::DynamicLoader loader;
    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT messenger;
    vk::UniqueSurfaceKHR surface;
    vk::PhysicalDevice physicalDevice;
    vk::UniqueDevice device;
    uint32_t queueFamily = 0;
    vk::Queue queue;
    vk::UniqueCommandPool commandPool;

    vk::PhysicalDeviceProperties properties;
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rtProperties;
    std::string deviceName;

private:
    bool pickPhysicalDevice(const std::vector<const char*>& extensions);
};
