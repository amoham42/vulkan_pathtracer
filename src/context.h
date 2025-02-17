#pragma once

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <functional>
#include <iostream>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

constexpr int WIDTH = 1200;
constexpr int HEIGHT = 1200;

extern vk::DispatchLoaderDynamic defaultDispatchLoaderDynamic;

struct Controls {
    glm::vec3 cameraPosition = glm::vec3(0, -1, 5);
    float fov = 45.0f;
    float light_intensity = 1.0f;
    int frame = 0;
    int accumulate = 0;
};

class Context {
    public:
    Context();

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                                      VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                                                                      VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
                                                                      void* pUserData) {
        std::cerr << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;
    bool checkDeviceExtensionSupport(const std::vector<const char*>& requiredExtensions) const;

    void oneTimeSubmit(const std::function<void(vk::CommandBuffer)>& func) const;
    vk::UniqueDescriptorSet allocateDescSet(vk::DescriptorSetLayout descSetLayout);
    const int WIDTH = 1200;
    const int HEIGHT = 1200;

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

class Buffer {
public:
    enum class Type {
        Scratch,
        AccelInput,
        AccelStorage,
        ShaderBindingTable,
    };

    Buffer() = default;
    Buffer(const Context& context, Type type, vk::DeviceSize size, const void* data = nullptr);

    uint64_t getDeviceAddress() const { return deviceAddress; }
    const vk::DescriptorBufferInfo& getDescriptorInfo() const { return descBufferInfo; }

    void allocateBuffer(const Context& context, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memoryProps);
    void copyData(const Context& context, const void* data, vk::DeviceSize size);

    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    vk::DescriptorBufferInfo descBufferInfo;
    uint64_t deviceAddress = 0;
};

class Image {
public:
    Image() = default;
    Image(const Context& context, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage);
    static vk::AccessFlags toAccessFlags(vk::ImageLayout layout);
    static void setImageLayout(vk::CommandBuffer commandBuffer, vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
    static void copyImage(vk::CommandBuffer commandBuffer, vk::Image srcImage, vk::Image dstImage);

    vk::UniqueImage image;
    vk::UniqueImageView view;
    vk::UniqueDeviceMemory memory;
    vk::DescriptorImageInfo descImageInfo;
};

struct Accel {
    Accel() = default;
    Accel(const Context& context, vk::AccelerationStructureGeometryKHR geometry, uint32_t primitiveCount, vk::AccelerationStructureTypeKHR type);

    Buffer buffer;
    vk::UniqueAccelerationStructureKHR accel;
    vk::WriteDescriptorSetAccelerationStructureKHR descAccelInfo;
};