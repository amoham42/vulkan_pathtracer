#pragma once

#include <vulkan/vulkan.hpp>
#include "context.h"

class Image {
public:
    Image() = default;
    Image(vkContext& context, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage);
    static vk::AccessFlags toAccessFlags(vk::ImageLayout layout);
    static void copyImage(vk::CommandBuffer commandBuffer, vk::Image srcImage, vk::Image dstImage);
    static void setImageLayout(vk::CommandBuffer commandBuffer, vk::Image image,
                               vk::ImageLayout oldLayout, vk::ImageLayout newLayout);

    vk::UniqueImage image;
    vk::UniqueImageView view;
    vk::UniqueDeviceMemory memory;
    vk::DescriptorImageInfo descImageInfo;
};
