#include <vulkan/vulkan.hpp>

struct Context;

enum class BufferType {
    Scratch,
    AccelInput,
    AccelStorage,
    ShaderBindingTable,
};

class Buffer {
public:
    Buffer() = default;
    Buffer(Context& context, BufferType type, vk::DeviceSize size, const void* data = nullptr);
    vk::UniqueBuffer& get_buffer();
    uint64_t get_address() const;
    vk::DescriptorBufferInfo& get_bufferInfo();

private:
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    vk::DescriptorBufferInfo descBufferInfo;
    uint64_t deviceAddress = 0;
};