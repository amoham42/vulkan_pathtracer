cmake_minimum_required(VERSION 3.0.0)
project(vulkan-tracer VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)

file(GLOB SHADERS shaders/*)
add_executable(vulkan-tracer main.cpp ${SHADERS}
        src/mesh_loader.h
        src/buffer.cpp
        src/buffer.h
        src/image.cpp
        src/image.h
        src/acceleration_structure.cpp
        src/acceleration_structure.h
)

source_group("Shader Files" FILES ${SHADERS})

target_link_libraries(${PROJECT_NAME} PUBLIC glfw)
target_include_directories(${PROJECT_NAME} PUBLIC
        "$ENV{VULKAN_SDK}/Include"
        "${PROJECT_SOURCE_DIR}/external/tinyobjloader"
        "${PROJECT_SOURCE_DIR}/external/glm"
)
