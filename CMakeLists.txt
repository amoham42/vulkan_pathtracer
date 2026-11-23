cmake_minimum_required(VERSION 3.24)
project(vulkan-tracer VERSION 2.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif ()

find_package(Vulkan 1.3 REQUIRED COMPONENTS glslc)

# Locate VMA (bundled in the LunarG Vulkan SDK, or system package on Linux)
find_path(VMA_INCLUDE_DIR
    NAMES vk_mem_alloc.h
    HINTS ${Vulkan_INCLUDE_DIRS} "$ENV{VULKAN_SDK}/Include"
    PATH_SUFFIXES vma
)
if (NOT VMA_INCLUDE_DIR)
    message(FATAL_ERROR "Vulkan Memory Allocator (vk_mem_alloc.h) not found. Install the Vulkan SDK or libvma-dev.")
endif ()

# Detect highest Vulkan target-env supported by glslc (1.4 if available, else 1.3; both emit SPIR-V 1.6)
execute_process(
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --help
    OUTPUT_VARIABLE GLSLC_HELP
    ERROR_QUIET
)
if (GLSLC_HELP MATCHES "vulkan1\\.4")
    set(VPT_GLSLC_TARGET_ENV "vulkan1.4")
else ()
    set(VPT_GLSLC_TARGET_ENV "vulkan1.3")
endif ()
message(STATUS "Shader compiler target-env: ${VPT_GLSLC_TARGET_ENV}")

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)

# ---- shaders: GLSL -> SPIR-V at build time ----
set(SHADER_SOURCES
    assets/shaders/pathtrace.comp
    assets/shaders/raygen.rgen
    assets/shaders/closesthit.rchit
    assets/shaders/miss.rmiss
    assets/shaders/shadow.rmiss
    assets/shaders/atrous.comp
    assets/shaders/tonemap.comp
)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
set(SPIRV_FILES)
foreach (SRC ${SHADER_SOURCES})
    get_filename_component(NAME ${SRC} NAME)
    set(OUT ${SHADER_OUT_DIR}/${NAME}.spv)
    add_custom_command(
        OUTPUT ${OUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUT_DIR}
        COMMAND Vulkan::glslc --target-env=${VPT_GLSLC_TARGET_ENV} -O -o ${OUT} ${CMAKE_SOURCE_DIR}/${SRC}
        DEPENDS ${CMAKE_SOURCE_DIR}/${SRC} ${CMAKE_SOURCE_DIR}/assets/shaders/common.glsl
        COMMENT "glslc ${NAME}"
        VERBATIM)
    list(APPEND SPIRV_FILES ${OUT})
endforeach ()
add_custom_target(shaders DEPENDS ${SPIRV_FILES})

# ---- tracer ----
add_executable(vulkan-tracer
    main.cpp
    src/context.cpp
    src/resources.cpp
    src/scene.cpp
    src/renderer.cpp
    src/options.cpp
)
add_dependencies(vulkan-tracer shaders)

# Copy compiled SPIR-V next to the executable for multi-config (VS/Xcode) and relocatable builds
add_custom_command(TARGET vulkan-tracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:vulkan-tracer>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${SHADER_OUT_DIR}" "$<TARGET_FILE_DIR:vulkan-tracer>/shaders"
    COMMENT "Syncing SPIR-V shaders next to executable"
)

target_include_directories(vulkan-tracer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/tinyobjloader
    ${CMAKE_SOURCE_DIR}/external/glm
    ${CMAKE_SOURCE_DIR}/external/stb
    ${VMA_INCLUDE_DIR}
)
target_link_libraries(vulkan-tracer PRIVATE Vulkan::Headers glfw)
target_compile_definitions(vulkan-tracer PRIVATE
    VPT_SHADER_DIR="${SHADER_OUT_DIR}"
    VPT_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets"
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_ENABLE_EXPERIMENTAL
    VULKAN_HPP_NO_SPACESHIP_OPERATOR
)

option(VPT_VALIDATION "Enable the Khronos validation layer by default (also togglable via --validation)" OFF)
if (VPT_VALIDATION)
    target_compile_definitions(vulkan-tracer PRIVATE VPT_VALIDATION_DEFAULT=1)
else ()
    target_compile_definitions(vulkan-tracer PRIVATE VPT_VALIDATION_DEFAULT=0)
endif ()

if (MSVC)
    target_compile_definitions(vulkan-tracer PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_options(vulkan-tracer PRIVATE /W4 /permissive- /Zc:__cplusplus /wd4100 /wd4324)
else ()
    target_compile_options(vulkan-tracer PRIVATE -Wall -Wextra -Wpedantic)
endif ()
