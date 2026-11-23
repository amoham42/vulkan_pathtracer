cmake_minimum_required(VERSION 3.24)
project(vulkan-tracer VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif ()

find_package(Vulkan 1.3 REQUIRED COMPONENTS glslc)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)

# ---- shaders: GLSL -> SPIR-V at build time, next to the executable ----
set(SHADER_SOURCES
    shaders/raygen.rgen
    shaders/closesthit.rchit
    shaders/miss.rmiss
    shaders/shadow.rmiss
    shaders/atrous.comp
    shaders/tonemap.comp
)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
set(SPIRV_FILES)
foreach (SRC ${SHADER_SOURCES})
    get_filename_component(NAME ${SRC} NAME)
    set(OUT ${SHADER_OUT_DIR}/${NAME}.spv)
    add_custom_command(
        OUTPUT ${OUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUT_DIR}
        COMMAND Vulkan::glslc --target-env=vulkan1.3 -O -o ${OUT} ${CMAKE_SOURCE_DIR}/${SRC}
        DEPENDS ${CMAKE_SOURCE_DIR}/${SRC} ${CMAKE_SOURCE_DIR}/shaders/common.glsl
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
target_include_directories(vulkan-tracer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/tinyobjloader
    ${CMAKE_SOURCE_DIR}/external/glm
    ${CMAKE_SOURCE_DIR}/external/stb
)
target_link_libraries(vulkan-tracer PRIVATE Vulkan::Headers glfw)
target_compile_definitions(vulkan-tracer PRIVATE
    VPT_SHADER_DIR="${SHADER_OUT_DIR}"
    VPT_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets"
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)
option(VPT_VALIDATION "Enable the Khronos validation layer" OFF)
if (VPT_VALIDATION)
    target_compile_definitions(vulkan-tracer PRIVATE VPT_VALIDATION)
endif ()
if (MSVC)
    target_compile_definitions(vulkan-tracer PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
    target_compile_options(vulkan-tracer PRIVATE /W3 /permissive-)
endif ()
