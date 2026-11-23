#pragma once

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Layouts shared with assets/shaders/common.glsl (scalar block layout). Verified at compile time below.

struct GpuVertex {
    glm::vec4 position;  // xyz
    glm::vec4 normal;    // xyz, w = 1 if the OBJ supplied a normal (else the shader uses the face normal)

    bool operator==(const GpuVertex&) const = default;
};
static_assert(sizeof(GpuVertex) == 32);
static_assert(offsetof(GpuVertex, normal) == 16);

enum class MaterialType : uint32_t { Diffuse = 0, Glossy = 1, Mirror = 2, Dielectric = 3 };

struct GpuMaterial {
    glm::vec4 diffuse;   // rgb albedo, w = MaterialType
    glm::vec4 specular;  // rgb, w = Phong exponent
    glm::vec4 emission;  // rgb radiance, w = index of refraction
};
static_assert(sizeof(GpuMaterial) == 48);
static_assert(offsetof(GpuMaterial, specular) == 16);
static_assert(offsetof(GpuMaterial, emission) == 32);

struct GpuLight {
    uint32_t triangle;
    float cdf;  // running sum of emitter area, for picking a light proportional to its area
};
static_assert(sizeof(GpuLight) == 8);
static_assert(offsetof(GpuLight, cdf) == 4);

/// A deduplicated triangle mesh loaded from Wavefront OBJ + MTL, ready for GPU upload.
struct Scene {
    std::vector<GpuVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> materialIds;  // one per triangle
    std::vector<GpuMaterial> materials;
    std::vector<GpuLight> lights;
    float lightArea = 0.0f;
    glm::vec3 boundsMin{0.0f}, boundsMax{0.0f};

    uint32_t triangleCount() const noexcept { return static_cast<uint32_t>(indices.size() / 3); }
};

Scene loadObj(const std::string& path);
