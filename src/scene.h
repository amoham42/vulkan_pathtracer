#pragma once

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Layouts shared with assets/shaders/common.glsl (scalar block layout). Verified at compile time below.

// Bit 31 of materialIds[tri] indicates the triangle has authored smooth vertex normals.
inline constexpr uint32_t kTriSmoothNormalBit = 0x80000000u;

enum class MaterialType : uint32_t { Diffuse = 0, Glossy = 1, Mirror = 2, Dielectric = 3 };

struct GpuMaterial {
    glm::vec4 diffuse;   // rgb albedo, w = MaterialType
    glm::vec4 specular;  // rgb, w = Phong exponent
    glm::vec4 emission;  // rgb radiance, w = ior (Dielectric) or specularProbability (Glossy)
};
static_assert(sizeof(GpuMaterial) == 48);
static_assert(offsetof(GpuMaterial, specular) == 16);
static_assert(offsetof(GpuMaterial, emission) == 32);

// Pre-unpacked emissive triangle so light sampling needs zero index/vertex/material pointer chasing.
struct GpuLight {
    glm::vec4 v0Cdf;      // xyz = vertex A, w = running area CDF
    glm::vec4 e1;         // xyz = vertex B - A
    glm::vec4 e2;         // xyz = vertex C - A
    glm::vec4 normalPad;  // xyz = unit geometric normal
    glm::vec4 emission;   // rgb = emitted radiance
};
static_assert(sizeof(GpuLight) == 80);
static_assert(offsetof(GpuLight, e1) == 16);
static_assert(offsetof(GpuLight, e2) == 32);
static_assert(offsetof(GpuLight, normalPad) == 48);
static_assert(offsetof(GpuLight, emission) == 64);

/// A deduplicated triangle mesh loaded from Wavefront OBJ + MTL, ready for GPU upload.
/// Positions (12 B) and octahedral snorm16x2 normals (4 B) are stored in compact parallel arrays;
/// flat triangles carry bit 31 clear in materialIds so position_fetch hits skip index/normal loads entirely.
struct Scene {
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> packedNormals;  // octahedral snorm16x2 per vertex
    std::vector<uint32_t> indices;
    std::vector<uint32_t> materialIds;    // bits 0..30 = material index, bit 31 = has smooth vertex normals
    std::vector<GpuMaterial> materials;
    std::vector<GpuLight> lights;
    float lightArea = 0.0f;
    glm::vec3 boundsMin{0.0f}, boundsMax{0.0f};

    uint32_t triangleCount() const noexcept { return static_cast<uint32_t>(indices.size() / 3); }
};

Scene loadObj(const std::string& path);
