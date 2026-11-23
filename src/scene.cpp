#include "scene.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <glm/gtx/hash.hpp>

#include <algorithm>
#include <filesystem>
#include <limits>
#include <print>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

glm::vec3 rgb(const tinyobj::real_t* c) noexcept { return {c[0], c[1], c[2]}; }
float maxComponent(glm::vec3 v) noexcept { return std::max({v.x, v.y, v.z}); }

struct VertexHash {
    size_t operator()(const GpuVertex& v) const noexcept {
        const size_t h1 = std::hash<glm::vec4>{}(v.position);
        const size_t h2 = std::hash<glm::vec4>{}(v.normal);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

// MTL -> one of four BSDFs. `illum` follows the MTL spec: 5 = mirror, 7 = refraction (glass),
// 3 = ray-traced reflection, which with a finite Ns we read as a glossy Phong lobe.
GpuMaterial convert(const tinyobj::material_t& m) {
    glm::vec3 kd = rgb(m.diffuse), ks = rgb(m.specular), ke = rgb(m.emission);
    const float ns = std::max(1.0f, static_cast<float>(m.shininess));
    const float ior = m.ior > 1.0f ? static_cast<float>(m.ior) : 1.5f;

    MaterialType type = MaterialType::Diffuse;
    if (m.illum == 5) {
        type = MaterialType::Mirror;
        if (maxComponent(ks) <= 0.0f) ks = glm::vec3(1.0f);
    } else if (m.illum == 7 || m.illum == 4 || m.illum == 6 || m.illum == 9) {
        type = MaterialType::Dielectric;
    } else if (maxComponent(ks) > 0.0f && m.illum >= 2) {
        type = ns >= 2000.0f ? MaterialType::Mirror : MaterialType::Glossy;
    }

    // Keep glossy surfaces energy-conserving: diffuse + specular cannot reflect more than came in.
    if (type == MaterialType::Glossy) {
        const float total = maxComponent(kd + ks);
        if (total > 0.99f) {
            kd *= 0.99f / total;
            ks *= 0.99f / total;
        }
    }
    return {glm::vec4(kd, static_cast<float>(std::to_underlying(type))), glm::vec4(ks, ns), glm::vec4(ke, ior)};
}

}  // namespace

Scene loadObj(const std::string& path) {
    const std::filesystem::path file(path);
    tinyobj::ObjReaderConfig config;
    config.mtl_search_path = file.parent_path().string();
    config.triangulate = true;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path, config))
        throw std::runtime_error("could not load " + path + ": " + reader.Error());
    if (!reader.Warning().empty()) std::print(stderr, "[obj] {}", reader.Warning());

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();
    const auto& materials = reader.GetMaterials();

    Scene scene;
    scene.materials.reserve(materials.size() + 1);
    for (const auto& m : materials) scene.materials.push_back(convert(m));
    const auto fallback = static_cast<uint32_t>(scene.materials.size());
    scene.materials.push_back({glm::vec4(0.7f, 0.7f, 0.7f, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
                               glm::vec4(0.0f, 0.0f, 0.0f, 1.5f)});

    scene.boundsMin = glm::vec3(std::numeric_limits<float>::max());
    scene.boundsMax = glm::vec3(std::numeric_limits<float>::lowest());

    std::unordered_map<GpuVertex, uint32_t, VertexHash> uniqueVertices;
    for (const auto& shape : shapes) {
        const auto& mesh = shape.mesh;
        for (size_t f = 0; f < mesh.num_face_vertices.size(); f++) {
            for (int k = 0; k < 3; k++) {
                const tinyobj::index_t idx = mesh.indices[3 * f + k];
                GpuVertex v{};
                v.position = glm::vec4(attrib.vertices[3 * idx.vertex_index + 0], attrib.vertices[3 * idx.vertex_index + 1],
                                       attrib.vertices[3 * idx.vertex_index + 2], 1.0f);
                if (idx.normal_index >= 0) {
                    v.normal = glm::vec4(attrib.normals[3 * idx.normal_index + 0], attrib.normals[3 * idx.normal_index + 1],
                                         attrib.normals[3 * idx.normal_index + 2], 1.0f);
                }
                const auto [it, inserted] = uniqueVertices.emplace(v, static_cast<uint32_t>(scene.vertices.size()));
                if (inserted) {
                    scene.boundsMin = glm::min(scene.boundsMin, glm::vec3(v.position));
                    scene.boundsMax = glm::max(scene.boundsMax, glm::vec3(v.position));
                    scene.vertices.push_back(v);
                }
                scene.indices.push_back(it->second);
            }
            const int mat = f < mesh.material_ids.size() ? mesh.material_ids[f] : -1;
            scene.materialIds.push_back(mat >= 0 && mat < static_cast<int>(materials.size()) ? static_cast<uint32_t>(mat)
                                                                                             : fallback);
        }
    }
    if (scene.indices.empty()) throw std::runtime_error(path + " has no triangles");

    // Every emissive triangle is pre-unpacked into a self-contained GpuLight record (no GPU pointer chasing).
    for (uint32_t t = 0; t < scene.triangleCount(); t++) {
        const glm::vec3 em(scene.materials[scene.materialIds[t]].emission);
        if (maxComponent(em) <= 0.0f) continue;
        const glm::vec3 a(scene.vertices[scene.indices[3 * t + 0]].position);
        const glm::vec3 b(scene.vertices[scene.indices[3 * t + 1]].position);
        const glm::vec3 c(scene.vertices[scene.indices[3 * t + 2]].position);
        const glm::vec3 e1 = b - a;
        const glm::vec3 e2 = c - a;
        const glm::vec3 crossVec = glm::cross(e1, e2);
        const float crossLen = glm::length(crossVec);
        const float area = 0.5f * crossLen;
        if (area <= 0.0f) continue;
        scene.lightArea += area;
        scene.lights.push_back({
            .v0Cdf = glm::vec4(a, scene.lightArea),
            .e1 = glm::vec4(e1, 0.0f),
            .e2 = glm::vec4(e2, 0.0f),
            .normalPad = glm::vec4(crossVec / crossLen, 0.0f),
            .emission = glm::vec4(em, 0.0f),
        });
    }
    if (scene.lights.empty())
        std::println(stderr, "[obj] warning: {} has no emissive (Ke) materials; the image will be black", path);
    return scene;
}
