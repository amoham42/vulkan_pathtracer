#include "scene.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

glm::vec3 rgb(const tinyobj::real_t* c) { return {c[0], c[1], c[2]}; }
float maxComponent(glm::vec3 v) { return std::max(v.x, std::max(v.y, v.z)); }

// MTL -> one of four BSDFs. `illum` follows the MTL spec: 5 = mirror, 7 = refraction (glass),
// 3 = ray-traced reflection, which with a finite Ns we read as a glossy Phong lobe.
GpuMaterial convert(const tinyobj::material_t& m) {
    glm::vec3 kd = rgb(m.diffuse), ks = rgb(m.specular), ke = rgb(m.emission);
    float ns = std::max(1.0f, static_cast<float>(m.shininess));
    float ior = m.ior > 1.0f ? static_cast<float>(m.ior) : 1.5f;

    MaterialType type = MaterialType::Diffuse;
    if (m.illum == 5) {
        type = MaterialType::Mirror;
        if (maxComponent(ks) <= 0.0f) ks = glm::vec3(1.0f);
    } else if (m.illum == 7 || m.illum == 4 || m.illum == 6 || m.illum == 9) {
        type = MaterialType::Dielectric;
    } else if (maxComponent(ks) > 0.0f && m.illum >= 2) {
        type = ns >= 2000.0f ? MaterialType::Mirror : MaterialType::Glossy;
    }

    // keep a glossy surface energy-conserving: diffuse + specular can't reflect more than came in
    if (type == MaterialType::Glossy) {
        const float total = maxComponent(kd + ks);
        if (total > 0.99f) {
            kd *= 0.99f / total;
            ks *= 0.99f / total;
        }
    }
    return {glm::vec4(kd, static_cast<float>(type)), glm::vec4(ks, ns), glm::vec4(ke, ior)};
}

}  // namespace

Scene loadObj(const std::string& path) {
    const std::filesystem::path file(path);
    tinyobj::ObjReaderConfig config;
    config.mtl_search_path = file.parent_path().string();
    config.triangulate = true;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path, config)) throw std::runtime_error("could not load " + path + ": " + reader.Error());
    if (!reader.Warning().empty()) std::cerr << "[obj] " << reader.Warning();

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();
    const auto& materials = reader.GetMaterials();

    Scene scene;
    for (const auto& m : materials) scene.materials.push_back(convert(m));
    const auto fallback = static_cast<uint32_t>(scene.materials.size());
    scene.materials.push_back({glm::vec4(0.7f, 0.7f, 0.7f, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.5f)});

    scene.boundsMin = glm::vec3(std::numeric_limits<float>::max());
    scene.boundsMax = glm::vec3(std::numeric_limits<float>::lowest());

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
                scene.boundsMin = glm::min(scene.boundsMin, glm::vec3(v.position));
                scene.boundsMax = glm::max(scene.boundsMax, glm::vec3(v.position));
                scene.indices.push_back(static_cast<uint32_t>(scene.vertices.size()));
                scene.vertices.push_back(v);
            }
            const int mat = f < mesh.material_ids.size() ? mesh.material_ids[f] : -1;
            scene.materialIds.push_back(mat >= 0 && mat < static_cast<int>(materials.size()) ? static_cast<uint32_t>(mat) : fallback);
        }
    }
    if (scene.indices.empty()) throw std::runtime_error(path + " has no triangles");

    // every emissive triangle is a light, picked in proportion to its area
    for (uint32_t t = 0; t < scene.triangleCount(); t++) {
        if (maxComponent(glm::vec3(scene.materials[scene.materialIds[t]].emission)) <= 0.0f) continue;
        const glm::vec3 a(scene.vertices[scene.indices[3 * t + 0]].position);
        const glm::vec3 b(scene.vertices[scene.indices[3 * t + 1]].position);
        const glm::vec3 c(scene.vertices[scene.indices[3 * t + 2]].position);
        const float area = 0.5f * glm::length(glm::cross(b - a, c - a));
        if (area <= 0.0f) continue;
        scene.lightArea += area;
        scene.lights.push_back({t, scene.lightArea});
    }
    if (scene.lights.empty()) std::cerr << "[obj] warning: " << path << " has no emissive (Ke) materials; the image will be black\n";
    return scene;
}
