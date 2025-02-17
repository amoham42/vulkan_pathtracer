#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
};

struct Face {
    float diffuse[3];
    float emission[3];
    float specular[3];
    float transmittance[3];
    float shininess;
    float ior;
    float illum;
};

inline void loadFromFile(std::vector<Vertex>& vertices,
                         std::vector<uint32_t>& indices,
                         std::vector<Face>& faces,
                         std::string file) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                           file.c_str(),
                           "../assets/CornellBox")) {
        throw std::runtime_error(warn + err);
    }

    for (const auto& shape : shapes) {
        size_t startIndex = vertices.size();
        bool hasNormals = !attrib.normals.empty();

        for (size_t i = 0; i < shape.mesh.indices.size(); ++i) {
            const auto& idx = shape.mesh.indices[i];
            Vertex vertex{};
            vertex.position.x = attrib.vertices[3 * idx.vertex_index + 0];
            vertex.position.y= -attrib.vertices[3 * idx.vertex_index + 1];
            vertex.position.z = attrib.vertices[3 * idx.vertex_index + 2];

            if (hasNormals && idx.normal_index >= 0) {
                vertex.normal.x = attrib.normals[3 * idx.normal_index + 0];
                vertex.normal.y = -attrib.normals[3 * idx.normal_index + 1];
                vertex.normal.z = attrib.normals[3 * idx.normal_index + 2];
            } else {
                vertex.normal.x = 0.f;
                vertex.normal.y = 0.f;
                vertex.normal.z = 0.f;
            }

            vertices.push_back(vertex);
            indices.push_back(static_cast<uint32_t>(indices.size()));
        }

        if (!hasNormals) {

            for (size_t i = startIndex; i + 2 < vertices.size(); i += 3) {
                glm::vec3 vec0 = vertices[i].position;
                glm::vec3 vec1 = vertices[i + 1].position;
                glm::vec3 vec2 = vertices[i + 2].position;

                glm::vec3 faceNormal = glm::normalize(glm::cross(vec1 - vec0, vec2 - vec0));

                vertices[i].normal     = faceNormal;
                vertices[i + 1].normal = faceNormal;
                vertices[i + 2].normal = faceNormal;
            }
        }

        for (const auto& matIndex : shape.mesh.material_ids) {
            Face face;
            face.diffuse[0] = materials[matIndex].diffuse[0];
            face.diffuse[1] = materials[matIndex].diffuse[1];
            face.diffuse[2] = materials[matIndex].diffuse[2];

            face.emission[0] = materials[matIndex].emission[0];
            face.emission[1] = materials[matIndex].emission[1];
            face.emission[2] = materials[matIndex].emission[2];

            face.specular[0] = materials[matIndex].specular[0];
            face.specular[1] = materials[matIndex].specular[1];
            face.specular[2] = materials[matIndex].specular[2];

            face.transmittance[0] = materials[matIndex].transmittance[0];
            face.transmittance[1] = materials[matIndex].transmittance[1];
            face.transmittance[2] = materials[matIndex].transmittance[2];

            face.shininess = materials[matIndex].shininess;
            face.ior = materials[matIndex].ior;
            face.illum = materials[matIndex].illum;

            faces.push_back(face);
        }
    }
}


inline std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}
