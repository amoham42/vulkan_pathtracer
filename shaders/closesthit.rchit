#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : enable
#include "common.glsl"

layout(binding = 2, set = 0) buffer Vertices{float vertices[];};
layout(binding = 3, set = 0) buffer Indices{uint indices[];};
layout(binding = 4, set = 0) buffer Faces{float faces[];};

layout(location = 0) rayPayloadInEXT HitPayload payload;
hitAttributeEXT vec3 attribs;

struct Vertex {
    vec3 position;
    vec3 normal;
};

struct Face {
    vec3 diffuse;
    vec3 emission;
    vec3 specular;
    vec3 transmittance;
    float shininess;
    float ior;
    float illum;
};

Vertex unpackVertex(uint index) {

    uint stride = 6;
    uint offset = index * stride;
    Vertex v;
    v.position = vec3(vertices[offset +  0], vertices[offset +  1], vertices[offset + 2]);
    v.normal = vec3(vertices[offset +  3], vertices[offset +  4], vertices[offset + 5]);

    return v;
}

Face unpackFace(uint index) {

    uint stride = 15;
    uint offset = index * stride;
    Face f;
    f.diffuse = vec3(faces[offset +  0], faces[offset +  1], faces[offset + 2]);
    f.emission = vec3(faces[offset +  3], faces[offset +  4], faces[offset + 5]);
    f.specular = vec3(faces[offset + 6], faces[offset + 7], faces[offset + 8]);
    f.transmittance = vec3(faces[offset + 9], faces[offset + 10], faces[offset + 11]);
    f.shininess = faces[offset + 12];
    f.ior = faces[offset + 13];
    f.illum = faces[offset + 14];
    return f;
}

vec3 calcNormal(Vertex v0, Vertex v1, Vertex v2) {

    vec3 e01 = v1.normal - v0.normal;
    vec3 e02 = v2.normal - v0.normal;
    return -normalize(cross(e01, e02));
}

void main() {

    const Vertex v0 = unpackVertex(indices[3 * gl_PrimitiveID + 0]);
    const Vertex v1 = unpackVertex(indices[3 * gl_PrimitiveID + 1]);
    const Vertex v2 = unpackVertex(indices[3 * gl_PrimitiveID + 2]);

    const vec3 barycentricCoords = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    const vec3 position = v0.position * barycentricCoords.x +
                          v1.position * barycentricCoords.y +
                          v2.position * barycentricCoords.z;
    const vec3 normal = v0.normal * barycentricCoords.x +
                        v1.normal * barycentricCoords.y +
                        v2.normal * barycentricCoords.z;

    const Face face = unpackFace(gl_PrimitiveID);
    payload.brdf = face.diffuse / M_PI;
    payload.emission = face.emission / 2.0;
    payload.position = position;
    payload.normal = normal;
    payload.specular = face.specular;
    payload.transmittance = face.transmittance;
    payload.shininess = face.shininess;
    payload.ior = face.ior;
    payload.illum = face.illum;
}
