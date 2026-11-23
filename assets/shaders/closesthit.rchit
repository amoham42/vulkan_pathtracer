#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(set = 0, binding = 2, scalar) readonly buffer Positions { vec3 positions[]; };
layout(set = 0, binding = 3) readonly buffer Indices { uint indices[]; };
layout(set = 0, binding = 5) readonly buffer MaterialIds { uint materialIds[]; };
layout(set = 0, binding = 9) readonly buffer Normals { uint packedNormals[]; };

layout(location = 0) rayPayloadInEXT HitPayload hit;
hitAttributeEXT vec2 bary;

void main() {
    const uint tri = gl_PrimitiveID;
    const uint i0 = indices[3u * tri + 0u];
    const uint i1 = indices[3u * tri + 1u];
    const uint i2 = indices[3u * tri + 2u];
    const vec3 b = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);

    const vec3 p0 = positions[i0], p1 = positions[i1], p2 = positions[i2];
    vec3 ng = normalize(cross(p1 - p0, p2 - p0));
    vec3 ns = ng;
    const uint matWord = materialIds[tri];
    if ((matWord & TRI_SMOOTH_NORMAL_BIT) != 0u) {
        const vec3 n = unpackOctNormal(packedNormals[i0]) * b.x +
                       unpackOctNormal(packedNormals[i1]) * b.y +
                       unpackOctNormal(packedNormals[i2]) * b.z;
        if (dot(n, n) > 1e-12) ns = normalize(n);
        if (dot(ng, ns) < 0.0) ng = -ng;
    }

    const bool front = dot(ng, gl_WorldRayDirectionEXT) < 0.0;
    if (!front) {
        ng = -ng;
        ns = -ns;
    }

    hit.position = p0 * b.x + p1 * b.y + p2 * b.z;
    hit.t = gl_HitTEXT;
    hit.normal = ns;
    hit.material = matWord & ~TRI_SMOOTH_NORMAL_BIT;
    hit.geoNormal = ng;
    hit.frontFace = front ? 1u : 0u;
}
