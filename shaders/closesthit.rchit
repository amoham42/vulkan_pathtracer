#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(set = 0, binding = 2, scalar) readonly buffer Vertices { GpuVertex vertices[]; };
layout(set = 0, binding = 3) readonly buffer Indices { uint indices[]; };
layout(set = 0, binding = 5) readonly buffer MaterialIds { uint materialIds[]; };

layout(location = 0) rayPayloadInEXT HitPayload hit;
hitAttributeEXT vec2 bary;

void main() {
    const uint tri = gl_PrimitiveID;
    const GpuVertex v0 = vertices[indices[3 * tri + 0]];
    const GpuVertex v1 = vertices[indices[3 * tri + 1]];
    const GpuVertex v2 = vertices[indices[3 * tri + 2]];
    const vec3 b = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);

    const vec3 p0 = v0.position.xyz, p1 = v1.position.xyz, p2 = v2.position.xyz;
    vec3 ng = normalize(cross(p1 - p0, p2 - p0));
    vec3 ns = ng;
    if (v0.normal.w > 0.0) {
        const vec3 n = v0.normal.xyz * b.x + v1.normal.xyz * b.y + v2.normal.xyz * b.z;
        if (dot(n, n) > 1e-12) ns = normalize(n);
        // authored normals say which side is outside; make the winding agree with them
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
    hit.material = materialIds[tri];
    hit.geoNormal = ng;
    hit.frontFace = front ? 1u : 0u;
}
