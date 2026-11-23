// Shared by every ray-tracing stage. Struct layouts mirror src/scene.h (scalar block layout).

#ifndef COMMON_GLSL
#define COMMON_GLSL

const float PI = 3.14159265358979323846;

const uint MAT_DIFFUSE = 0u;
const uint MAT_GLOSSY = 1u;
const uint MAT_MIRROR = 2u;
const uint MAT_DIELECTRIC = 3u;

// Bit 31 of materialIds[tri] indicates the triangle has authored smooth vertex normals.
const uint TRI_SMOOTH_NORMAL_BIT = 0x80000000u;

struct GpuMaterial {
    vec4 diffuse;   // rgb albedo, w = type
    vec4 specular;  // rgb, w = Phong exponent
    vec4 emission;  // rgb radiance, w = ior (Dielectric) or specularProbability (Glossy)
};

// Pre-unpacked emissive triangle so sampleLight() needs zero index/vertex/material pointer chasing.
struct GpuLight {
    vec4 v0Cdf;      // xyz = vertex A, w = running area CDF
    vec4 e1;         // xyz = vertex B - A
    vec4 e2;         // xyz = vertex C - A
    vec4 normalPad;  // xyz = unit geometric normal
    vec4 emission;   // rgb = emitted radiance
};

struct HitPayload {
    vec3 position;
    float t;          // < 0: the ray escaped
    vec3 normal;      // shading normal, on the side the ray came from
    uint material;
    vec3 geoNormal;   // face normal, on the side the ray came from
    uint frontFace;   // 1 when the ray hit the outside of the surface
};

// Decodes a 32-bit octahedral snorm16x2 normal back into a unit vector (Cigolle et al., JCGT 2014).
vec3 unpackOctNormal(uint packed) {
    vec2 e = unpackSnorm2x16(packed);
    vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        vec2 s = vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
        v.xy = (1.0 - abs(v.yx)) * s;
    }
    return normalize(v);
}

// ---- random numbers: PCG (O'Neill) + Roberts R6 low-discrepancy Kronecker sequence ----

uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rnd(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    word = (word >> 22u) ^ word;
    return float(word >> 8u) * (1.0 / 16777216.0);
}

// Constant-time 2D low-discrepancy sample using Cranley-Patterson per-pixel rotation (zero loops, zero LUTs).
vec2 qmcPair(uint sampleIdx, uint pixSeed, vec2 alpha, uint rot) {
    const uint h = (pixSeed << rot) | (pixSeed >> (32u - rot));
    const vec2 shift = vec2(h & 0xffffu, h >> 16u) * (1.0 / 65536.0);
    return fract(shift + float(sampleIdx) * alpha);
}

float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

#endif
