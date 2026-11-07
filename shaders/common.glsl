// Shared by every ray-tracing stage. Struct layouts mirror src/scene.h (scalar block layout).

#ifndef COMMON_GLSL
#define COMMON_GLSL

const float PI = 3.14159265358979323846;

const uint MAT_DIFFUSE = 0u;
const uint MAT_GLOSSY = 1u;
const uint MAT_MIRROR = 2u;
const uint MAT_DIELECTRIC = 3u;

struct GpuVertex {
    vec4 position;
    vec4 normal;  // w = 1 when the mesh supplied a normal
};

struct GpuMaterial {
    vec4 diffuse;   // rgb albedo, w = type
    vec4 specular;  // rgb, w = Phong exponent
    vec4 emission;  // rgb radiance, w = index of refraction
};

struct GpuLight {
    uint triangle;
    float cdf;
};

struct HitPayload {
    vec3 position;
    float t;          // < 0: the ray escaped
    vec3 normal;      // shading normal, on the side the ray came from
    uint material;
    vec3 geoNormal;   // face normal, on the side the ray came from
    uint frontFace;   // 1 when the ray hit the outside of the surface
};

// ---- random numbers: PCG (O'Neill), one 32-bit state per path ----

uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rnd(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    word = (word >> 22u) ^ word;
    return float(word) * (1.0 / 4294967296.0);
}

float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

#endif
