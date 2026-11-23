#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) rayPayloadInEXT HitPayload hit;

void main() {
    hit.t = -1.0;  // escaped; the scenes are closed rooms, so there is no sky to add
}
