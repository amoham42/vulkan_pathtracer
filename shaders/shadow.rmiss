#version 460
#extension GL_EXT_ray_tracing : require

// Shadow rays skip the closest-hit shader and stop at the first hit, so only reaching this
// shader proves the light is visible.
layout(location = 1) rayPayloadInEXT uint shadowVisible;

void main() {
    shadowVisible = 1u;
}
