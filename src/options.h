#pragma once

#include <glm/glm.hpp>

#include <optional>
#include <string>

struct Options {
    std::string scene;          // OBJ file
    uint32_t width = 1024;
    uint32_t height = 1024;
    uint32_t samplesPerFrame = 4;
    bool frameSppGiven = false;
    uint32_t maxDepth = 8;
    uint32_t targetSpp = 0;     // headless: stop after this many samples per pixel (default 1024)
    bool headless = false;
    bool denoise = true;
    bool compare = false;       // headless: also write the un-denoised image
    float exposure = 1.0f;
    float fov = 39.0f;          // vertical, degrees
    std::optional<glm::vec3> eye, target;
    std::string output = "output.png";
};

/// Parses argv. Prints usage and exits on --help or a bad flag.
Options parseOptions(int argc, char** argv);
