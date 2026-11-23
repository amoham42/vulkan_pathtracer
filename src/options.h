#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>

enum class TraceBackend { Auto, RayQuery, RayPipeline };

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
    bool validation = VPT_VALIDATION_DEFAULT != 0;
    TraceBackend backend = TraceBackend::Auto;
    float exposure = 1.0f;
    float fov = 39.0f;          // vertical, degrees
    std::optional<float> clampLuminance;  // default: 64 when denoising, 0 (off) with --no-denoise
    std::optional<glm::vec3> eye, target;
    std::string output = "output.png";
};

struct ParseFailure {
    bool showHelp = false;
    std::string message;
};

/// Returns the directory containing the running executable (empty if unavailable).
std::filesystem::path executableDir();

/// Returns usage text suitable for --help.
std::string_view usageText();

/// Parses command-line arguments without calling std::exit().
std::expected<Options, ParseFailure> parseOptions(std::span<const char* const> args);
