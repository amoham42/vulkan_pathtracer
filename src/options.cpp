#include "options.h"

#include <charconv>
#include <cmath>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace {

constexpr std::string_view kUsage =
    R"(Usage: vulkan-tracer [scene.obj] [options]

  Opens a resizable window and renders progressively. With --headless, renders offscreen
  and writes PNG, JPG, or HDR (selected by --out extension).

Options
  --size WxH          image size                          (default 1024x1024)
  --spp N             headless: exact samples per pixel   (default 1024)
  --frame-spp N       samples per pixel per frame         (default 4, headless 16)
  --depth N           max path length                     (default 8)
  --exposure E        exposure multiplier (> 0)           (default 1)
  --fov DEG           vertical field of view (1..179)     (default 39)
  --clamp L           per-path firefly clamp (0 = off)    (default 64, 0 with --no-denoise)
  --eye X,Y,Z         camera position                     (default: framed on the scene)
  --target X,Y,Z      point the camera looks at
  --no-denoise        show / write the raw path-traced image
  --headless          no window; render to --out and exit
  --compare           headless: also write <out>_noisy.<ext>
  --out FILE          output image (.png, .jpg, .hdr)     (default output.png)
  --validation        enable Khronos validation layers
  --no-validation     disable Khronos validation layers
  -h, --help          this text

Window controls
  WASD / QE  move     right-drag  look     R / F  zoom
  N  denoiser on/off  + / -  exposure      P  save PNG     Esc  quit
)";

std::expected<uint32_t, std::string> parsePositiveUint(std::string_view s) {
    uint32_t v = 0;
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || ptr != s.data() + s.size() || v == 0)
        return std::unexpected(std::format("expected a positive integer, got '{}'", s));
    return v;
}

std::expected<float, std::string> parseFloat(std::string_view s) {
    float v = 0.0f;
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || ptr != s.data() + s.size() || !std::isfinite(v))
        return std::unexpected(std::format("expected a finite number, got '{}'", s));
    return v;
}

std::expected<glm::vec3, std::string> parseVec3(std::string_view s) {
    const auto c1 = s.find(',');
    if (c1 == std::string_view::npos) return std::unexpected(std::format("expected X,Y,Z, got '{}'", s));
    const auto c2 = s.find(',', c1 + 1);
    if (c2 == std::string_view::npos) return std::unexpected(std::format("expected X,Y,Z, got '{}'", s));
    auto x = parseFloat(s.substr(0, c1));
    auto y = parseFloat(s.substr(c1 + 1, c2 - c1 - 1));
    auto z = parseFloat(s.substr(c2 + 1));
    if (!x || !y || !z) return std::unexpected(std::format("expected X,Y,Z, got '{}'", s));
    return glm::vec3{*x, *y, *z};
}

std::expected<std::pair<uint32_t, uint32_t>, std::string> parseSize(std::string_view s) {
    const auto sep = s.find_first_of("xX");
    if (sep == std::string_view::npos) return std::unexpected(std::format("expected WxH, got '{}'", s));
    auto w = parsePositiveUint(s.substr(0, sep));
    auto h = parsePositiveUint(s.substr(sep + 1));
    if (!w || !h) return std::unexpected(std::format("expected positive WxH, got '{}'", s));
    return std::pair{*w, *h};
}

std::string defaultScenePath() {
    constexpr std::string_view rel = "models/CornellBox/CornellBox-Sphere.obj";
    std::error_code ec;
    const auto exe = executableDir();
    if (!exe.empty()) {
        for (const auto& candidate : {exe / "assets" / rel, exe.parent_path() / "assets" / rel,
                                      exe.parent_path().parent_path() / "assets" / rel}) {
            if (std::filesystem::exists(candidate, ec)) return candidate.lexically_normal().make_preferred().string();
        }
    }
    const auto cwdCandidate = std::filesystem::current_path(ec) / "assets" / rel;
    if (std::filesystem::exists(cwdCandidate, ec)) return cwdCandidate.lexically_normal().make_preferred().string();
    return (std::filesystem::path(VPT_ASSET_DIR) / rel).lexically_normal().make_preferred().string();
}

}  // namespace

std::filesystem::path executableDir() {
#if defined(_WIN32)
    std::wstring buf(MAX_PATH, L'\0');
    for (;;) {
        const DWORD len = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
        if (len == 0) return {};
        if (len < buf.size()) {
            buf.resize(len);
            return std::filesystem::path(buf).parent_path();
        }
        buf.resize(buf.size() * 2);
    }
#elif defined(__linux__)
    std::error_code ec;
    const auto p = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::filesystem::path{} : p.parent_path();
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buf(size, '\0');
    if (_NSGetExecutablePath(buf.data(), &size) != 0) return {};
    return std::filesystem::path(buf.c_str()).parent_path();
#else
    return {};
#endif
}

std::string_view usageText() { return kUsage; }

std::expected<Options, ParseFailure> parseOptions(std::span<const char* const> args) {
    Options o;
    auto fail = [](std::string msg) -> std::unexpected<ParseFailure> {
        return std::unexpected(ParseFailure{false, std::move(msg)});
    };

    for (size_t i = 1; i < args.size(); i++) {
        const std::string_view a = args[i];
        auto next = [&]() -> std::expected<std::string_view, ParseFailure> {
            if (i + 1 >= args.size()) return fail(std::format("{} requires a value", a));
            return std::string_view(args[++i]);
        };

        if (a == "-h" || a == "--help") {
            return std::unexpected(ParseFailure{true, {}});
        } else if (a == "--size") {
            auto val = next();
            if (!val) return std::unexpected(val.error());
            auto sz = parseSize(*val);
            if (!sz) return fail(sz.error());
            std::tie(o.width, o.height) = *sz;
        } else if (a == "--spp") {
            auto val = next();
            if (!val) return std::unexpected(val.error());
            auto v = parsePositiveUint(*val);
            if (!v) return fail(v.error());
            o.targetSpp = *v;
        } else if (a == "--frame-spp") {
            auto val = next();
            if (!val) return std::unexpected(val.error());
            auto v = parsePositiveUint(*val);
            if (!v) return fail(v.error());
            o.samplesPerFrame = *v;
            o.frameSppGiven = true;
        } else if (a == "--depth") {
            auto val = next();
            if (!val) return std::unexpected(val.error());
            auto v = parsePositiveUint(*val);
            if (!v) return fail(v.error());
            o.maxDepth = *v;
        } else if (a == "--exposure") {
            auto val = next();
            if (!val) return std::unexpected(val.error());
            auto v = parseFloat(*val);
            if (!v || *v <= 0.0f) return fail(std::format("--exposure must be > 0, got '{}'", *val));
            o.exposure = *v;
        } else if (a == "--fov") {
            auto val = next();
            if (!val) return std::unexpected(val.error());
            auto v = parseFloat(*val);
            if (!v || *v <= 0.5f || *v >= 179.5f) return fail(std::format("--fov must be in (0.5, 179.5), got '{}'", *val));
            o.fov = *v;
        } else if (a == "--clamp") {
            auto val = next();
            if (!val) return std::unexpected(val.error());
            auto v = parseFloat(*val);
            if (!v || *v < 0.0f) return fail(std::format("--clamp must be >= 0, got '{}'", *val));
            o.clampLuminance = *v;
        } else if (a == "--eye") {
            auto val = next();
            if (!val) return std::unexpected(val.error());
            auto v = parseVec3(*val);
            if (!v) return fail(v.error());
            o.eye = *v;
        } else if (a == "--target") {
            auto val = next();
            if (!val) return std::unexpected(val.error());
            auto v = parseVec3(*val);
            if (!v) return fail(v.error());
            o.target = *v;
        } else if (a == "--no-denoise") {
            o.denoise = false;
        } else if (a == "--headless") {
            o.headless = true;
        } else if (a == "--compare") {
            o.compare = true;
        } else if (a == "--validation") {
            o.validation = true;
        } else if (a == "--no-validation") {
            o.validation = false;
        } else if (a == "--out") {
            auto val = next();
            if (!val) return std::unexpected(val.error());
            o.output = std::string(*val);
        } else if (!a.empty() && a[0] == '-') {
            return fail(std::format("unknown option '{}'", a));
        } else {
            o.scene = std::string(a);
        }
    }

    if (o.scene.empty()) o.scene = defaultScenePath();
    if (o.headless && o.targetSpp == 0) o.targetSpp = 1024;
    if (o.headless && !o.frameSppGiven) o.samplesPerFrame = 16;
    return o;
}
