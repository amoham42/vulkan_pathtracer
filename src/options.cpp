#include "options.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>

namespace {

[[noreturn]] void usage(int code) {
    std::cout <<
        R"(Usage: vulkan-tracer [scene.obj] [options]

  Opens a window and renders progressively. With --headless, renders offscreen and writes a PNG.

Options
  --size WxH          image size                          (default 1024x1024)
  --spp N             headless: samples per pixel         (default 1024)
  --frame-spp N       samples per pixel per frame         (default 4, headless 16)
  --depth N           max path length                     (default 8)
  --exposure E        exposure multiplier                 (default 1)
  --fov DEG           vertical field of view              (default 39)
  --eye X,Y,Z         camera position                     (default: framed on the scene)
  --target X,Y,Z      point the camera looks at
  --no-denoise        show / write the raw path-traced image
  --headless          no window; render to --out and exit
  --compare           headless: also write <out>_noisy.png
  --out FILE          output PNG                          (default output.png)
  -h, --help          this text

Window controls
  WASD / QE  move     right-drag  look     R / F  zoom
  N  denoiser on/off  + / -  exposure      P  save PNG     Esc  quit
)";
    std::exit(code);
}

glm::vec3 parseVec3(const char* s) {
    glm::vec3 v;
    if (std::sscanf(s, "%f,%f,%f", &v.x, &v.y, &v.z) != 3) {
        std::cerr << "expected X,Y,Z but got '" << s << "'\n";
        usage(1);
    }
    return v;
}

uint32_t parseUint(const char* s) {
    char* end = nullptr;
    const unsigned long v = std::strtoul(s, &end, 10);
    if (!end || *end || v == 0) {
        std::cerr << "expected a positive integer but got '" << s << "'\n";
        usage(1);
    }
    return static_cast<uint32_t>(v);
}

}  // namespace

Options parseOptions(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i++) {
        const std::string_view a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::cerr << a << " needs a value\n";
                usage(1);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") usage(0);
        else if (a == "--size") {
            if (std::sscanf(next(), "%ux%u", &o.width, &o.height) != 2 || !o.width || !o.height) usage(1);
        } else if (a == "--spp") o.targetSpp = parseUint(next());
        else if (a == "--frame-spp") o.samplesPerFrame = parseUint(next()), o.frameSppGiven = true;
        else if (a == "--depth") o.maxDepth = parseUint(next());
        else if (a == "--exposure") o.exposure = std::strtof(next(), nullptr);
        else if (a == "--fov") o.fov = std::strtof(next(), nullptr);
        else if (a == "--eye") o.eye = parseVec3(next());
        else if (a == "--target") o.target = parseVec3(next());
        else if (a == "--no-denoise") o.denoise = false;
        else if (a == "--headless") o.headless = true;
        else if (a == "--compare") o.compare = true;
        else if (a == "--out") o.output = next();
        else if (!a.empty() && a[0] == '-') {
            std::cerr << "unknown option " << a << "\n";
            usage(1);
        } else o.scene = a;
    }
    if (o.scene.empty()) o.scene = std::string(VPT_ASSET_DIR) + "/CornellBox/CornellBox-Sphere.obj";
    if (o.headless && o.targetSpp == 0) o.targetSpp = 1024;
    return o;
}
