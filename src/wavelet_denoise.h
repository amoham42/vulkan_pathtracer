#pragma once
#include <cmath>
#include <vector>
#include <algorithm>

inline void haar2DTransform(std::vector<float>& data, const int width, const int height) {
    for (int level = 0; level < 2; level++) {
        int w = width >> level;
        int h = height >> level;
        if (w < 2 || h < 2)
            break;

        std::vector<float> temp(data.size());

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x += 2) {
                int i = y * width + x;
                int j = y * width + (x + 1);
                temp[y * width + (x / 2)] = (data[i] + data[j]) / std::sqrt(2.0f);
                temp[y * width + (w / 2) + (x / 2)] = (data[i] - data[j]) / std::sqrt(2.0f);
            }
        }

        std::vector<float> temp2(data.size());
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y += 2) {
                int i = y * width + x;
                int j = (y + 1) * width + x;
                temp2[(y / 2) * width + x] = (temp[i] + temp[j]) / std::sqrt(2.0f);
                temp2[((h / 2) + (y / 2)) * width + x] = (temp[i] - temp[j]) / std::sqrt(2.0f);
            }
        }
        data = std::move(temp2);
    }
}

inline void inverseHaar2DTransform(std::vector<float>& data, const int width, const int height) {
    for (int level = 1; level >= 0; level--) {
        int w = width >> level;
        int h = height >> level;
        if (w < 2 || h < 2)
            continue;

        std::vector<float> temp(data.size());

        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h / 2; y++) {
                int avgIndex = y * width + x;
                int diffIndex = (y + h / 2) * width + x;
                float avg = data[avgIndex];
                float diff = data[diffIndex];
                temp[(2 * y) * width + x]     = (avg + diff) / std::sqrt(2.0f);
                temp[(2 * y + 1) * width + x]   = (avg - diff) / std::sqrt(2.0f);
            }
        }

        // Inverse horizontal transform:
        std::vector<float> temp2(data.size());
        for (int y = 0; y < h; y++) {
            // In each row, 'w' was halved in the horizontal transform.
            for (int x = 0; x < w / 2; x++) {
                int avgIndex = y * width + x;
                int diffIndex = y * width + (x + w / 2);
                float avg = temp[avgIndex];
                float diff = temp[diffIndex];
                temp2[y * width + 2 * x]     = (avg + diff) / std::sqrt(2.0f);
                temp2[y * width + 2 * x + 1]   = (avg - diff) / std::sqrt(2.0f);
            }
        }
        data = std::move(temp2);
    }
}

inline void thresholdDetailCoefficients(std::vector<float>& data, const int width, const int height, float threshold) {
    const int halfWidth = width / 2;
    const int halfHeight = height / 2;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if ((x >= halfWidth) || (y >= halfHeight)) {
                int idx = y * width + x;
                if (std::abs(data[idx]) < threshold) {
                    data[idx] = 0.0f;
                }
            }
        }
    }
}

inline void denoiseChannel(std::vector<float>& channel, const int width, const int height, const float threshold) {
    haar2DTransform(channel, width, height);
    thresholdDetailCoefficients(channel, width, height, threshold);
    inverseHaar2DTransform(channel, width, height);
}

inline void waveletDenoiseImage(unsigned char* image, const int width, const int height, const int channels, const float threshold) {
    const int totalPixels = width * height;
    std::vector<std::vector<float>> channelData(channels, std::vector<float>(totalPixels));
    for (int i = 0; i < totalPixels; i++) {
        for (int c = 0; c < channels; c++) {
            channelData[c][i] = static_cast<float>(image[i * channels + c]);
        }
    }
    for (int c = 0; c < channels; c++) {
        denoiseChannel(channelData[c], width, height, threshold);
    }
    for (int i = 0; i < totalPixels; i++) {
        for (int c = 0; c < channels; c++) {
            int value = static_cast<int>(std::round(channelData[c][i]));
            value = std::clamp(value, 0, 255);
            image[i * channels + c] = static_cast<unsigned char>(value);
        }
    }
}
