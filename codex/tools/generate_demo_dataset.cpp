#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int imageSize = 32;
constexpr int channels = 3;
using PixelBuffer = std::vector<std::uint8_t>;

std::size_t offset(const int x, const int y, const int channel) {
    return (static_cast<std::size_t>(y) * imageSize + static_cast<std::size_t>(x)) * channels
        + static_cast<std::size_t>(channel);
}

void setPixel(
    PixelBuffer& pixels,
    const int x,
    const int y,
    const std::array<int, 3>& color) {
    if (x < 0 || y < 0 || x >= imageSize || y >= imageSize) {
        return;
    }
    for (int channel = 0; channel < channels; ++channel) {
        pixels[offset(x, y, channel)] = static_cast<std::uint8_t>(
            std::clamp(color[static_cast<std::size_t>(channel)], 0, 255));
    }
}

void drawPattern(PixelBuffer& pixels, const int classId, const int shiftX, const int shiftY) {
    const std::array<int, 3> foreground =
        classId == 0 ? std::array<int, 3>{225, 45, 45} : std::array<int, 3>{35, 100, 225};
    const int centerX = imageSize / 2 + shiftX;
    const int centerY = imageSize / 2 + shiftY;

    if (classId == 0) {
        for (int y = 5; y < imageSize - 5; ++y) {
            for (int x = 5; x < imageSize - 5; ++x) {
                const int distanceSquared =
                    (x - centerX) * (x - centerX) + (y - centerY) * (y - centerY);
                if (distanceSquared >= 80 && distanceSquared <= 125) {
                    setPixel(pixels, x, y, foreground);
                }
            }
        }
    } else {
        for (int y = 6; y < imageSize - 5; ++y) {
            const int halfWidth = (y - 5) / 2;
            for (int x = centerX - halfWidth; x <= centerX + halfWidth; ++x) {
                if (y + shiftY >= 0 && (x == centerX - halfWidth
                    || x == centerX + halfWidth
                    || y == imageSize - 6)) {
                    setPixel(pixels, x, y + shiftY, foreground);
                }
            }
        }
    }
}

PixelBuffer createImage(
    const int classId,
    std::mt19937& generator) {
    std::uniform_int_distribution<int> noise(-12, 12);
    std::uniform_int_distribution<int> shift(-2, 2);
    PixelBuffer pixels(static_cast<std::size_t>(imageSize * imageSize * channels));

    for (std::uint8_t& value : pixels) {
        value = static_cast<std::uint8_t>(std::clamp(235 + noise(generator), 0, 255));
    }
    drawPattern(pixels, classId, shift(generator), shift(generator));
    return pixels;
}

void writePpm(const std::filesystem::path& path, const PixelBuffer& pixels) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Could not create demo image: " + path.string());
    }
    output << "P6\n" << imageSize << ' ' << imageSize << "\n255\n";
    output.write(
        reinterpret_cast<const char*>(pixels.data()),
        static_cast<std::streamsize>(pixels.size()));
}

void generateSplit(
    const std::filesystem::path& root,
    const std::string& split,
    const int imagesPerClass,
    std::mt19937& generator) {
    for (int classId = 0; classId < 2; ++classId) {
        const auto directory =
            root / split / (classId == 0 ? "00000" : "00001");
        std::filesystem::create_directories(directory);
        for (int index = 0; index < imagesPerClass; ++index) {
            writePpm(
                directory / ("sample_" + std::to_string(index) + ".ppm"),
                createImage(classId, generator));
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const std::filesystem::path outputRoot =
            argc > 1 ? std::filesystem::path(argv[1]) : "demo_data";
        std::mt19937 generator(2026);
        generateSplit(outputRoot, "train", 20, generator);
        generateSplit(outputRoot, "test", 6, generator);
        std::cout << "Demo dataset generated under: "
                  << std::filesystem::absolute(outputRoot).string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
