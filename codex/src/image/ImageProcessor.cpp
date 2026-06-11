#include "image/ImageProcessor.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef CPP_CNN_WITH_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace cppcnn {

namespace {

std::string readPpmToken(std::istream& input) {
    std::string token;
    char character = '\0';

    while (input.get(character)) {
        if (character == '#') {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(character))) {
            token.push_back(character);
            break;
        }
    }

    while (input.get(character)) {
        if (character == '#') {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            break;
        }
        if (std::isspace(static_cast<unsigned char>(character))) {
            break;
        }
        token.push_back(character);
    }
    return token;
}

std::size_t pixelOffset(
    const RawImage& image,
    const int x,
    const int y,
    const int channel) {
    return (static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width)
            + static_cast<std::size_t>(x))
        * static_cast<std::size_t>(image.channels)
        + static_cast<std::size_t>(channel);
}

}  // namespace

bool RawImage::empty() const noexcept {
    return width <= 0 || height <= 0 || channels <= 0 || pixels.empty();
}

RawImage ImageProcessor::load(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Image file does not exist: " + path.string());
    }

    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](const unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    if (extension == ".ppm" || extension == ".pnm") {
        return loadPpm(path);
    }

#ifdef CPP_CNN_WITH_OPENCV
    const cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("OpenCV could not read image: " + path.string());
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    RawImage result;
    result.width = rgb.cols;
    result.height = rgb.rows;
    result.channels = rgb.channels();
    result.pixels.assign(rgb.data, rgb.data + rgb.total() * rgb.elemSize());
    return result;
#else
    throw std::runtime_error(
        "This build supports PPM images only. Install OpenCV for PNG/JPEG/BMP support: "
        + path.string());
#endif
}

RawImage ImageProcessor::resize(
    const RawImage& image,
    const int targetWidth,
    const int targetHeight) {
    if (image.empty()) {
        throw std::invalid_argument("Cannot resize an empty image.");
    }
    if (targetWidth <= 0 || targetHeight <= 0) {
        throw std::invalid_argument("Target image dimensions must be positive.");
    }

    RawImage result;
    result.width = targetWidth;
    result.height = targetHeight;
    result.channels = image.channels;
    result.pixels.resize(
        static_cast<std::size_t>(targetWidth)
        * static_cast<std::size_t>(targetHeight)
        * static_cast<std::size_t>(image.channels));

    const float xScale = static_cast<float>(image.width) / static_cast<float>(targetWidth);
    const float yScale = static_cast<float>(image.height) / static_cast<float>(targetHeight);

    for (int targetY = 0; targetY < targetHeight; ++targetY) {
        const float sourceY = (static_cast<float>(targetY) + 0.5F) * yScale - 0.5F;
        const int y0 = std::clamp(static_cast<int>(std::floor(sourceY)), 0, image.height - 1);
        const int y1 = std::min(y0 + 1, image.height - 1);
        const float yWeight = std::clamp(sourceY - static_cast<float>(y0), 0.0F, 1.0F);

        for (int targetX = 0; targetX < targetWidth; ++targetX) {
            const float sourceX = (static_cast<float>(targetX) + 0.5F) * xScale - 0.5F;
            const int x0 = std::clamp(static_cast<int>(std::floor(sourceX)), 0, image.width - 1);
            const int x1 = std::min(x0 + 1, image.width - 1);
            const float xWeight = std::clamp(sourceX - static_cast<float>(x0), 0.0F, 1.0F);

            for (int channel = 0; channel < image.channels; ++channel) {
                const float top = static_cast<float>(image.pixels[pixelOffset(image, x0, y0, channel)])
                        * (1.0F - xWeight)
                    + static_cast<float>(image.pixels[pixelOffset(image, x1, y0, channel)]) * xWeight;
                const float bottom = static_cast<float>(image.pixels[pixelOffset(image, x0, y1, channel)])
                        * (1.0F - xWeight)
                    + static_cast<float>(image.pixels[pixelOffset(image, x1, y1, channel)]) * xWeight;
                const float value = top * (1.0F - yWeight) + bottom * yWeight;
                result.pixels[pixelOffset(result, targetX, targetY, channel)] =
                    static_cast<std::uint8_t>(std::clamp(std::lround(value), 0L, 255L));
            }
        }
    }
    return result;
}

Tensor ImageProcessor::normalize(const RawImage& image) {
    if (image.empty()) {
        throw std::invalid_argument("Cannot normalize an empty image.");
    }

    Tensor tensor(
        static_cast<std::size_t>(image.channels),
        static_cast<std::size_t>(image.height),
        static_cast<std::size_t>(image.width));

    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            for (int channel = 0; channel < image.channels; ++channel) {
                tensor.at(
                    static_cast<std::size_t>(channel),
                    static_cast<std::size_t>(y),
                    static_cast<std::size_t>(x)) =
                    static_cast<float>(image.pixels[pixelOffset(image, x, y, channel)]) / 255.0F;
            }
        }
    }
    return tensor;
}

Tensor ImageProcessor::loadAndPreprocess(
    const std::filesystem::path& path,
    const int targetWidth,
    const int targetHeight) {
    return normalize(resize(load(path), targetWidth, targetHeight));
}

RawImage ImageProcessor::loadPpm(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open PPM image: " + path.string());
    }

    const std::string magic = readPpmToken(input);
    if (magic != "P6" && magic != "P3") {
        throw std::runtime_error("Unsupported PPM format in: " + path.string());
    }

    const int width = std::stoi(readPpmToken(input));
    const int height = std::stoi(readPpmToken(input));
    const int maximumValue = std::stoi(readPpmToken(input));
    if (width <= 0 || height <= 0 || maximumValue <= 0 || maximumValue > 255) {
        throw std::runtime_error("Invalid PPM header in: " + path.string());
    }

    RawImage result;
    result.width = width;
    result.height = height;
    result.channels = 3;
    result.pixels.resize(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U);

    if (magic == "P6") {
        input.read(
            reinterpret_cast<char*>(result.pixels.data()),
            static_cast<std::streamsize>(result.pixels.size()));
        if (input.gcount() != static_cast<std::streamsize>(result.pixels.size())) {
            throw std::runtime_error("PPM pixel data is incomplete: " + path.string());
        }
    } else {
        for (auto& value : result.pixels) {
            const std::string token = readPpmToken(input);
            if (token.empty()) {
                throw std::runtime_error("PPM pixel data is incomplete: " + path.string());
            }
            value = static_cast<std::uint8_t>(std::stoi(token));
        }
    }

    if (maximumValue != 255) {
        for (auto& value : result.pixels) {
            value = static_cast<std::uint8_t>(
                static_cast<int>(value) * 255 / maximumValue);
        }
    }
    return result;
}

}  // namespace cppcnn
