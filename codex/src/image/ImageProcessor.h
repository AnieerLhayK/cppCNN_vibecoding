#pragma once

#include "cnn/Tensor.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace cppcnn {

struct RawImage {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<std::uint8_t> pixels;

    [[nodiscard]] bool empty() const noexcept;
};

class ImageProcessor {
public:
    static RawImage load(const std::filesystem::path& path);
    static RawImage resize(const RawImage& image, int targetWidth, int targetHeight);
    static Tensor normalize(const RawImage& image);
    static Tensor loadAndPreprocess(
        const std::filesystem::path& path,
        int targetWidth = 32,
        int targetHeight = 32);

private:
    static RawImage loadPpm(const std::filesystem::path& path);
};

}  // namespace cppcnn
