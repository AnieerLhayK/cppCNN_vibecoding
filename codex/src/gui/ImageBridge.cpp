#include "gui/ImageBridge.h"

#include <stdexcept>

cppcnn::Tensor ImageBridge::toTensor(
    const QImage& image,
    const int targetWidth,
    const int targetHeight) {
    if (image.isNull()) {
        throw std::invalid_argument("Cannot preprocess an empty image.");
    }

    const QImage converted = image
        .convertedTo(QImage::Format_RGB888)
        .scaled(targetWidth, targetHeight, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);

    cppcnn::Tensor tensor(3, static_cast<std::size_t>(targetHeight), static_cast<std::size_t>(targetWidth));
    for (int y = 0; y < targetHeight; ++y) {
        const auto* line = converted.constScanLine(y);
        for (int x = 0; x < targetWidth; ++x) {
            const int offset = x * 3;
            tensor.at(0, static_cast<std::size_t>(y), static_cast<std::size_t>(x)) =
                static_cast<float>(line[offset]) / 255.0F;
            tensor.at(1, static_cast<std::size_t>(y), static_cast<std::size_t>(x)) =
                static_cast<float>(line[offset + 1]) / 255.0F;
            tensor.at(2, static_cast<std::size_t>(y), static_cast<std::size_t>(x)) =
                static_cast<float>(line[offset + 2]) / 255.0F;
        }
    }
    return tensor;
}
