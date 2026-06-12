#pragma once

#include "cnn/Tensor.h"

#include <QImage>

class ImageBridge {
public:
    [[nodiscard]] static cppcnn::Tensor toTensor(
        const QImage& image,
        int targetWidth = 32,
        int targetHeight = 32);
};
