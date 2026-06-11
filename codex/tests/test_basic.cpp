#include "cnn/Tensor.h"
#include "image/ImageProcessor.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

void expect(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expectNear(const float actual, const float expected, const float tolerance, const char* message) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

void testTensor() {
    cppcnn::Tensor tensor(3, 2, 4, 0.25F);
    expect(tensor.channels() == 3, "Tensor channel count is incorrect.");
    expect(tensor.height() == 2, "Tensor height is incorrect.");
    expect(tensor.width() == 4, "Tensor width is incorrect.");
    expect(tensor.size() == 24, "Tensor element count is incorrect.");

    tensor.at(2, 1, 3) = 0.75F;
    expectNear(tensor.at(2, 1, 3), 0.75F, 1.0e-6F, "Tensor indexing failed.");

    bool threw = false;
    try {
        static_cast<void>(tensor.at(3, 0, 0));
    } catch (const std::out_of_range&) {
        threw = true;
    }
    expect(threw, "Tensor should reject an invalid channel index.");
}

void testImagePreprocessing() {
    cppcnn::RawImage image;
    image.width = 2;
    image.height = 2;
    image.channels = 3;
    image.pixels = {
        255, 0, 0, 0, 255, 0,
        0, 0, 255, 255, 255, 255,
    };

    const auto resized = cppcnn::ImageProcessor::resize(image, 4, 4);
    expect(resized.width == 4 && resized.height == 4, "Image resize dimensions are incorrect.");

    const auto tensor = cppcnn::ImageProcessor::normalize(resized);
    expect(tensor.channels() == 3, "Normalized image channel count is incorrect.");
    expect(tensor.height() == 4 && tensor.width() == 4, "Normalized image dimensions are incorrect.");
    for (const float value : tensor.values()) {
        expect(value >= 0.0F && value <= 1.0F, "Normalized pixel is outside [0, 1].");
    }
    expectNear(tensor.at(0, 0, 0), 1.0F, 1.0e-6F, "RGB channel ordering is incorrect.");
}

}  // namespace

int main() {
    try {
        testTensor();
        testImagePreprocessing();
        std::cout << "All basic tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
