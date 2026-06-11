#include "cnn/Activation.h"
#include "cnn/CNN.h"
#include "cnn/ConvLayer.h"
#include "cnn/FCLayer.h"
#include "cnn/FlattenLayer.h"
#include "cnn/Loss.h"
#include "cnn/PoolingLayer.h"
#include "cnn/Tensor.h"
#include "image/ImageProcessor.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
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

void testLayerDimensions() {
    cppcnn::Tensor input(3, 32, 32, 0.5F);
    cppcnn::ConvLayer convolution(3, 6, 5, 1, 0, 7);
    cppcnn::ReLULayer relu;
    cppcnn::MaxPoolingLayer pooling(2, 2);
    cppcnn::FlattenLayer flatten;
    cppcnn::FCLayer fullyConnected(6 * 14 * 14, 10, 8);
    cppcnn::SoftmaxLayer softmax;

    const auto convolved = convolution.forward(input);
    expect(
        convolved.channels() == 6 && convolved.height() == 28 && convolved.width() == 28,
        "Convolution output dimensions are incorrect.");
    const auto activated = relu.forward(convolved);
    const auto pooled = pooling.forward(activated);
    expect(
        pooled.channels() == 6 && pooled.height() == 14 && pooled.width() == 14,
        "Pooling output dimensions are incorrect.");
    const auto flattened = flatten.forward(pooled);
    expect(flattened.size() == 6 * 14 * 14, "Flatten output size is incorrect.");
    const auto logits = fullyConnected.forward(flattened);
    const auto probabilities = softmax.forward(logits);
    expect(probabilities.size() == 10, "Softmax output size is incorrect.");
    expectNear(
        std::accumulate(probabilities.values().begin(), probabilities.values().end(), 0.0F),
        1.0F,
        1.0e-5F,
        "Softmax probabilities do not sum to one.");

    const auto lossGradient = cppcnn::CrossEntropyLoss::gradient(probabilities, 3);
    const auto logitsGradient = softmax.backward(lossGradient);
    const auto flattenedGradient = fullyConnected.backward(logitsGradient);
    const auto pooledGradient = flatten.backward(flattenedGradient);
    const auto activatedGradient = pooling.backward(pooledGradient);
    const auto convolvedGradient = relu.backward(activatedGradient);
    const auto inputGradient = convolution.backward(convolvedGradient);
    expect(
        inputGradient.channels() == 3
            && inputGradient.height() == 32
            && inputGradient.width() == 32,
        "Backward dimension flow is incorrect.");
}

void testLeNetForward() {
    cppcnn::CNN network(10, 9);
    cppcnn::Tensor input(3, 32, 32, 0.25F);
    const auto probabilities = network.forward(input);
    expect(probabilities.size() == 10, "CNN output class count is incorrect.");
    expectNear(
        std::accumulate(probabilities.values().begin(), probabilities.values().end(), 0.0F),
        1.0F,
        1.0e-5F,
        "CNN output probabilities do not sum to one.");
}

}  // namespace

int main() {
    try {
        testTensor();
        testImagePreprocessing();
        testLayerDimensions();
        testLeNetForward();
        std::cout << "All basic tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
