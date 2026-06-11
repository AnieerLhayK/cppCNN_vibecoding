#include "cnn/CNN.h"

#include "cnn/Activation.h"
#include "cnn/ConvLayer.h"
#include "cnn/FCLayer.h"
#include "cnn/FlattenLayer.h"
#include "cnn/Loss.h"
#include "cnn/PoolingLayer.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace cppcnn {

namespace {

constexpr std::array<char, 8> modelMagic = {'C', 'P', 'P', 'C', 'N', 'N', '1', '\0'};
constexpr std::uint32_t modelVersion = 1;
constexpr std::uint32_t convolutionType = 1;
constexpr std::uint32_t fullyConnectedType = 2;

template <typename Value>
void writeValue(std::ostream& output, const Value& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(Value));
    if (!output) {
        throw std::runtime_error("Failed while writing model data.");
    }
}

template <typename Value>
Value readValue(std::istream& input) {
    Value value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(Value));
    if (!input) {
        throw std::runtime_error("Model file is truncated or invalid.");
    }
    return value;
}

void writeVector(std::ostream& output, const std::vector<float>& values) {
    const auto count = static_cast<std::uint64_t>(values.size());
    writeValue(output, count);
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("Failed while writing model parameters.");
    }
}

void readVector(std::istream& input, std::vector<float>& values) {
    const auto count = readValue<std::uint64_t>(input);
    if (count != values.size()) {
        throw std::runtime_error("Model parameter count does not match this network.");
    }
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!input) {
        throw std::runtime_error("Model parameter data is truncated.");
    }
}

}  // namespace

CNN::CNN(const std::size_t classCount, const std::uint32_t seed)
    : classCount_(classCount) {
    if (classCount == 0) {
        throw std::invalid_argument("CNN class count must be positive.");
    }

    // LeNet-style flow for 3 x 32 x 32 input:
    // 6 x 28 x 28 -> 6 x 14 x 14 -> 16 x 10 x 10 -> 16 x 5 x 5.
    layers_.push_back(std::make_unique<ConvLayer>(3, 6, 5, 1, 0, seed));
    layers_.push_back(std::make_unique<ReLULayer>());
    layers_.push_back(std::make_unique<MaxPoolingLayer>(2, 2));
    layers_.push_back(std::make_unique<ConvLayer>(6, 16, 5, 1, 0, seed + 1));
    layers_.push_back(std::make_unique<ReLULayer>());
    layers_.push_back(std::make_unique<MaxPoolingLayer>(2, 2));
    layers_.push_back(std::make_unique<FlattenLayer>());
    layers_.push_back(std::make_unique<FCLayer>(16 * 5 * 5, 120, seed + 2));
    layers_.push_back(std::make_unique<ReLULayer>());
    layers_.push_back(std::make_unique<FCLayer>(120, classCount_, seed + 3));
    layers_.push_back(std::make_unique<SoftmaxLayer>());
}

Tensor CNN::forward(const Tensor& input) {
    if (input.channels() != 3 || input.height() != 32 || input.width() != 32) {
        throw std::invalid_argument("CNN expects a normalized 3 x 32 x 32 input tensor.");
    }

    Tensor output = input;
    for (const auto& layer : layers_) {
        output = layer->forward(output);
    }
    return output;
}

Tensor CNN::backward(const Tensor& outputGradient) {
    Tensor gradient = outputGradient;
    for (auto iterator = layers_.rbegin(); iterator != layers_.rend(); ++iterator) {
        gradient = (*iterator)->backward(gradient);
    }
    return gradient;
}

void CNN::zeroGrad() {
    for (const auto& layer : layers_) {
        layer->zeroGrad();
    }
}

void CNN::update(
    const float learningRate,
    const float gradientScale,
    const float weightDecay) {
    for (const auto& layer : layers_) {
        layer->update(learningRate, gradientScale, weightDecay);
    }
}

std::size_t CNN::predict(const Tensor& input, float* confidence) {
    const Tensor probabilities = forward(input);
    const std::size_t prediction = argmax(probabilities);
    if (confidence != nullptr) {
        *confidence = probabilities[prediction];
    }
    return prediction;
}

void CNN::saveModel(const std::filesystem::path& path) const {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Could not create model file: " + path.string());
    }

    output.write(modelMagic.data(), static_cast<std::streamsize>(modelMagic.size()));
    writeValue(output, modelVersion);
    writeValue(output, static_cast<std::uint64_t>(classCount_));

    std::uint32_t trainableLayerCount = 0;
    for (const auto& layer : layers_) {
        if (dynamic_cast<const ConvLayer*>(layer.get()) != nullptr
            || dynamic_cast<const FCLayer*>(layer.get()) != nullptr) {
            ++trainableLayerCount;
        }
    }
    writeValue(output, trainableLayerCount);

    for (const auto& layer : layers_) {
        if (const auto* convolution = dynamic_cast<const ConvLayer*>(layer.get())) {
            writeValue(output, convolutionType);
            writeVector(output, convolution->weights());
            writeVector(output, convolution->biases());
        } else if (const auto* fullyConnected = dynamic_cast<const FCLayer*>(layer.get())) {
            writeValue(output, fullyConnectedType);
            writeVector(output, fullyConnected->weights());
            writeVector(output, fullyConnected->biases());
        }
    }
}

void CNN::loadModel(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open model file: " + path.string());
    }

    std::array<char, modelMagic.size()> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input || magic != modelMagic) {
        throw std::runtime_error("File is not a supported cppCNN model: " + path.string());
    }
    if (readValue<std::uint32_t>(input) != modelVersion) {
        throw std::runtime_error("Model version is not supported.");
    }
    if (readValue<std::uint64_t>(input) != classCount_) {
        throw std::runtime_error("Model class count does not match the current network.");
    }

    std::vector<Layer*> trainableLayers;
    for (const auto& layer : layers_) {
        if (dynamic_cast<ConvLayer*>(layer.get()) != nullptr
            || dynamic_cast<FCLayer*>(layer.get()) != nullptr) {
            trainableLayers.push_back(layer.get());
        }
    }

    const auto fileLayerCount = readValue<std::uint32_t>(input);
    if (fileLayerCount != trainableLayers.size()) {
        throw std::runtime_error("Model architecture does not match the current network.");
    }

    for (Layer* layer : trainableLayers) {
        const auto type = readValue<std::uint32_t>(input);
        if (auto* convolution = dynamic_cast<ConvLayer*>(layer)) {
            if (type != convolutionType) {
                throw std::runtime_error("Model layer order does not match the current network.");
            }
            readVector(input, convolution->mutableWeights());
            readVector(input, convolution->mutableBiases());
        } else if (auto* fullyConnected = dynamic_cast<FCLayer*>(layer)) {
            if (type != fullyConnectedType) {
                throw std::runtime_error("Model layer order does not match the current network.");
            }
            readVector(input, fullyConnected->mutableWeights());
            readVector(input, fullyConnected->mutableBiases());
        }
    }
    zeroGrad();
}

std::size_t CNN::classCount() const noexcept {
    return classCount_;
}

std::size_t CNN::layerCount() const noexcept {
    return layers_.size();
}

}  // namespace cppcnn
