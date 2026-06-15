#include "LibTorchModel.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace cppcnn {

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

std::string libTorchArchToString(LibTorchArchitecture arch) {
    switch (arch) {
        case LibTorchArchitecture::LeNet:    return "LeNet";
        case LibTorchArchitecture::Enhanced: return "Enhanced";
    }
    return "Unknown";
}

// ---------------------------------------------------------------------------
// LeNet
// ---------------------------------------------------------------------------

LibTorchLeNetImpl::LibTorchLeNetImpl(int64_t numClasses)
    : conv1_(register_module("conv1", torch::nn::Conv2d(3, 6, 5)))
    , conv2_(register_module("conv2", torch::nn::Conv2d(6, 16, 5)))
    , fc1_(register_module("fc1", torch::nn::Linear(400, 120)))
    , fc2_(register_module("fc2", torch::nn::Linear(120, numClasses)))
{}

torch::Tensor LibTorchLeNetImpl::forward(torch::Tensor x) {
    // Input: (B, 3, 32, 32)
    x = torch::relu(conv1_->forward(x));          // (B, 6, 28, 28)
    x = torch::max_pool2d(x, 2);                  // (B, 6, 14, 14)
    x = torch::relu(conv2_->forward(x));          // (B, 16, 10, 10)
    x = torch::max_pool2d(x, 2);                  // (B, 16, 5, 5)
    x = x.view({-1, 400});                        // (B, 400)
    x = torch::relu(fc1_->forward(x));            // (B, 120)
    x = fc2_->forward(x);                         // (B, numClasses)
    return torch::log_softmax(x, /*dim=*/1);      // (B, numClasses)
}

// ---------------------------------------------------------------------------
// Enhanced
// ---------------------------------------------------------------------------

LibTorchEnhancedImpl::LibTorchEnhancedImpl(int64_t numClasses)
    : conv1_(register_module("conv1", torch::nn::Conv2d(
          torch::nn::Conv2dOptions(3, 32, 5).padding(2))))
    , conv2_(register_module("conv2", torch::nn::Conv2d(
          torch::nn::Conv2dOptions(32, 64, 3).padding(1))))
    , conv3_(register_module("conv3", torch::nn::Conv2d(
          torch::nn::Conv2dOptions(64, 128, 3).padding(1))))
    , fc1_(register_module("fc1", torch::nn::Linear(2048, 512)))
    , fc2_(register_module("fc2", torch::nn::Linear(512, 256)))
    , fc3_(register_module("fc3", torch::nn::Linear(256, numClasses)))
    , dropout1_(register_module("dropout1", torch::nn::Dropout(0.5)))
    , dropout2_(register_module("dropout2", torch::nn::Dropout(0.5)))
{}

torch::Tensor LibTorchEnhancedImpl::forward(torch::Tensor x) {
    // Input: (B, 3, 32, 32)
    x = torch::relu(conv1_->forward(x));          // (B, 32, 32, 32)
    x = torch::max_pool2d(x, 2);                  // (B, 32, 16, 16)
    x = torch::relu(conv2_->forward(x));          // (B, 64, 16, 16)
    x = torch::max_pool2d(x, 2);                  // (B, 64, 8, 8)
    x = torch::relu(conv3_->forward(x));          // (B, 128, 8, 8)
    x = torch::max_pool2d(x, 2);                  // (B, 128, 4, 4)
    x = x.view({-1, 2048});                       // (B, 2048)
    x = torch::relu(fc1_->forward(x));            // (B, 512)
    x = dropout1_->forward(x);                    // (B, 512)
    x = torch::relu(fc2_->forward(x));            // (B, 256)
    x = dropout2_->forward(x);                    // (B, 256)
    x = fc3_->forward(x);                         // (B, numClasses)
    return torch::log_softmax(x, /*dim=*/1);      // (B, numClasses)
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::shared_ptr<LibTorchModule> createLibTorchModel(
    LibTorchArchitecture arch, int64_t numClasses)
{
    switch (arch) {
        case LibTorchArchitecture::LeNet:
            return std::make_shared<LibTorchLeNetImpl>(numClasses);
        case LibTorchArchitecture::Enhanced:
            return std::make_shared<LibTorchEnhancedImpl>(numClasses);
    }
    return std::make_shared<LibTorchLeNetImpl>(numClasses);
}

// ---------------------------------------------------------------------------
// Parameter counting
// ---------------------------------------------------------------------------

std::size_t countParams(torch::nn::Module& module) {
    std::size_t total = 0;
    for (const auto& p : module.parameters()) {
        total += static_cast<std::size_t>(p.numel());
    }
    return total;
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------


}  // namespace cppcnn

