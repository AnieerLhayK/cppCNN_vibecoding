#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cppcnn {

// ---------------------------------------------------------------------------
// Polymorphic base — exposes forward() and archive-based serialization so
// the trainer can work with either LeNet or Enhanced without knowing the
// concrete type at compile time.
// ---------------------------------------------------------------------------

class LibTorchModule : public torch::nn::Module {
public:
    ~LibTorchModule() override = default;

    /// Must be implemented by each concrete architecture.
    virtual torch::Tensor forward(torch::Tensor x) = 0;

    /// Archive-based serialization (works through virtual dispatch).
    virtual void serialize(torch::serialize::OutputArchive& archive) const {
        torch::nn::Module::save(archive);
    }
    virtual void deserialize(torch::serialize::InputArchive& archive) {
        torch::nn::Module::load(archive);
    }

    /// High-level save / load wrappers.
    void saveToFile(const std::string& path) const {
        torch::serialize::OutputArchive archive;
        serialize(archive);
        archive.save_to(path);
    }

    void loadFromFile(const std::string& path) {
        torch::serialize::InputArchive archive;
        archive.load_from(path, torch::Device(torch::kCPU));
        deserialize(archive);
    }
};

// ---------------------------------------------------------------------------
// Architecture selector
// ---------------------------------------------------------------------------

enum class LibTorchArchitecture {
    LeNet,
    Enhanced,
};

std::string libTorchArchToString(LibTorchArchitecture arch);

// ---------------------------------------------------------------------------
// LeNet: Conv(3->6,5) -> ReLU -> Pool(2) -> Conv(6->16,5) -> ReLU -> Pool(2)
//      -> FC(400->120) -> ReLU -> FC(120->N) -> LogSoftmax ~56K params
// ---------------------------------------------------------------------------

class LibTorchLeNetImpl : public LibTorchModule {
public:
    explicit LibTorchLeNetImpl(int64_t numClasses);
    torch::Tensor forward(torch::Tensor x) override;

private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::Conv2d conv2_{nullptr};
    torch::nn::Linear fc1_{nullptr};
    torch::nn::Linear fc2_{nullptr};
};
TORCH_MODULE(LibTorchLeNet);

// ---------------------------------------------------------------------------
// Enhanced: 3 conv + 3 FC + dropout ~1.3M params
// ---------------------------------------------------------------------------

class LibTorchEnhancedImpl : public LibTorchModule {
public:
    explicit LibTorchEnhancedImpl(int64_t numClasses);
    torch::Tensor forward(torch::Tensor x) override;

private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::Conv2d conv2_{nullptr};
    torch::nn::Conv2d conv3_{nullptr};
    torch::nn::Linear fc1_{nullptr};
    torch::nn::Linear fc2_{nullptr};
    torch::nn::Linear fc3_{nullptr};
    torch::nn::Dropout dropout1_{nullptr};
    torch::nn::Dropout dropout2_{nullptr};
};
TORCH_MODULE(LibTorchEnhanced);

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::shared_ptr<LibTorchModule> createLibTorchModel(
    LibTorchArchitecture arch, int64_t numClasses);

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

std::size_t countParams(torch::nn::Module& module);

}  // namespace cppcnn


