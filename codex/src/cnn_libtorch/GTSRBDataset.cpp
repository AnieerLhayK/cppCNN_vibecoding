#include "GTSRBDataset.h"

#include "data/DataLoader.h"
#include "image/ImageProcessor.h"

#include <torch/torch.h>

#include <cstddef>
#include <filesystem>
#include <vector>

namespace cppcnn {

// ---------------------------------------------------------------------------
// GTSRBDataset
// ---------------------------------------------------------------------------

GTSRBDataset::GTSRBDataset(std::vector<SamplePair> samples,
                           int imageWidth, int imageHeight)
    : samples_(std::move(samples))
    , imageWidth_(imageWidth)
    , imageHeight_(imageHeight)
{}

torch::data::Example<> GTSRBDataset::get(size_t index) {
    const auto& [path, label] = samples_[index];
    torch::Tensor image = loadImage(path);
    torch::Tensor target = torch::tensor(label, torch::kInt64);
    return {image, target};
}

torch::optional<size_t> GTSRBDataset::size() const {
    return samples_.size();
}

const std::vector<SamplePair>& GTSRBDataset::samples() const noexcept {
    return samples_;
}

torch::Tensor GTSRBDataset::loadImage(
    const std::filesystem::path& path) const
{
    // Reuse the existing well-tested image loading pipeline.
    // This handles PPM, PNG, JPG, BMP (with OpenCV) and returns a
    // normalized [0,1] float32 Tensor in CHW layout (3 x H x W).
    Tensor cpuTensor = ImageProcessor::loadAndPreprocess(
        path, imageWidth_, imageHeight_);

    // Convert the CPU Tensor's float buffer to a torch::Tensor.
    // The CPU Tensor uses CHW layout internally (channel, height, width)
    // which matches LibTorch's expected format for unbatched images.
    const auto& data = cpuTensor.values();

    // from_blob is non-owning, so clone() to get an owning tensor.
    auto result = torch::from_blob(
        const_cast<float*>(data.data()),
        {static_cast<int64_t>(cpuTensor.channels()),
         static_cast<int64_t>(cpuTensor.height()),
         static_cast<int64_t>(cpuTensor.width())},
        torch::kFloat32
    ).clone();

    return result;
}

// ---------------------------------------------------------------------------
// extractSamplePairs
// ---------------------------------------------------------------------------

std::vector<SamplePair> extractSamplePairs(
    const std::vector<DataSample>& samples)
{
    std::vector<SamplePair> result;
    result.reserve(samples.size());
    for (const auto& s : samples) {
        result.emplace_back(s.imagePath, static_cast<int64_t>(s.label));
    }
    return result;
}

}  // namespace cppcnn
