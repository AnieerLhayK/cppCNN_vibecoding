#pragma once

#include <torch/torch.h>

#include <filesystem>
#include <string>
#include <vector>

namespace cppcnn {

/// DataSample pair: (image path, label).
using SamplePair = std::pair<std::filesystem::path, int64_t>;

/// A torch::data::Dataset that loads GTSRB images on the fly.
/// Each call to get() reads the image from disk, normalizes it,
/// and returns a {torch::Tensor, label} pair.
class GTSRBDataset : public torch::data::Dataset<GTSRBDataset> {
public:
    explicit GTSRBDataset(std::vector<SamplePair> samples,
                          int imageWidth = 32,
                          int imageHeight = 32);

    /// Return the (image, label) pair at the given index.
    torch::data::Example<> get(size_t index) override;

    /// Total number of samples.
    torch::optional<size_t> size() const override;

    /// Access the raw samples (for inspection).
    const std::vector<SamplePair>& samples() const noexcept;

private:
    std::vector<SamplePair> samples_;
    int imageWidth_;
    int imageHeight_;

    /// Load a single image file and return a normalized CHW float tensor.
    torch::Tensor loadImage(const std::filesystem::path& path) const;
};

// ---------------------------------------------------------------------------
// Helper: convert a DataSample list (from the original DataLoader) into
// SamplePairs for GTSRBDataset.
// ---------------------------------------------------------------------------

struct DataSample;  // forward decl from DataLoader.h

/// Extract (path, label) pairs from the original Dataset's sample list.
std::vector<SamplePair> extractSamplePairs(
    const std::vector<DataSample>& samples);

}  // namespace cppcnn
