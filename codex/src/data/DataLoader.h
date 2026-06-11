#pragma once

#include "cnn/Tensor.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace cppcnn {

struct DataSample {
    std::filesystem::path imagePath;
    std::size_t label = 0;
    int originalClassId = 0;
};

struct Dataset {
    std::vector<DataSample> samples;
    std::vector<int> classIds;

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t classCount() const noexcept;
};

struct DataLoaderOptions {
    std::size_t classLimit = 10;
    std::size_t samplesPerClass = 0;
    int imageWidth = 32;
    int imageHeight = 32;
    bool shuffle = false;
    std::uint32_t seed = 42;
};

class DataLoader {
public:
    explicit DataLoader(DataLoaderOptions options = {});

    Dataset loadDirectory(const std::filesystem::path& splitDirectory) const;
    Tensor loadTensor(const DataSample& sample) const;

    [[nodiscard]] const DataLoaderOptions& options() const noexcept;

private:
    [[nodiscard]] static bool isSupportedImage(const std::filesystem::path& path);
    [[nodiscard]] static int parseClassId(const std::filesystem::path& directory);

    DataLoaderOptions options_;
};

}  // namespace cppcnn
