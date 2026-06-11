#include "data/DataLoader.h"

#include "image/ImageProcessor.h"

#include <algorithm>
#include <charconv>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace cppcnn {

bool Dataset::empty() const noexcept {
    return samples.empty();
}

std::size_t Dataset::size() const noexcept {
    return samples.size();
}

std::size_t Dataset::classCount() const noexcept {
    return classIds.size();
}

DataLoader::DataLoader(DataLoaderOptions options)
    : options_(std::move(options)) {
    if (options_.classLimit == 0) {
        throw std::invalid_argument("DataLoader class limit must be positive.");
    }
    if (options_.imageWidth <= 0 || options_.imageHeight <= 0) {
        throw std::invalid_argument("DataLoader image dimensions must be positive.");
    }
}

Dataset DataLoader::loadDirectory(
    const std::filesystem::path& splitDirectory) const {
    if (!std::filesystem::exists(splitDirectory)) {
        throw std::runtime_error(
            "Dataset directory does not exist: " + splitDirectory.string());
    }
    if (!std::filesystem::is_directory(splitDirectory)) {
        throw std::runtime_error(
            "Dataset path is not a directory: " + splitDirectory.string());
    }

    std::vector<std::pair<int, std::filesystem::path>> classDirectories;
    for (const auto& entry : std::filesystem::directory_iterator(splitDirectory)) {
        if (!entry.is_directory()) {
            continue;
        }
        try {
            classDirectories.emplace_back(parseClassId(entry.path()), entry.path());
        } catch (const std::invalid_argument&) {
            // Files such as README directories are ignored; at least one numeric class is required.
        }
    }

    std::sort(classDirectories.begin(), classDirectories.end(), [](const auto& left, const auto& right) {
        return left.first < right.first;
    });
    if (classDirectories.empty()) {
        throw std::runtime_error(
            "No numeric class directories were found under: " + splitDirectory.string());
    }
    if (classDirectories.size() > options_.classLimit) {
        classDirectories.resize(options_.classLimit);
    }

    Dataset dataset;
    dataset.classIds.reserve(classDirectories.size());

    for (std::size_t label = 0; label < classDirectories.size(); ++label) {
        const int classId = classDirectories[label].first;
        const auto& classDirectory = classDirectories[label].second;
        dataset.classIds.push_back(classId);

        std::vector<std::filesystem::path> imagePaths;
        for (const auto& entry : std::filesystem::directory_iterator(classDirectory)) {
            if (entry.is_regular_file() && isSupportedImage(entry.path())) {
                imagePaths.push_back(entry.path());
            }
        }
        std::sort(imagePaths.begin(), imagePaths.end());
        if (options_.samplesPerClass > 0 && imagePaths.size() > options_.samplesPerClass) {
            imagePaths.resize(options_.samplesPerClass);
        }
        for (const auto& imagePath : imagePaths) {
            dataset.samples.push_back({imagePath, label, classId});
        }
    }

    if (dataset.samples.empty()) {
        throw std::runtime_error(
            "Class directories contain no supported images: " + splitDirectory.string());
    }
    if (options_.shuffle) {
        std::mt19937 generator(options_.seed);
        std::shuffle(dataset.samples.begin(), dataset.samples.end(), generator);
    }
    return dataset;
}

Tensor DataLoader::loadTensor(const DataSample& sample) const {
    return ImageProcessor::loadAndPreprocess(
        sample.imagePath,
        options_.imageWidth,
        options_.imageHeight);
}

const DataLoaderOptions& DataLoader::options() const noexcept {
    return options_;
}

bool DataLoader::isSupportedImage(const std::filesystem::path& path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](const unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return extension == ".ppm"
        || extension == ".pnm"
        || extension == ".png"
        || extension == ".jpg"
        || extension == ".jpeg"
        || extension == ".bmp";
}

int DataLoader::parseClassId(const std::filesystem::path& directory) {
    const std::string name = directory.filename().string();
    int classId = 0;
    const auto result = std::from_chars(name.data(), name.data() + name.size(), classId);
    if (result.ec != std::errc() || result.ptr != name.data() + name.size() || classId < 0) {
        throw std::invalid_argument("Class directory name is not a non-negative integer.");
    }
    return classId;
}

}  // namespace cppcnn
