#include "data/DataLoader.h"

#include "image/ImageProcessor.h"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
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

Dataset DataLoader::loadDataset(
    const std::filesystem::path& datasetPath,
    const DatasetSplit split) const {
    if (!std::filesystem::exists(datasetPath)) {
        throw std::runtime_error(
            "Dataset path does not exist: " + datasetPath.string()
            + ". Download GTSRB or generate GTSRB_subset; see docs/dataset_guide.md.");
    }

    try {
        return loadDirectory(resolveClassDirectory(datasetPath, split));
    } catch (const std::runtime_error&) {
        if (split == DatasetSplit::Test) {
            return loadOfficialTestSet(datasetPath);
        }
        throw;
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
            // Ignore non-numeric directories
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
            DataSample sample;
            sample.imagePath = imagePath;
            sample.label = label;
            sample.originalClassId = classId;
            sample.trackId = parseTrackId(imagePath);
            dataset.samples.push_back(std::move(sample));
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

Dataset DataLoader::loadOfficialTestSet(
    const std::filesystem::path& datasetPath) const {
    const auto imagesDirectory = findExistingPath({
        datasetPath / "Final_Test" / "Images",
        datasetPath / "GTSRB" / "Final_Test" / "Images",
        datasetPath,
    });
    const auto csvPath = findExistingPath({
        datasetPath / "GT-final_test.csv",
        datasetPath / "GTSRB" / "GT-final_test.csv",
        datasetPath.parent_path() / "GT-final_test.csv",
        datasetPath.parent_path().parent_path() / "GT-final_test.csv",
    });
    if (imagesDirectory.empty() || csvPath.empty()) {
        throw std::runtime_error(
            "Could not find the official GTSRB test images and GT-final_test.csv under: "
            + datasetPath.string());
    }

    std::ifstream input(csvPath);
    if (!input) {
        throw std::runtime_error("Could not open GTSRB test labels: " + csvPath.string());
    }

    std::string line;
    std::getline(input, line);
    std::vector<std::pair<std::filesystem::path, int>> rows;
    std::set<int> availableClassIds;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> fields;
        std::stringstream rowStream(line);
        std::string field;
        while (std::getline(rowStream, field, ';')) {
            fields.push_back(field);
        }
        if (fields.size() < 8) {
            throw std::runtime_error("Invalid GTSRB test CSV row: " + line);
        }
        int classId = 0;
        const auto result = std::from_chars(
            fields[7].data(),
            fields[7].data() + fields[7].size(),
            classId);
        if (result.ec != std::errc() || result.ptr != fields[7].data() + fields[7].size()) {
            throw std::runtime_error("Invalid class ID in GTSRB test CSV: " + fields[7]);
        }
        rows.emplace_back(imagesDirectory / fields[0], classId);
        availableClassIds.insert(classId);
    }

    std::vector<int> selectedClassIds(
        availableClassIds.begin(),
        availableClassIds.end());
    if (selectedClassIds.size() > options_.classLimit) {
        selectedClassIds.resize(options_.classLimit);
    }
    std::map<int, std::size_t> labelByClass;
    for (std::size_t label = 0; label < selectedClassIds.size(); ++label) {
        labelByClass[selectedClassIds[label]] = label;
    }

    Dataset dataset;
    dataset.classIds = selectedClassIds;
    std::map<int, std::size_t> classCounts;
    for (const auto& [imagePath, classId] : rows) {
        const auto label = labelByClass.find(classId);
        if (label == labelByClass.end()) {
            continue;
        }
        if (options_.samplesPerClass > 0
            && classCounts[classId] >= options_.samplesPerClass) {
            continue;
        }
        if (!std::filesystem::exists(imagePath)) {
            throw std::runtime_error(
                "GTSRB test image listed by CSV is missing: " + imagePath.string());
        }
        DataSample sample;
        sample.imagePath = imagePath;
        sample.label = label->second;
        sample.originalClassId = classId;
        sample.trackId = parseTrackId(imagePath);
        dataset.samples.push_back(std::move(sample));
        ++classCounts[classId];
    }
    if (dataset.samples.empty()) {
        throw std::runtime_error("GTSRB test CSV did not produce any selected samples.");
    }
    if (options_.shuffle) {
        std::mt19937 generator(options_.seed);
        std::shuffle(dataset.samples.begin(), dataset.samples.end(), generator);
    }
    return dataset;
}

TrainValSplit DataLoader::splitByTrack(
    const Dataset& fullDataset,
    const float validationRatio,
    const std::uint32_t seed) {
    if (fullDataset.empty()) {
        throw std::invalid_argument("Cannot split an empty dataset.");
    }
    if (validationRatio <= 0.0F || validationRatio >= 1.0F) {
        throw std::invalid_argument("Validation ratio must be in (0, 1).");
    }

    // Group samples by (classId, trackId)
    std::map<std::pair<int, int>, std::vector<std::size_t>> trackGroups;
    for (std::size_t index = 0; index < fullDataset.samples.size(); ++index) {
        const auto& sample = fullDataset.samples[index];
        const int trackId = sample.trackId >= 0
            ? sample.trackId
            : -static_cast<int>(index) - 1;
        trackGroups[{sample.originalClassId, trackId}].push_back(index);
    }

    TrainValSplit result;
    result.train.classIds = fullDataset.classIds;
    result.validation.classIds = fullDataset.classIds;

    // Split tracks independently per class so minority classes remain represented.
    std::map<int, std::vector<std::pair<int, int>>> tracksByClass;
    for (const auto& [key, _] : trackGroups) {
        tracksByClass[key.first].push_back(key);
    }

    std::mt19937 generator(seed);
    for (auto& [classId, trackKeys] : tracksByClass) {
        static_cast<void>(classId);
        std::shuffle(trackKeys.begin(), trackKeys.end(), generator);

        std::size_t classSampleCount = 0;
        for (const auto& key : trackKeys) {
            classSampleCount += trackGroups.at(key).size();
        }
        const std::size_t valTarget = std::max<std::size_t>(
            1, static_cast<std::size_t>(classSampleCount * validationRatio));
        std::size_t valCount = 0;

        for (std::size_t keyIndex = 0; keyIndex < trackKeys.size(); ++keyIndex) {
            const auto& indices = trackGroups.at(trackKeys[keyIndex]);
            const bool leaveTrackForTraining = keyIndex + 1 < trackKeys.size();
            const bool assignToVal = valCount < valTarget && leaveTrackForTraining;
            for (const auto index : indices) {
                if (assignToVal) {
                    result.validation.samples.push_back(fullDataset.samples[index]);
                } else {
                    result.train.samples.push_back(fullDataset.samples[index]);
                }
            }
            if (assignToVal) {
                valCount += indices.size();
            }
        }
    }

    // Ensure no empty split
    if (result.train.samples.empty() || result.validation.samples.empty()) {
        throw std::runtime_error("Track-based split produced an empty set. Adjust ratio.");
    }

    return result;
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

std::filesystem::path DataLoader::resolveClassDirectory(
    const std::filesystem::path& datasetPath,
    const DatasetSplit split) {
    std::vector<std::filesystem::path> candidates = {datasetPath};
    if (split == DatasetSplit::Training) {
        candidates.push_back(datasetPath / "train");
        candidates.push_back(datasetPath / "Final_Training" / "Images");
        candidates.push_back(datasetPath / "GTSRB" / "Final_Training" / "Images");
    } else {
        candidates.push_back(datasetPath / "test");
        candidates.push_back(datasetPath / "Final_Test" / "Images");
        candidates.push_back(datasetPath / "GTSRB" / "Final_Test" / "Images");
    }
    for (const auto& candidate : candidates) {
        if (containsNumericClassDirectories(candidate)) {
            return candidate;
        }
    }
    throw std::runtime_error(
        "No class-directory split was found under: " + datasetPath.string());
}

std::filesystem::path DataLoader::findExistingPath(
    const std::vector<std::filesystem::path>& candidates) {
    for (const auto& candidate : candidates) {
        if (!candidate.empty() && std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    return {};
}

bool DataLoader::containsNumericClassDirectories(
    const std::filesystem::path& directory) {
    if (!std::filesystem::is_directory(directory)) {
        return false;
    }
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_directory()) {
            continue;
        }
        try {
            static_cast<void>(parseClassId(entry.path()));
            return true;
        } catch (const std::invalid_argument&) {
        }
    }
    return false;
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

int DataLoader::parseTrackId(const std::filesystem::path& imagePath) {
    // GTSRB filename: XXXXX_YYYYY.ext -> XXXXX is track ID
    const std::string stem = imagePath.stem().string();
    const auto underscore = stem.find('_');
    if (underscore == std::string::npos || underscore == 0) {
        return -1;  // No track information
    }
    int trackId = 0;
    const auto result = std::from_chars(
        stem.data(), stem.data() + underscore, trackId);
    if (result.ec != std::errc()) {
        return -1;
    }
    return trackId;
}

}  // namespace cppcnn
