#pragma once

#include "cnn/CNN.h"
#include "cnn/Trainer.h"

#include <filesystem>
#include <string>
#include <vector>

namespace cppcnn {

class App {
public:
    int run(int argc, char* argv[]);

private:
    int runTrain(const std::vector<std::string>& arguments);
    int runEvaluate(const std::vector<std::string>& arguments);
    int runPredict(const std::vector<std::string>& arguments);
    int runInteractive(const std::vector<std::string>& arguments);
    int runTrainAdvanced(const std::vector<std::string>& arguments);

    static void printUsage();
    static void printResourceStatus();
    static void requireDataset(
        const std::filesystem::path& path,
        const std::string& purpose);
    static void requireModel(const std::filesystem::path& path);
    static std::size_t parseSize(
        const std::string& value,
        const std::string& argumentName,
        bool allowZero = false);
    static float parseFloat(
        const std::string& value,
        const std::string& argumentName,
        bool allowZero = false);
    static std::vector<std::string> loadLabels(
        const std::filesystem::path& path,
        std::size_t classCount);
};

}  // namespace cppcnn
