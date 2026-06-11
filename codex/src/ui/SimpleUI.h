#pragma once

#include <filesystem>
#include <string>

namespace cppcnn {

class SimpleUI {
public:
    static void showPrediction(
        const std::filesystem::path& imagePath,
        const std::string& label,
        float confidence);
};

}  // namespace cppcnn
