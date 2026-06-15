#pragma once

#include <cstddef>

namespace cppcnn {

class LRScheduler {
public:
    enum class Type { None, Step, MultiStep, Cosine };

    struct Config {
        Type type = Type::None;
        float initialLR = 0.01F;
        float decayFactor = 0.1F;
        std::size_t stepSize = 30;       // for StepLR
        std::size_t warmupEpochs = 0;
        float minLR = 1.0e-6F;
    };

    explicit LRScheduler(Config config);

    float getLR(std::size_t epoch) const;
    void onEpochEnd();

    [[nodiscard]] const Config& config() const noexcept;

private:
    Config config_;
    std::size_t currentEpoch_ = 0;
};

}  // namespace cppcnn
