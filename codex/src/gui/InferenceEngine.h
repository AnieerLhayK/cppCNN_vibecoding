#pragma once

#include "cnn/CNN.h"
#include "cnn/Tensor.h"
#include "gui/PredictionResult.h"

#include <QStringList>

#include <memory>

class InferenceEngine {
public:
    [[nodiscard]] static PredictionPayload predict(
        const std::shared_ptr<cppcnn::CNN>& network,
        const cppcnn::Tensor& input,
        const QStringList& labels,
        std::size_t topK = 3);
};
