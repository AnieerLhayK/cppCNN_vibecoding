#include "gui/InferenceEngine.h"

#include <QElapsedTimer>

#include <algorithm>
#include <exception>
#include <numeric>
#include <utility>

namespace {

QString displayLabel(
    const QStringList& labels,
    const std::size_t classId) {
    if (classId < static_cast<std::size_t>(labels.size())
        && !labels.at(static_cast<qsizetype>(classId)).isEmpty()) {
        return labels.at(static_cast<qsizetype>(classId));
    }
    return QStringLiteral("Class %1").arg(classId);
}

}  // namespace

PredictionPayload InferenceEngine::predict(
    const std::shared_ptr<cppcnn::CNN>& network,
    const cppcnn::Tensor& input,
    const QStringList& labels,
    const std::size_t topK) {
    PredictionPayload payload;
    if (!network) {
        payload.error = QStringLiteral("The CNN model is not loaded.");
        return payload;
    }

    try {
        QElapsedTimer timer;
        timer.start();
        const cppcnn::Tensor probabilities = network->forward(input);
        payload.elapsedMilliseconds = timer.elapsed();

        std::vector<std::size_t> indices(probabilities.size());
        std::iota(indices.begin(), indices.end(), 0U);
        const std::size_t resultCount = std::min(topK, indices.size());
        std::partial_sort(
            indices.begin(),
            indices.begin() + static_cast<std::ptrdiff_t>(resultCount),
            indices.end(),
            [&probabilities](const std::size_t left, const std::size_t right) {
                return probabilities[left] > probabilities[right];
            });

        for (std::size_t rank = 0; rank < resultCount; ++rank) {
            const std::size_t classId = indices[rank];
            PredictionResult result;
            result.classId = static_cast<int>(classId);
            result.label = displayLabel(labels, classId);
            result.confidence = probabilities[classId];
            payload.results.push_back(std::move(result));
        }
    } catch (const std::exception& error) {
        payload.error = QString::fromUtf8(error.what());
    }
    return payload;
}
