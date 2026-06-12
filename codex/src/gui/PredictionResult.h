#pragma once

#include <QString>
#include <QVector>

struct PredictionResult {
    int classId = -1;
    QString label;
    double confidence = 0.0;
};

struct PredictionPayload {
    QVector<PredictionResult> results;
    qint64 elapsedMilliseconds = 0;
    QString error;
};
