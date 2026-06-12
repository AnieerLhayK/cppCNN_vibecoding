#pragma once

#include <QString>

struct PredictionResult {
    int classId = -1;
    QString label;
    double confidence = 0.0;
};
