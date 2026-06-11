#include "ui/SimpleUI.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef CPP_CNN_WITH_OPENCV
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace cppcnn {

void SimpleUI::showPrediction(
    const std::filesystem::path& imagePath,
    const std::string& label,
    const float confidence) {
#ifdef CPP_CNN_WITH_OPENCV
    const cv::Mat source = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
    if (source.empty()) {
        throw std::runtime_error(
            "OpenCV could not display image: " + imagePath.string());
    }

    cv::Mat resized;
    cv::resize(source, resized, cv::Size(384, 384), 0.0, 0.0, cv::INTER_NEAREST);
    cv::Mat canvas(434, 384, CV_8UC3, cv::Scalar(245, 245, 245));
    resized.copyTo(canvas(cv::Rect(0, 50, resized.cols, resized.rows)));

    std::ostringstream title;
    title << label << "  " << std::fixed << std::setprecision(1)
          << confidence * 100.0F << "%";
    cv::putText(
        canvas,
        title.str(),
        cv::Point(10, 32),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(20, 20, 20),
        2,
        cv::LINE_AA);
    cv::imshow("cppCNN traffic sign prediction", canvas);
    std::cout << "Press any key in the image window to continue.\n";
    cv::waitKey(0);
    cv::destroyWindow("cppCNN traffic sign prediction");
#else
    static_cast<void>(imagePath);
    static_cast<void>(label);
    static_cast<void>(confidence);
    std::cout
        << "Image window is unavailable because OpenCV was not found at build time.\n";
#endif
}

}  // namespace cppcnn
