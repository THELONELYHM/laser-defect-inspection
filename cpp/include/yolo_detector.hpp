#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <string>
#include <vector>

struct Detection {
    int class_id{};
    float confidence{};
    cv::Rect box{};
};

class YoloDetector {
public:
    YoloDetector(
        const std::string& model_path,
        std::vector<std::string> class_names,
        int input_size = 640,
        float confidence_threshold = 0.25F,
        float iou_threshold = 0.45F,
        bool use_cuda = false);

    [[nodiscard]] std::vector<Detection> detect(const cv::Mat& image);
    [[nodiscard]] const std::vector<std::string>& classNames() const noexcept;

private:
    struct LetterboxInfo {
        float scale{};
        int pad_x{};
        int pad_y{};
    };

    cv::Mat letterbox(const cv::Mat& image, LetterboxInfo& info) const;
    cv::Rect restoreBox(
        float center_x,
        float center_y,
        float width,
        float height,
        const LetterboxInfo& info,
        const cv::Size& original_size) const;

    cv::dnn::Net net_;
    std::vector<std::string> class_names_;
    int input_size_{};
    float confidence_threshold_{};
    float iou_threshold_{};
};

