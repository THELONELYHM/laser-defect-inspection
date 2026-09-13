#include "yolo_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

YoloDetector::YoloDetector(
    const std::string& model_path,
    std::vector<std::string> class_names,
    int input_size,
    float confidence_threshold,
    float iou_threshold,
    bool use_cuda)
    : net_(cv::dnn::readNetFromONNX(model_path)),
      class_names_(std::move(class_names)),
      input_size_(input_size),
      confidence_threshold_(confidence_threshold),
      iou_threshold_(iou_threshold) {
    if (net_.empty()) {
        throw std::runtime_error("Failed to load ONNX model: " + model_path);
    }
    if (class_names_.empty()) {
        throw std::invalid_argument("At least one class name is required");
    }
    if (use_cuda) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    } else {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

const std::vector<std::string>& YoloDetector::classNames() const noexcept {
    return class_names_;
}

cv::Mat YoloDetector::letterbox(const cv::Mat& image, LetterboxInfo& info) const {
    const float scale = std::min(
        static_cast<float>(input_size_) / static_cast<float>(image.cols),
        static_cast<float>(input_size_) / static_cast<float>(image.rows));
    const int resized_width = static_cast<int>(std::round(image.cols * scale));
    const int resized_height = static_cast<int>(std::round(image.rows * scale));
    const int pad_width = input_size_ - resized_width;
    const int pad_height = input_size_ - resized_height;
    const int left = pad_width / 2;
    const int right = pad_width - left;
    const int top = pad_height / 2;
    const int bottom = pad_height - top;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);
    cv::Mat padded;
    cv::copyMakeBorder(
        resized,
        padded,
        top,
        bottom,
        left,
        right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114));
    info = {scale, left, top};
    return padded;
}

cv::Rect YoloDetector::restoreBox(
    float center_x,
    float center_y,
    float width,
    float height,
    const LetterboxInfo& info,
    const cv::Size& original_size) const {
    const float x1 = (center_x - width / 2.0F - info.pad_x) / info.scale;
    const float y1 = (center_y - height / 2.0F - info.pad_y) / info.scale;
    const float x2 = (center_x + width / 2.0F - info.pad_x) / info.scale;
    const float y2 = (center_y + height / 2.0F - info.pad_y) / info.scale;
    const int left = std::clamp(static_cast<int>(std::round(x1)), 0, original_size.width - 1);
    const int top = std::clamp(static_cast<int>(std::round(y1)), 0, original_size.height - 1);
    const int right = std::clamp(static_cast<int>(std::round(x2)), left + 1, original_size.width);
    const int bottom = std::clamp(static_cast<int>(std::round(y2)), top + 1, original_size.height);
    return {left, top, right - left, bottom - top};
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty");
    }
    LetterboxInfo letterbox_info;
    cv::Mat input = letterbox(image, letterbox_info);
    cv::Mat blob = cv::dnn::blobFromImage(
        input,
        1.0 / 255.0,
        cv::Size(input_size_, input_size_),
        cv::Scalar(),
        true,
        false,
        CV_32F);
    net_.setInput(blob);

    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    if (outputs.empty()) {
        throw std::runtime_error("The network returned no output tensors");
    }

    const cv::Mat& raw = outputs.front();
    cv::Mat predictions;
    if (raw.dims == 3) {
        const int dimension_a = raw.size[1];
        const int dimension_b = raw.size[2];
        cv::Mat view(dimension_a, dimension_b, CV_32F, const_cast<float*>(raw.ptr<float>()));
        if (dimension_a < dimension_b) {
            cv::transpose(view, predictions); // [1, channels, candidates] -> [candidates, channels]
        } else {
            predictions = view; // already [1, candidates, channels]
        }
    } else if (raw.dims == 2) {
        predictions = raw;
    } else {
        throw std::runtime_error("Unsupported ONNX output rank; expected 2 or 3 dimensions");
    }

    const int expected_columns = 4 + static_cast<int>(class_names_.size());
    if (predictions.cols < expected_columns) {
        throw std::runtime_error(
            "Unexpected ONNX output shape. Export an Ultralytics detection model without NMS.");
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    boxes.reserve(predictions.rows);
    scores.reserve(predictions.rows);
    class_ids.reserve(predictions.rows);

    for (int row_index = 0; row_index < predictions.rows; ++row_index) {
        const float* row = predictions.ptr<float>(row_index);
        const cv::Mat class_scores(1, static_cast<int>(class_names_.size()), CV_32F,
                                   const_cast<float*>(row + 4));
        cv::Point best_class;
        double best_score = 0.0;
        cv::minMaxLoc(class_scores, nullptr, &best_score, nullptr, &best_class);
        if (best_score < confidence_threshold_) {
            continue;
        }
        boxes.push_back(restoreBox(row[0], row[1], row[2], row[3], letterbox_info, image.size()));
        scores.push_back(static_cast<float>(best_score));
        class_ids.push_back(best_class.x);
    }

    // Run NMS per class so overlapping boxes from different defect classes are preserved.
    std::vector<Detection> detections;
    for (int class_id = 0; class_id < static_cast<int>(class_names_.size()); ++class_id) {
        std::vector<cv::Rect> class_boxes;
        std::vector<float> class_scores;
        std::vector<int> original_indices;
        for (std::size_t index = 0; index < boxes.size(); ++index) {
            if (class_ids[index] == class_id) {
                class_boxes.push_back(boxes[index]);
                class_scores.push_back(scores[index]);
                original_indices.push_back(static_cast<int>(index));
            }
        }
        std::vector<int> kept;
        cv::dnn::NMSBoxes(
            class_boxes,
            class_scores,
            confidence_threshold_,
            iou_threshold_,
            kept);
        for (int local_index : kept) {
            const int original_index = original_indices[local_index];
            detections.push_back({class_id, scores[original_index], boxes[original_index]});
        }
    }
    std::sort(
        detections.begin(),
        detections.end(),
        [](const Detection& left, const Detection& right) {
            return left.confidence > right.confidence;
        });
    return detections;
}

