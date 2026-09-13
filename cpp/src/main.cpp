#include "yolo_detector.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    std::string model;
    std::string source;
    std::string classes{"config/classes.txt"};
    std::string output{"cpp_result.jpg"};
    int image_size{640};
    float confidence{0.25F};
    float iou{0.45F};
    float mm_per_pixel{0.0F};
    int max_defects{0};
    bool cuda{false};
    bool show{false};
};

void printUsage() {
    std::cout
        << "Usage:\n"
        << "  laser_inspection --model model.onnx --source image.jpg [options]\n\n"
        << "Options:\n"
        << "  --classes PATH       class-name file (default config/classes.txt)\n"
        << "  --output PATH        annotated image/video path\n"
        << "  --imgsz N            square model input size (default 640)\n"
        << "  --conf F             confidence threshold (default 0.25)\n"
        << "  --iou F              NMS IoU threshold (default 0.45)\n"
        << "  --mm-per-pixel F     optional physical calibration\n"
        << "  --max-defects N      maximum defects for an OK result\n"
        << "  --cuda                use OpenCV CUDA backend\n"
        << "  --show                display the live result\n";
}

Options parseArguments(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto nextValue = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument("Missing value after " + argument);
            }
            return argv[++index];
        };
        if (argument == "--model") {
            options.model = nextValue();
        } else if (argument == "--source") {
            options.source = nextValue();
        } else if (argument == "--classes") {
            options.classes = nextValue();
        } else if (argument == "--output") {
            options.output = nextValue();
        } else if (argument == "--imgsz") {
            options.image_size = std::stoi(nextValue());
        } else if (argument == "--conf") {
            options.confidence = std::stof(nextValue());
        } else if (argument == "--iou") {
            options.iou = std::stof(nextValue());
        } else if (argument == "--mm-per-pixel") {
            options.mm_per_pixel = std::stof(nextValue());
        } else if (argument == "--max-defects") {
            options.max_defects = std::stoi(nextValue());
        } else if (argument == "--cuda") {
            options.cuda = true;
        } else if (argument == "--show") {
            options.show = true;
        } else if (argument == "--help" || argument == "-h") {
            printUsage();
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + argument);
        }
    }
    if (options.model.empty() || options.source.empty()) {
        throw std::invalid_argument("--model and --source are required");
    }
    if (options.image_size <= 0 || options.confidence < 0.0F || options.iou < 0.0F) {
        throw std::invalid_argument("Invalid image size or threshold");
    }
    if (options.max_defects < 0 || options.mm_per_pixel < 0.0F) {
        throw std::invalid_argument("Defect count and calibration cannot be negative");
    }
    return options;
}

std::vector<std::string> readClassNames(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open class-name file: " + path.string());
    }
    std::vector<std::string> names;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            names.push_back(line);
        }
    }
    return names;
}

cv::Mat readImage(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return {};
    }
    const std::vector<unsigned char> bytes(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    return cv::imdecode(bytes, cv::IMREAD_COLOR);
}

void writeImage(const fs::path& path, const cv::Mat& image) {
    const std::string extension = path.has_extension() ? path.extension().string() : ".jpg";
    std::vector<unsigned char> bytes;
    if (!cv::imencode(extension, image, bytes)) {
        throw std::runtime_error("Failed to encode output image");
    }
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

cv::Mat drawDetections(
    const cv::Mat& image,
    const std::vector<Detection>& detections,
    const std::vector<std::string>& names,
    double elapsed_ms,
    float mm_per_pixel,
    int max_defects) {
    cv::Mat canvas = image.clone();
    for (const Detection& detection : detections) {
        const cv::Scalar color(40 + (detection.class_id * 70) % 180,
                               180 - (detection.class_id * 30) % 120,
                               80 + (detection.class_id * 50) % 170);
        cv::rectangle(canvas, detection.box, color, 2);
        std::ostringstream label;
        label << names.at(detection.class_id) << ' ' << std::fixed << std::setprecision(2)
              << detection.confidence;
        if (mm_per_pixel > 0.0F) {
            label << ' ' << detection.box.width * mm_per_pixel << 'x'
                  << detection.box.height * mm_per_pixel << "mm";
        }
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(
            label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        const int text_y = std::max(text_size.height + 4, detection.box.y);
        cv::rectangle(
            canvas,
            cv::Rect(detection.box.x, text_y - text_size.height - 4, text_size.width + 4,
                     text_size.height + baseline + 4),
            color,
            cv::FILLED);
        cv::putText(canvas, label.str(), {detection.box.x + 2, text_y - 2},
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
    const bool accepted = static_cast<int>(detections.size()) <= max_defects;
    const cv::Scalar banner_color = accepted ? cv::Scalar(0, 170, 0) : cv::Scalar(0, 0, 220);
    std::ostringstream banner;
    banner << (accepted ? "OK" : "NG") << " | defects=" << detections.size() << " | "
           << std::fixed << std::setprecision(1) << elapsed_ms << " ms";
    cv::rectangle(canvas, {0, 0}, {canvas.cols, 28}, banner_color, cv::FILLED);
    cv::putText(canvas, banner.str(), {8, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    return canvas;
}

bool isCameraIndex(const std::string& source) {
    return !source.empty() &&
           std::all_of(source.begin(), source.end(), [](unsigned char value) {
               return std::isdigit(value) != 0;
           });
}

int processImage(YoloDetector& detector, const Options& options, const cv::Mat& image) {
    const auto started = std::chrono::steady_clock::now();
    const std::vector<Detection> detections = detector.detect(image);
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - started)
                                  .count();
    const cv::Mat annotated = drawDetections(
        image,
        detections,
        detector.classNames(),
        elapsed_ms,
        options.mm_per_pixel,
        options.max_defects);
    writeImage(options.output, annotated);
    for (const Detection& detection : detections) {
        std::cout << detector.classNames().at(detection.class_id) << " conf="
                  << std::fixed << std::setprecision(3) << detection.confidence
                  << " box=[" << detection.box.x << ',' << detection.box.y << ','
                  << detection.box.width << ',' << detection.box.height << "]\n";
    }
    std::cout << (static_cast<int>(detections.size()) <= options.max_defects ? "OK" : "NG")
              << ", defects=" << detections.size() << ", inference_ms=" << elapsed_ms
              << ", output=" << options.output << '\n';
    if (options.show) {
        cv::imshow("Laser Defect Inspection", annotated);
        cv::waitKey(0);
    }
    return 0;
}

int processStream(YoloDetector& detector, const Options& options) {
    cv::VideoCapture capture;
    if (isCameraIndex(options.source)) {
        capture.open(std::stoi(options.source));
    } else {
        capture.open(options.source);
    }
    if (!capture.isOpened()) {
        throw std::runtime_error("Cannot open video or camera: " + options.source);
    }
    const double fps = capture.get(cv::CAP_PROP_FPS) > 1.0 ? capture.get(cv::CAP_PROP_FPS) : 25.0;
    cv::VideoWriter writer;
    cv::Mat frame;
    std::size_t frame_index = 0;
    while (capture.read(frame)) {
        const auto started = std::chrono::steady_clock::now();
        const std::vector<Detection> detections = detector.detect(frame);
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - started)
                                      .count();
        const cv::Mat annotated = drawDetections(
            frame,
            detections,
            detector.classNames(),
            elapsed_ms,
            options.mm_per_pixel,
            options.max_defects);
        if (!writer.isOpened()) {
            writer.open(options.output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                        annotated.size());
            if (!writer.isOpened()) {
                throw std::runtime_error("Cannot create output video: " + options.output);
            }
        }
        writer.write(annotated);
        if (options.show) {
            cv::imshow("Laser Defect Inspection - Q to quit", annotated);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {
                break;
            }
        }
        ++frame_index;
    }
    std::cout << "Processed " << frame_index << " frames. Output=" << options.output << '\n';
    return 0;
}

int main(int argc, char** argv) {
    try {
        const Options options = parseArguments(argc, argv);
        const std::vector<std::string> class_names = readClassNames(options.classes);
        YoloDetector detector(
            options.model,
            class_names,
            options.image_size,
            options.confidence,
            options.iou,
            options.cuda);

        if (!isCameraIndex(options.source)) {
            const cv::Mat image = readImage(options.source);
            if (!image.empty()) {
                return processImage(detector, options, image);
            }
        }
        return processStream(detector, options);
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        printUsage();
        return 1;
    }
}
