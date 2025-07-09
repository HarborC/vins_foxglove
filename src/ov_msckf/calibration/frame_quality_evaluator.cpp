#include "frame_quality_evaluator.h"
#include <cmath>
#include <algorithm>

bool FrameQualityEvaluator::isFrameAcceptable(const cv::Mat& image, const std::vector<cv::Point2f>& corners) {
    initialize(image);

    // if (!isSharp(image)) return false;
    if (!hasEnoughCorners(corners)) return false;

    // if (!isDistributionReasonable(corners, img_size)) return false;

    std::vector<float> hist_current = local_coverage_.getHistogramOfCorners(corners);

    if (!isNewView(hist_current)) return false;

    // 更新历史记录
    saved_histograms_.push_back(hist_current);  // 只用第一个尺度做视角判断即可

    // 更新全局覆盖矩阵
    global_coverage_.add(corners);
    local_coverage_.add(corners);

    return true;
}

void FrameQualityEvaluator::initialize(const cv::Mat& img) {
    global_coverage_.initialize(img, 10);
    local_coverage_.initialize(img, 20);
}

double FrameQualityEvaluator::calcCoverageRatio() const {
    return global_coverage_.calcCoverageRatio();
}

bool FrameQualityEvaluator::isSharp(const cv::Mat& img) const {
    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img;

    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);

    double variance = stddev.val[0] * stddev.val[0];
    return variance > 100.0;
}

bool FrameQualityEvaluator::hasEnoughCorners(const std::vector<cv::Point2f>& corners) const {
    return corners.size() >= min_corners_num_;
}

bool FrameQualityEvaluator::isNewView(const std::vector<float>& hist_current) const {
    if (saved_histograms_.empty()) return true;

    for (const auto& hist_saved : saved_histograms_) {
        double sim = cosineSimilarity(hist_current, hist_saved);
        if (sim > similarity_threshold_)
            return false;
    }

    return true;
}

double cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, norm_a = 0, norm_b = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (sqrt(norm_a) * sqrt(norm_b));
}

std::vector<float> FrameQualityEvaluator::DGrid::getHistogramOfCorners(const std::vector<cv::Point2f>& corners) const {
    std::vector<float> hist(rows * cols, 0);
    for (const auto& pt : corners) {
        int row = std::min(static_cast<int>(pt.y) / cell_height, rows - 1);
        int col = std::min(static_cast<int>(pt.x) / cell_width, cols - 1);
        hist[row * cols + col]++;
    }

    float total = 0;
    for (float v : hist) total += v;
    if (total > 0)
        for (float& v : hist) v /= total;

    return hist;
}

void FrameQualityEvaluator::DGrid::add(const std::vector<cv::Point2f>& corners) {
    for (const auto& pt : corners) {
        int row = std::min(static_cast<int>(pt.y) / cell_height, rows - 1);
        int col = std::min(static_cast<int>(pt.x) / cell_width, cols - 1);
        coverage[row][col] += 1;

        // 绿色的点画上去
        cv::circle(viz_mat, pt, 1, cv::Scalar(0, 255, 0), -1);
    }
}

void FrameQualityEvaluator::DGrid::initialize(const cv::Mat& img, int window_num) {
    if (!is_initial) {
        is_initial = true;
        int max_length = std::max(img.cols, img.rows);
        int window_size = max_length / window_num;

        is_initial = true;
        cell_width = window_size;
        cell_height = window_size;

        rows = (img.rows + window_size - 1) / window_size;
        cols = (img.cols + window_size - 1) / window_size;

        coverage.resize(rows, std::vector<int>(cols, 0));
        // 创建黑色背景的可视化图像
        viz_mat = cv::Mat::zeros(img.size(), CV_8UC3); // 黑色背景

        // 绘制红色网格线
        for (int r = 0; r < rows; ++r) {
            int y = r * cell_height;
            cv::line(viz_mat, cv::Point(0, y), cv::Point(img.cols, y), cv::Scalar(0, 0, 255), 1);
        }

        for (int c = 0; c < cols; ++c) {
            int x = c * cell_width;
            cv::line(viz_mat, cv::Point(x, 0), cv::Point(x, img.rows), cv::Scalar(0, 0, 255), 1);
        }
    }
}

double FrameQualityEvaluator::DGrid::calcCoverageRatio() const {
    if (!is_initial) return 0.0;
    
    double sum_num = 0;
    for (int i = 0; i < coverage.size(); i++) {
        for (int j = 0; j < coverage[0].size(); j++) {
            if (coverage[i][j] > 0) {
                sum_num++;
            }
        }
    }

    return sum_num / (rows * cols);
}