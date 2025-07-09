#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class FrameQualityEvaluator {
public:
    struct DGrid {
        bool is_initial = false;
        int rows, cols;  // 网格的行数和列数
        int cell_width, cell_height;  // 每个网格的宽度和高度
        std::vector<std::vector<int>> coverage;  // 每个网格的覆盖状态
        cv::Mat viz_mat;

        std::vector<float> getHistogramOfCorners(const std::vector<cv::Point2f>& corners) const;
        void add(const std::vector<cv::Point2f>& corners);
        void initialize(const cv::Mat& img, int window_num);
        double calcCoverageRatio() const;
    };

public:
    // 构造函数：传入多个格子尺寸
    FrameQualityEvaluator() {};

    // 主接口：判断当前帧是否合格
    bool isFrameAcceptable(const cv::Mat& image,
                           const std::vector<cv::Point2f>& corners);

    // 获取已保存帧的数量
    int getSavedFrameCount() const { return saved_histograms_.size(); }

    // 判断是否达到了足够的分布覆盖（可作为终止条件）
    double calcCoverageRatio() const;

public:
    // 图像质量评估函数
    bool isSharp(const cv::Mat& img) const;
    bool hasEnoughCorners(const std::vector<cv::Point2f>& corners) const;
    bool isDistributionReasonable(const std::vector<cv::Point2f>& corners,
                                  const cv::Size& img_size) const;
    bool isNewView(const std::vector<float>& hist_current) const;
    void initialize(const cv::Mat& img);

    // 多尺度配置
    int min_corners_num_ = 12;
    double similarity_threshold_ = 0.75;

    // 历史记录
    DGrid global_coverage_;
    DGrid local_coverage_;
    std::vector<std::vector<float>> saved_histograms_;             // 用于新视角判断
};

double cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);