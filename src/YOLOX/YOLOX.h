#ifndef _YOLOX_H_
#define _YOLOX_H_

#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "Logging.h"
#include "Tracking.h"

namespace ORB_SLAM2
{
class Tracking;

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

#define DEVICE 0 // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.3

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    int nFrames = 0;
    int lostFrames = 0;
    int idx = 0;
    int similarity = -1;
    int track_id = -1;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

class YOLOX
{
private:
    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;

    float *prob;

    cv::Mat frame;

    nvinfer1::Dims out_dims;
    int output_size = 1;
    const int NUM_CLASSES = 1;

    Logger gLogger;

    const int INPUT_W = 640;
    const int INPUT_H = 640;
    const char *INPUT_BLOB_NAME = "input_0";
    const char *OUTPUT_BLOB_NAME = "output_0";

    // yolox thread
    Tracking *mpTracker;
    bool mbFinishRequested;
    bool mbFinished;
    bool mbGetResult;

    std::list<cv::Mat> IMGQueue;
    std::list<std::vector<Object> > ResQueue;

    std::mutex mMutexFinish;
    std::mutex mMutexYOLOXQueue;
    std::mutex mMutexResQueue;

public:
    YOLOX(const std::string &engine_file_path);

    ~YOLOX();

    cv::Mat StaticResize(cv::Mat &img);

    int GenerateGridsAndStride(std::vector<int> &strides, std::vector<GridAndStride> &grid_strides);

    float IntersectionArea(const Object &a, const Object &b);

    void QsortDescentInplace(std::vector<Object> &faceobjects, int left, int right);

    void QsortDescentInplace(std::vector<Object> &objects);

    void NmsSortedBboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, float nms_threshold);

    void GenerateYoloxProposals(std::vector<GridAndStride> grid_strides, float *feat_blob, float prob_threshold, std::vector<Object> &objects);

    float *BlobFromImage(cv::Mat &img);

    void DecodeOutputs(float *prob, std::vector<Object> &objects, float scale, const int img_w, const int img_h);

    void DrawObjects(const cv::Mat &bgr, const std::vector<Object> &objects, const std::string &windowName);

    void DoInference(nvinfer1::IExecutionContext &context, float *input, float *output, const int output_size, cv::Size input_shape);

    void Detect();

    // yolox thread
    void Run();
    void SetTracker(Tracking *pTracker);
    void InsertImage(cv::Mat rgb);
    void SetFinish();
    void RequestFinish();
    bool CheckFinish();
    bool CheckResult();
    bool isFinished();
    bool CheckFrames();
    void GetResult(std::vector<Object> &output);
};

const float color_list[80][3] =
    {
        {0.000, 0.447, 0.741},
        {0.850, 0.325, 0.098},
        {0.929, 0.694, 0.125},
        {0.494, 0.184, 0.556},
        {0.466, 0.674, 0.188},
        {0.301, 0.745, 0.933},
        {0.635, 0.078, 0.184},
        {0.300, 0.300, 0.300},
        {0.600, 0.600, 0.600},
        {1.000, 0.000, 0.000},
        {1.000, 0.500, 0.000},
        {0.749, 0.749, 0.000},
        {0.000, 1.000, 0.000},
        {0.000, 0.000, 1.000},
        {0.667, 0.000, 1.000},
        {0.333, 0.333, 0.000},
        {0.333, 0.667, 0.000},
        {0.333, 1.000, 0.000},
        {0.667, 0.333, 0.000},
        {0.667, 0.667, 0.000},
        {0.667, 1.000, 0.000},
        {1.000, 0.333, 0.000},
        {1.000, 0.667, 0.000},
        {1.000, 1.000, 0.000},
        {0.000, 0.333, 0.500},
        {0.000, 0.667, 0.500},
        {0.000, 1.000, 0.500},
        {0.333, 0.000, 0.500},
        {0.333, 0.333, 0.500},
        {0.333, 0.667, 0.500},
        {0.333, 1.000, 0.500},
        {0.667, 0.000, 0.500},
        {0.667, 0.333, 0.500},
        {0.667, 0.667, 0.500},
        {0.667, 1.000, 0.500},
        {1.000, 0.000, 0.500},
        {1.000, 0.333, 0.500},
        {1.000, 0.667, 0.500},
        {1.000, 1.000, 0.500},
        {0.000, 0.333, 1.000},
        {0.000, 0.667, 1.000},
        {0.000, 1.000, 1.000},
        {0.333, 0.000, 1.000},
        {0.333, 0.333, 1.000},
        {0.333, 0.667, 1.000},
        {0.333, 1.000, 1.000},
        {0.667, 0.000, 1.000},
        {0.667, 0.333, 1.000},
        {0.667, 0.667, 1.000},
        {0.667, 1.000, 1.000},
        {1.000, 0.000, 1.000},
        {1.000, 0.333, 1.000},
        {1.000, 0.667, 1.000},
        {0.333, 0.000, 0.000},
        {0.500, 0.000, 0.000},
        {0.667, 0.000, 0.000},
        {0.833, 0.000, 0.000},
        {1.000, 0.000, 0.000},
        {0.000, 0.167, 0.000},
        {0.000, 0.333, 0.000},
        {0.000, 0.500, 0.000},
        {0.000, 0.667, 0.000},
        {0.000, 0.833, 0.000},
        {0.000, 1.000, 0.000},
        {0.000, 0.000, 0.167},
        {0.000, 0.000, 0.333},
        {0.000, 0.000, 0.500},
        {0.000, 0.000, 0.667},
        {0.000, 0.000, 0.833},
        {0.000, 0.000, 1.000},
        {0.000, 0.000, 0.000},
        {0.143, 0.143, 0.143},
        {0.286, 0.286, 0.286},
        {0.429, 0.429, 0.429},
        {0.571, 0.571, 0.571},
        {0.714, 0.714, 0.714},
        {0.857, 0.857, 0.857},
        {0.000, 0.447, 0.741},
        {0.314, 0.717, 0.741},
        {0.50, 0.5, 0}};


}
#endif
