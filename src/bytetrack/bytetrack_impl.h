#ifndef _BYTETRACK_IMPL_H_
#define _BYTETRACK_IMPL_H_

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
#include "logging.h"
#include "BYTETracker.h"
#include "Tracking.h"

namespace ORB_SLAM2
{

class Tracking;

// namespace BYTE_TRACK
// {


#define CHECK(status)                                \
    do                                               \
    {                                                \
        auto ret = (status);                         \
        if (ret != 0)                                \
        {                                            \
            cerr << "Cuda failure: " << ret << endl; \
            abort();                                 \
        }                                            \
    } while (0)

#define DEVICE 0 // GPU id
#define NMS_THRESH 0.7
#define BBOX_CONF_THRESH 0.1


class BYTETrackerImpl
{
private:

    const int fps = 30;

    // const int num_class = 1;
    // const int INPUT_W = 1088;
    // const int INPUT_H = 608;

    const int num_class = 1;
    const int INPUT_W = 640;
    const int INPUT_H = 640;

    const char *INPUT_BLOB_NAME = "input_0";
    const char *OUTPUT_BLOB_NAME = "output_0";

    BYTE_TRACK::Logger gLogger;

    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;

    float *prob;
    cv::Mat frame;
    nvinfer1::Dims out_dims;
    int output_size = 1;

    Tracking *mpTracker;
    BYTE_TRACK::BYTETracker *tracker;

    bool mbFinishRequested;
    bool mbFinished;
    bool mbGetResult;

    std::list<cv::Mat> IMGQueue;
    std::list<std::vector<BYTE_TRACK::STrack>> ResQueue;

    std::mutex mMutexFinish;
    std::mutex mMutexYOLOXQueue;
    std::mutex mMutexResQueue;

public:
    BYTETrackerImpl(const std::string &engine_file_path);
    ~BYTETrackerImpl();

    static float IntersectionArea(const BYTE_TRACK::Object &a, const BYTE_TRACK::Object &b);

    static void QsortDescentInplace(std::vector<BYTE_TRACK::Object> &faceobjects, int left, int right);

    static void QsortDescentInplace(std::vector<BYTE_TRACK::Object> &objects);

    static void NMSSortedBboxes(const std::vector<BYTE_TRACK::Object> &faceobjects, std::vector<int> &picked, float nms_threshold);

    static BYTE_TRACK::Object STrack2Object(BYTE_TRACK::STrack &strack);

    cv::Mat StaticResize(cv::Mat &img);

    void GenerateGridsAndStride(const int target_w, const int target_h, std::vector<int> &strides, std::vector<BYTE_TRACK::GridAndStride> &grid_strides);

    void DoInference(nvinfer1::IExecutionContext &context, float *input, float *output, const int output_size, cv::Size input_shape);

    void GenerateYoloxProposals(std::vector<BYTE_TRACK::GridAndStride> grid_strides, float *feat_blob, float prob_threshold, std::vector<BYTE_TRACK::Object> &objects);

    float *BlobFromImage(cv::Mat &img);

    void DecodeOutputs(float *prob, std::vector<BYTE_TRACK::Object> &objects, float scale, const int img_w, const int img_h);

    void Detect();

    void GetResult(std::vector<BYTE_TRACK::STrack> &output);

    void Run();
    void SetTracker(Tracking *pTracker);
    void InsertImage(cv::Mat rgb);
    void SetFinish();
    void RequestFinish();
    bool CheckFinish();
    bool CheckResult();
    bool isFinished();
    bool CheckFrames();
};

// }


}
#endif
