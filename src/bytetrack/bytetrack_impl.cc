#include "bytetrack_impl.h"


namespace ORB_SLAM2
{

// namespace BYTE_TRACK
// {

cv::Mat BYTETrackerImpl::StaticResize(cv::Mat &img)
{
    float r = min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(Rect(0, 0, re.cols, re.rows)));
    return out;
}

void BYTETrackerImpl::GenerateGridsAndStride(const int target_w, const int target_h, std::vector<int> &strides, std::vector<BYTE_TRACK::GridAndStride> &grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                grid_strides.push_back((BYTE_TRACK::GridAndStride){g0, g1, stride});
            }
        }
    }
}

float BYTETrackerImpl::IntersectionArea(const BYTE_TRACK::Object &a, const BYTE_TRACK::Object &b)
{
    Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

// 快速排序法
void BYTETrackerImpl::QsortDescentInplace(std::vector<BYTE_TRACK::Object> &faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) QsortDescentInplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) QsortDescentInplace(faceobjects, i, right);
        }
    }
}

void BYTETrackerImpl::QsortDescentInplace(std::vector<BYTE_TRACK::Object> &objects)
{
    if (objects.empty())
        return;

    QsortDescentInplace(objects, 0, objects.size() - 1);
}

void BYTETrackerImpl::NMSSortedBboxes(const std::vector<BYTE_TRACK::Object> &faceobjects, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const BYTE_TRACK::Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const BYTE_TRACK::Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = IntersectionArea(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void BYTETrackerImpl::GenerateYoloxProposals(std::vector<BYTE_TRACK::GridAndStride> grid_strides, float *feat_blob, float prob_threshold, std::vector<BYTE_TRACK::Object> &objects)
{
    const int num_anchors = grid_strides.size();

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        const int basic_pos = anchor_idx * (num_class + 5);

        // yolox/models/yolo_head.py decode logic
        float x_center = (feat_blob[basic_pos+0] + grid0) * stride;
        float y_center = (feat_blob[basic_pos+1] + grid1) * stride;
        float w = exp(feat_blob[basic_pos+2]) * stride;
        float h = exp(feat_blob[basic_pos+3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_blob[basic_pos+4];
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                BYTE_TRACK::Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop

    } // point anchor loop
}

float* BYTETrackerImpl::BlobFromImage(cv::Mat &img)
{
    cv::cvtColor(img, img, COLOR_BGR2RGB);

    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    std::vector<float> mean = {0.485, 0.456, 0.406};
    std::vector<float> std = {0.229, 0.224, 0.225};
    for (size_t c = 0; c < channels; c++)
    {
        for (size_t  h = 0; h < img_h; h++)
        {
            for (size_t w = 0; w < img_w; w++)
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (((float)img.at<Vec3b>(h, w)[c]) / 255.0f - mean[c]) / std[c];
            }
        }
    }
    return blob;
}

void BYTETrackerImpl::DecodeOutputs(float *prob, std::vector<BYTE_TRACK::Object> &objects, float scale, const int img_w, const int img_h)
{
    std::vector<BYTE_TRACK::Object> proposals;
    std::vector<int> strides = {8, 16, 32};
    std::vector<BYTE_TRACK::GridAndStride> grid_strides;
    GenerateGridsAndStride(INPUT_W, INPUT_H, strides, grid_strides);
    GenerateYoloxProposals(grid_strides, prob, BBOX_CONF_THRESH, proposals);
    // std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

    QsortDescentInplace(proposals);

    std::vector<int> picked;
    NMSSortedBboxes(proposals, picked, NMS_THRESH);

    int count = picked.size();

    // std::cout << "num of boxes: " << count << std::endl;

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // clip
        // x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        // y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        // x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        // y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
}

void BYTETrackerImpl::DoInference(nvinfer1::IExecutionContext &context, float *input, float *output, const int output_size, cv::Size input_shape)
{
    const nvinfer1::ICudaEngine &engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = engine.getMaxBatchSize();

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

BYTETrackerImpl::BYTETrackerImpl(const std::string &engine_file_path)
{
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    out_dims = engine->getBindingDimensions(1);
    output_size = 1;
    for (int j = 0; j < out_dims.nbDims; j++)
    {
        output_size *= out_dims.d[j];
    }
    prob = new float[output_size];
    mbFinishRequested = false;

    tracker = new BYTE_TRACK::BYTETracker(fps, 30);
}

BYTETrackerImpl::~BYTETrackerImpl()
{
    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

void BYTETrackerImpl::Detect()
{
    {
        unique_lock<mutex> lock(mMutexYOLOXQueue);
        frame = IMGQueue.front();
        IMGQueue.clear();
    }

    if (frame.empty())
    {
        std::cout << "[ERRO] Read frame failed!" << std::endl;
        return;
    }

    int img_w = frame.cols;
    int img_h = frame.rows;
    Mat pr_img = StaticResize(frame);

    float *blob;
    blob = BlobFromImage(pr_img);
    float scale = std::min(INPUT_W / (frame.cols * 1.0), INPUT_H / (frame.rows * 1.0));

    // run inference
    // auto start = std::chrono::system_clock::now();
    DoInference(*context, blob, prob, output_size, pr_img.size());
    std::vector<BYTE_TRACK::Object> objects;
    DecodeOutputs(prob, objects, scale, img_w, img_h);
    std::vector<BYTE_TRACK::STrack> output_stracks = tracker->update(objects);
    // auto end = chrono::system_clock::now();
    // cout << "[INFO] Cost Time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() /1000.0 <<  "ms." << endl;
    // auto cost_ms = chrono::duration_cast<chrono::microseconds>(end - start).count()/1000.0;

    delete[] blob;

    {
        unique_lock<mutex> lock(mMutexResQueue);
        ResQueue.push_back(output_stracks);
        // std::cout << "ResQueue.size() : " << ResQueue.size() << std::endl;
    }
}

void BYTETrackerImpl::GetResult(std::vector<BYTE_TRACK::STrack> &output)
{
    unique_lock<mutex> lock(mMutexResQueue);
    output = ResQueue.front();
    ResQueue.clear();
}

void BYTETrackerImpl::Run()
{
    mbFinished = false;

    while (1)
    {
        // Check if there are keyframes in the queue
        if (CheckFrames())
        {
            Detect();
        }
        else
        {
            usleep(5000);
        }

        if (CheckFinish())
            break;
    }

    SetFinish();
}

void BYTETrackerImpl::SetTracker(Tracking *pTracker)
{
    mpTracker = pTracker;
}

bool BYTETrackerImpl::CheckFrames()
{
    unique_lock<mutex> lock(mMutexYOLOXQueue);
    return (!IMGQueue.empty());
}

bool BYTETrackerImpl::CheckResult()
{
    unique_lock<mutex> lock(mMutexResQueue);
    return (!ResQueue.empty());
}

void BYTETrackerImpl::InsertImage(cv::Mat rgb)
{
    unique_lock<mutex> lock(mMutexYOLOXQueue);
    IMGQueue.push_back(rgb);
}

void BYTETrackerImpl::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool BYTETrackerImpl::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void BYTETrackerImpl::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool BYTETrackerImpl::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

BYTE_TRACK::Object BYTETrackerImpl::STrack2Object(BYTE_TRACK::STrack &strack)
{
    BYTE_TRACK::Object obj;
    obj.label = strack.label;
    obj.prob = strack.score;
    cv::Rect rect(strack.tlwh[0], strack.tlwh[1], strack.tlwh[2], strack.tlwh[3]);
    obj.rect = rect;
    obj.idx = strack.track_id;
    obj.nFrame = strack.frame_id;
    return obj;
}

// }
}
