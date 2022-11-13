/**
* This file is part of ORB-SLAM2.
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* Modification: EAO-SLAM
* Version: 1.0
* Created: 03/21/2019
* Author: Yanmin Wu
* E-mail: wuyanminmax@gmail.com
*/
#include <pcl/search/impl/search.hpp>

#ifndef PCL_NO_PRECOMPILE
#include <pcl/impl/instantiate.hpp>
#include <pcl/point_types.h>
PCL_INSTANTIATE(Search, PCL_POINT_TYPES)
#endif // PCL_NO_PRECOMPILE

#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "Initializer.h"
#include "Optimizer.h"
#include "PnPsolver.h"

#include <iostream>

#include <mutex>

#include "Object.h"
#include "FrameDrawer.h"
#include "Global.h"

#include <cmath>
#include <algorithm>

using namespace std;

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
typedef Eigen::Matrix<float,5,1> Vector5f;

namespace ORB_SLAM2
{

bool Tracking::mbReadedGroundtruth = false;
int frame_id_tracking = -1;

// rank.
// typedef Eigen::Vector4f VI;
typedef Vector5f VI;
int index=0;
bool VIC(const VI& lhs, const VI& rhs)
{
    return lhs[index] > rhs[index];
}

Tracking::Tracking( System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer,
                    MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase *pKFDB, const string &strSettingPath, const int sensor, const string &flag, const bool SemanticOnline ):
                    mState(NO_IMAGES_YET),
                    mSensor(sensor),
                    mbOnlyTracking(false),
                    mbVO(false),
                    mpORBVocabulary(pVoc),
                    mpKeyFrameDB(pKFDB),
                    mpInitializer(static_cast<Initializer *>(NULL)),
                    mpSystem(pSys),
                    mpFrameDrawer(pFrameDrawer),
                    mpMapDrawer(pMapDrawer),
                    mpMap(pMap),
                    mnLastRelocFrameId(0),
                    mbSemanticOnline(SemanticOnline)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if (fps == 0)
        fps = 30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl
         << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if (DistCoef.rows == 5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if (mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (sensor == System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (sensor == System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    // demo flag.
    mflag = flag;


    cout << endl
         << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if (sensor == System::STEREO || sensor == System::RGBD)
    {
        mThDepth = mbf * (float)fSettings["ThDepth"] / fx;
        cout << endl
             << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if (sensor == System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if (mDepthMapFactor == 0)
            mDepthMapFactor = 1;
        else
            mDepthMapFactor = 1.0f / mDepthMapFactor;
    }

    // STEP read groundtruth ++++++++++++++++++++++++++++++++++++++++++++++++
    // notice: Only the camera pose of the initial frame is used to determine the ground plane normal.
    miConstraintType = fSettings["ConstraintType"];
    if ( miConstraintType != 0 && miConstraintType != 1 && miConstraintType != 2){
        std::cerr << ">>>>>> [WARRNING] USE NO PARAM CONSTRAINT TYPE!" << std::endl;
        miConstraintType = 0;
    }

    if (miConstraintType == 1)
    {
       if (mbReadedGroundtruth == false)
        {
            std::string filePath = WORK_SPACE_PATH + "/data/groundtruth.txt";
            ifstream infile(filePath, ios::in);
            if (!infile.is_open())
            {
                cout << "tum groundtruth file open fail" << endl;
                exit(233);
            }
            else
            {
                std::cout << "read groundtruth.txt" << std::endl;
                mbReadedGroundtruth = true;
            }

            vector<double> row;
            double tmp;
            string line;
            cv::Mat cam_pose_mat;

            string s0;
            getline(infile, s0);
            getline(infile, s0);
            getline(infile, s0);

            // save as vector<vector<int>> _mat format.
            while (getline(infile, line))
            {
                // string to int.
                istringstream istr(line);
                while (istr >> tmp)
                {
                    row.push_back(tmp);
                }

                mGroundtruth_mat.push_back(row); // vector<int> row.

                row.clear();
                istr.clear();
                line.clear();
            }
            infile.close();
        }
    }
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing = pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer = pViewer;
}

#ifdef USE_YOLOX_AND_NO_TRACK
void Tracking::SetSemanticer(YOLOX *detector)
{
    Semanticer = detector;
}
#else
void Tracking::SetSemanticer(BYTETrackerImpl *detector)
{
    ByteTracker = detector;
}
#endif

cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp, const bool bSemanticOnline)
{
    frame_id_tracking++;

    cv::Mat rawImage = imRGB.clone();

    // JBF filter
    // JBF jointBilateralFilter;
    // cv::Mat DepthFilter = jointBilateralFilter.Processor(rawImage, imDepth);
    // JBF filter end

    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }

    // step 2 ：将深度相机的disparity转为Depth , 也就是转换成为真正尺度下的深度
    if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
        imDepth.convertTo(    //将图像转换成为另外一种数据类型,具有可选的数据大小缩放系数
            imDepth,          //输出图像
            CV_32F,           //输出图像的数据类型
            mDepthMapFactor); //缩放系数

    cv::Mat mImDepth = imDepth.clone();

    vector<vector<int> > _mat;
    mCurrentFrame.have_detected = false;
    mCurrentFrame.finish_detected = false;

    if (bSemanticOnline)
    {
        auto start = std::chrono::system_clock::now();
        int StopCount = 0;
        std::vector<ORB_SLAM2::Object> currentObjs;

#ifdef USE_YOLOX_AND_NO_TRACK
        while (1)
        {
            if (Semanticer->CheckResult())
            {
                Semanticer->GetResult(currentObjs);
                break;
            }
            // Semanticer
            if (StopCount >= 5 && mState != NOT_INITIALIZED)
            {
                std::cout << "[WARRNING] DETECTER ERROR!" << std::endl;
                break;
            }
            usleep(5000);
            StopCount++;
        }

#else
        while (1){
            if (ByteTracker->CheckResult()){
                ByteTracker->GetResult(track_anchors);
                break;
            }
            // ByteTracker
            if (StopCount >= 5 && mState != NOT_INITIALIZED){
                std::cout << "[WARRNING] DETECTER ERROR!" << std::endl;
                break;
            }
            usleep(5000);
            StopCount++;
        }
        if (track_anchors.size() == 0)
        {
            std::cout << "[WARNNING] OBJECTS SIZE IS ZERO" << std::endl;
        }
#endif

        // 耗时计算
        auto end = std::chrono::system_clock::now();
        // std::cout << "[INFO] Cost Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        std::vector<BoxSE> boxes_online;
#ifdef USE_YOLOX_AND_NO_TRACK
        for (auto &strack : currentObjs)
        {
#else
        for (auto &strack : track_anchors)
        {
            auto objInfo = BYTETrackerImpl::STrack2Object(strack);

#endif

            auto objInfo = strack;
            // 0: person; 26: handbag; 28: suitcase; 32: sports ball; 39: bottle; 40: wine glass; 41: cup; 56: chair; 57: couch; 58: potted plant; 59: bed; 60: dining table; 62: tv; 63: laptop; 64:mouse; 65: remote; 66: keyboard; 67: cell phone; 73: book; 74: clock; 75: vase;
            // if (
            // objInfo.label != 0  && objInfo.label != 26 && objInfo.label != 28 && objInfo.label != 32 && objInfo.label != 39 && objInfo.label != 40 && objInfo.label != 41 && objInfo.label != 56 && objInfo.label != 57 && objInfo.label != 58 && objInfo.label != 59 && objInfo.label != 60 && objInfo.label != 63 && objInfo.label != 64 &&objInfo.label != 65 && objInfo.label != 66 && objInfo.label != 67 && objInfo.label != 73 && objInfo.label != 74 && objInfo.label != 75)
            //     continue;

            BoxSE box;
            box.m_class = 45;
            box.m_score = objInfo.prob;
            box.x = objInfo.rect.x;
            box.y = objInfo.rect.y;
            box.width = objInfo.rect.width;
            box.height = objInfo.rect.height;
            box.m_track_id = objInfo.track_id;
            // box.m_class_name = "";
            boxes_online.push_back(box);
        }
        std::sort(boxes_online.begin(), boxes_online.end(), [](BoxSE a, BoxSE b) -> bool
                  { return a.m_score > b.m_score; });

        refine_flag = 0;
        for (auto &box: boxes_online){
            // 如果框过大，超过图像的3/4，那么则使用精细分割
            if (box.m_class == 0 && box.area() > 0.75 * 640 * 480){
                refine_flag == 1;
            }
        }
        if (refine_flag == 1){
            LightTrack();
            cv::Mat imMask;
            mGeometry.GeometricModelCorrection(mCurrentFrame, mImGray, imMask);

            mCurrentFrame = Frame(rawImage, // new: color image.
                                  mImGray,
                                  imDepth, // new: 深度图像
                                  timestamp,
                                  mpORBextractorLeft,
                                  mpORBVocabulary,
                                  mK,
                                  mDistCoef,
                                  mbf,
                                  mThDepth,
                                  mImGray,
                                  imMask);
        }else{
            mCurrentFrame = Frame(rawImage, // new: color image.
                                  mImGray,
                                  imDepth, // new: 深度图像
                                  timestamp,
                                  mpORBextractorLeft,
                                  mpORBVocabulary,
                                  mK,
                                  mDistCoef,
                                  mbf,
                                  mThDepth,
                                  mImGray,
                                  boxes_online);
        }
        mCurrentFrame.mDepthMapFactor = mDepthMapFactor;

        // save to current frame.
        // for (vector<BoxSE>::iterator iter = boxes_online.begin(); iter != boxes_online.end();){
        //     if (iter->m_class == 0)
        //         iter = boxes_online.erase(iter);
        //     else
        //         iter++;
        // }
        // mCurrentFrame.boxes = boxes_online;

        // std::vector<BoxSE> --> Eigen::MatrixXd.
        int i = 0;
        Eigen::MatrixXd eigenMat;

        eigenMat.resize((int)mCurrentFrame.boxes.size(), 6);
        for (auto &box : mCurrentFrame.boxes)
        {
            eigenMat(i, 0) = box.x;
            eigenMat(i, 1) = box.y;
            eigenMat(i, 2) = box.width;
            eigenMat(i, 3) = box.height;
            eigenMat(i, 4) = box.m_score;
            eigenMat(i, 5) = box.m_track_id;
            i++;
        }
        // save to current frame.
        mCurrentFrame.boxes_eigen = eigenMat;
    }
    else
    {
        // offline object box.
        std::string filePath = WORK_SPACE_PATH + "/data/yolo_txts/" + to_string(timestamp) + ".txt";
        ifstream infile(filePath, ios::in);
        if (!infile.is_open())
        {
            cout << "yolo_detection file open fail" << endl;
            exit(233);
        }
        // else
        //     cout << "read offline boundingbox" << endl;

        vector<int> row; // one row, one object.
        int tmp;
        string line;

        // save as vector<vector<int>> format.
        while (getline(infile, line))
        {
            // string to int.
            istringstream istr(line);
            while (istr >> tmp)
            {
                row.push_back(tmp);
            }

            _mat.push_back(row); // vector<vector<int>>.
            row.clear();
            istr.clear();
            line.clear();
        }
        infile.close();

        //  vector<vector<int>> --> std::vector<BoxSE>.
        std::vector<BoxSE> boxes_offline;
        for (auto &mat_row : _mat)
        {
            BoxSE box;
            box.m_class = mat_row[0];
            box.m_score = mat_row[5];
            box.x = mat_row[1];
            box.y = mat_row[2];
            box.width = mat_row[3];
            box.height = mat_row[4];
            // box.m_class_name = "";
            boxes_offline.push_back(box);
        }
        std::sort(boxes_offline.begin(), boxes_offline.end(), [](BoxSE a, BoxSE b) -> bool
                  { return a.m_score > b.m_score; });
        // save to current frame.

        mCurrentFrame = Frame(rawImage, // new: color image.
                              mImGray,
                              imDepth, // new: 深度图像
                              timestamp,
                              mpORBextractorLeft,
                              mpORBVocabulary,
                              mK,
                              mDistCoef,
                              mbf,
                              mThDepth,
                              mImGray,
                              boxes_offline);

        // mCurrentFrame.boxes = boxes_offline;
        mCurrentFrame.mDepthMapFactor = mDepthMapFactor;

        // std::vector<BoxSE> --> Eigen::MatrixXd.
        int i = 0;
        Eigen::MatrixXd eigenMat;

        eigenMat.resize((int)mCurrentFrame.boxes.size(), 5);
        for (auto &box : mCurrentFrame.boxes)
        {
            // std::cout << box.m_class << " " << box.x << " " << box.y << " "
            //           << box.width << " " << box.height << " " << box.m_score << std::endl;
            /**
            * keyboard  66 199 257 193 51
            * mouse     64 377 320 31 39
            * cup       41 442 293 51 63
            * tvmonitor 62 232 93 156 141
            * remote    65 44 260 38 57
            */
            eigenMat(i, 0) = box.x;
            eigenMat(i, 1) = box.y;
            eigenMat(i, 2) = box.width;
            eigenMat(i, 3) = box.height;
            eigenMat(i, 4) = box.m_score;
            i++;
        }
        // save to current frame.
        mCurrentFrame.boxes_eigen = eigenMat;
    }
    // there are objects in current frame?
    if (!mCurrentFrame.boxes.empty())
        mCurrentFrame.have_detected = true;
    // object detection.------------------------------------------------------------------------

    // STEP get current camera groundtruth by timestamp. +++++++++++++++++++++++++++++++++++++++++++
    // notice: only use the first frame's pose.
    string timestamp_string = to_string(timestamp);
    string timestamp_short_string = timestamp_string.substr(0, timestamp_string.length() - 4);

    Eigen::MatrixXd truth_frame_poses(1, 8); // camera pose Eigen format.
    cv::Mat cam_pose_mat; // camera pose Mat format.
    if (miConstraintType == 1){
        // std::cout << "mGroundtruth_mat " << mGroundtruth_mat.size() << std::endl;
        for (auto &row : mGroundtruth_mat)
        {
            string row_string = to_string(row[0]);
            double delta_time = fabs(row[0] - timestamp);
            // if (delta_time < 1)
            //     std::cout << " [INFO] delta_time : " << delta_time << std::endl;
            // std::cout << " [INFO] cur time : " << std::setprecision(20) << timestamp << std::endl;
            // std::cout << " [INFO] tag time : " << std::setprecision(20) << row[0] << std::endl;

            string row_short_string = row_string.substr(0, row_string.length() - 4);

            // if (row_short_string == timestamp_short_string)
            if (delta_time <= 0.05)
            {
                // std::cout << " [INFO] result_time : " << row_short_string << std::endl;
                // vector --> Eigen.
                for (int i = 0; i < (int)row.size(); i++)
                {
                    truth_frame_poses(0) = row[0];
                    truth_frame_poses(1) = row[1];
                    truth_frame_poses(2) = row[2];
                    truth_frame_poses(3) = row[3];
                    truth_frame_poses(4) = row[4];
                    truth_frame_poses(5) = row[5];
                    truth_frame_poses(6) = row[6];
                    truth_frame_poses(7) = row[7];
                }

                // Eigen --> SE3.
                g2o::SE3Quat cam_pose_se3(truth_frame_poses.row(0).tail<7>());
                // std::cout << "cam_pose_se3\n" << cam_pose_se3 << std::endl;

                // SE3 --> Mat.
                cam_pose_mat = Converter::toCvMat(cam_pose_se3);

                // save to current frame.
                mCurrentFrame.mGroundtruthPose_mat = cam_pose_mat;
                if (!mCurrentFrame.mGroundtruthPose_mat.empty())
                {
                    mCurrentFrame.mGroundtruthPose_eigen = Converter::toEigenMatrixXd(mCurrentFrame.mGroundtruthPose_mat);
                }
                break;
            }
            else
            {
                mCurrentFrame.mGroundtruthPose_mat = cv::Mat::eye(4, 4, CV_32F);
                mCurrentFrame.mGroundtruthPose_eigen = Eigen::Matrix4d::Identity(4, 4);
            }
        }
    }else if (miConstraintType ==2){
        mCurrentFrame.mGroundtruthPose_eigen = INIT_POSE;
        // mCurrentFrame.mGroundtruthPose_mat = cv::Mat::eye(4, 4, CV_32F);
        cv::Mat cv_mat_32f;
        cv::eigen2cv(mCurrentFrame.mGroundtruthPose_eigen, cv_mat_32f);
        cv_mat_32f.convertTo(mCurrentFrame.mGroundtruthPose_mat, CV_32F);
    }else{
        mCurrentFrame.mGroundtruthPose_mat = cv::Mat::eye(4, 4, CV_32F);
    }
    // get the camera groundtruth by timestamp. ----------------------------------------------------------------------
    Track();
    if (refine_flag == 1)
    {
        mGeometry.GeometricModelUpdateDB(mCurrentFrame);
    }

    return mCurrentFrame.mTcw.clone();
}


void Tracking::Track()
{
    if (mState == NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }


    mLastProcessedState = mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if (mState == NOT_INITIALIZED)
    {
        if (mSensor == System::STEREO || mSensor == System::RGBD){
            StereoInitialization();
        }
        else{
            MonocularInitialization();
        }
        mpFrameDrawer->Update(this);
        if (mState != OK)
            return;
        // mSensor = System::MONOCULAR;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if (!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if (mState == OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    // note [EAO] for this opensource version, the mainly modifications are in the TrackWithMotionModel().
                    bOK = TrackWithMotionModel();

                    if (!bOK){
                        bOK = TrackReferenceKeyFrame();
                    }

                }
            }
            else
            {
                bOK = Relocalization();
            }
        }
        else
        {
            // Only Tracking: Local Mapping is deactivated

            if (mState == LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if (!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if (!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint *> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if (!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if (bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if (mbVO)
                        {
                            for (int i = 0; i < mCurrentFrame.N; i++)
                            {
                                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if (bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if (!mbOnlyTracking)
        {
            if (bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if (bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if (bOK)
            mState = OK;
        else
            mState = LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if (bOK)
        {
            // add plane -------------------
            // Update Planes
            // std::cout << "[DEBUG] mnPlaneNum is " << mCurrentFrame.mnPlaneNum << std::endl;
            for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i)
            {
                MapPlane *pMP = mCurrentFrame.mvpMapPlanes[i];
                if (pMP && pMP->mbSeen)
                {
                    // 更新边界点
                    // TODO: 为什么不更新新生成的平面？
                    pMP->UpdateBoundary(mCurrentFrame, i);
                }
                else
                {
                    mCurrentFrame.mbNewPlane = true;
                }
            }
            // add plane end -----------------

            // Update motion model
            if (!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                mVelocity = mCurrentFrame.mTcw * LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean temporal point matches
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP)
                    if (pMP->Observations() < 1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit != lend; lit++)
            {
                MapPoint *pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if (NeedNewKeyFrame() == 1)
                CreateNewKeyFrame(false);
            else if (NeedNewKeyFrame() == 2)    // note [EAO] create keyframes by the new object.
            {
                CreateNewKeyFrame(true);
            }

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if (mState == LOST)
        {
            if (mpMap->KeyFramesInMap() <= 5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if (!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState == LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState == LOST);
    }
}

void Tracking::StereoInitialization()
{

    if (mCurrentFrame.N > 50)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

        mInitialFrame = Frame(mCurrentFrame);
        mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
        mInitialSecendFrame = Frame(mCurrentFrame); // [EAO] the second frame when initialization.

        // Create KeyFrame
        KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
        KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        // pKFini->ComputeBoW();
        // pKFcur->ComputeBoW();

        // Insert KFs in the map
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);

        // Create MapPoints and asscoiate to KeyFrame
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap);
                pNewMP->AddObservation(pKFini, i);
                pNewMP->AddObservation(pKFcur, i);

                pKFini->AddMapPoint(pNewMP, i);
                pKFcur->AddMapPoint(pNewMP, i);

                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();

                // mCurrentFrame.mvbOutlier[i] = false;
                mCurrentFrame.mvpMapPoints[i] = pNewMP;

                mpMap->AddMapPoint(pNewMP);

            }
        }

        // Update Connections
        // pKFini->UpdateConnections();
        // pKFcur->UpdateConnections();
        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        pKFini->SetPose(pKFini->GetPose());
        pKFcur->SetPose(pKFcur->GetPose());

        for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i)
        {
            // 平面的系数转到世界坐标系下是H^{-T}
            // 虽然是p3D，但是好像是4维
            cv::Mat p3D = mCurrentFrame.ComputePlaneWorldCoeff(i);
            MapPlane *pNewMP = new MapPlane(p3D, pKFini, i, mpMap);
            mpMap->AddMapPlane(pNewMP);
            pKFini->AddMapPlane(pNewMP, i);
        }

        {
            // NOTE [EAO] rotate the world coordinate to the initial frame (groundtruth provides the normal vector of the ground).
            // only use the groundtruth of the first frame.
            // TODO: 替换这种方法
            cv::Mat InitToGround = mCurrentFrame.mGroundtruthPose_mat;
            // cv::Mat InitToGround = cv::Mat::eye(4, 4, CV_32F);
            cv::Mat R = InitToGround.rowRange(0, 3).colRange(0, 3);
            cv::Mat t = InitToGround.rowRange(0, 3).col(3);
            cv::Mat Rinv = R.t();
            cv::Mat Ow = -Rinv * t;
            cv::Mat GroundToInit = cv::Mat::eye(4, 4, CV_32F);
            Rinv.copyTo(GroundToInit.rowRange(0, 3).colRange(0, 3));
            Ow.copyTo(GroundToInit.rowRange(0, 3).col(3));

            // std::cout << GroundToInit << std::endl;

            bool build_worldframe_on_ground = true;
            std::vector<MapPoint *> vpAllMapPoints = pKFcur->GetMapPointMatches();
            if (build_worldframe_on_ground) // transform initial pose and map to ground frame
            {
                pKFini->SetPose(pKFini->GetPose() * GroundToInit);
                pKFcur->SetPose(pKFcur->GetPose() * GroundToInit);

                for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
                {
                    if (vpAllMapPoints[iMP])
                    {
                        MapPoint *pMP = vpAllMapPoints[iMP];
                        pMP->SetWorldPos(InitToGround.rowRange(0, 3).colRange(0, 3) * pMP->GetWorldPos() + InitToGround.rowRange(0, 3).col(3));
                    }
                }
            }
            // [EAO] rotate the world coordinate to the initial frame -----------------------------------------------------------
        }

        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);

        // mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
        // mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        mLastFrame = Frame(mCurrentFrame);
        mLastFrame.SetPose(pKFcur->GetPose());

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->mvpKeyFrameOrigins.push_back(pKFini);
        mpMapDrawer->SetCurrentCameraPose(pKFini->GetPose());

        mState = OK;

    }
}

void Tracking::MonocularInitialization()
{
    if (!mpInitializer)
    {
        // Set Reference Frame
        if (mCurrentFrame.mvKeys.size() > 100)
        {
            mInitialFrame = Frame(mCurrentFrame);

            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

            if (mpInitializer)
                delete mpInitializer;

            mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if ((int)mCurrentFrame.mvKeys.size() <= 100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9, true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,
                                                       mCurrentFrame,
                                                       mvbPrevMatched,
                                                       mvIniMatches,
                                                       100);

        // Check if there are enough correspondences
        if (nmatches < 100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
            return;
        }

        cv::Mat Rcw;                 // Current Camera Rotation
        cv::Mat tcw;                 // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
            {
                if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                {
                    mvIniMatches[i] = -1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
            Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(Tcw.rowRange(0, 3).col(3));
            mCurrentFrame.SetPose(Tcw);

            mInitialSecendFrame = Frame(mCurrentFrame); // [EAO] the second frame when initialization.

            CreateInitialMapMonocular();
        }
    }
}

// note [EAO] modify: rotate the world coordinate to the initial frame.
void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
    KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for (size_t i = 0; i < mvIniMatches.size(); i++)
    {
        if (mvIniMatches[i] < 0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f / medianDepth;

    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
    {
        if (vpAllMapPoints[iMP])
        {
            MapPoint *pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
        }
    }

    {
        // NOTE [EAO] rotate the world coordinate to the initial frame (groundtruth provides the normal vector of the ground).
        // only use the groundtruth of the first frame.
        // TODO: 替换这种方法
        cv::Mat InitToGround = mInitialFrame.mGroundtruthPose_mat;
        // std::cout << InitToGround << std::endl;

        // InitToGround = cv::Mat::eye(4, 4, CV_32F);
        cv::Mat R = InitToGround.rowRange(0, 3).colRange(0, 3);
        cv::Mat t = InitToGround.rowRange(0, 3).col(3);
        cv::Mat Rinv = R.t();
        cv::Mat Ow = -Rinv * t;
        cv::Mat GroundToInit = cv::Mat::eye(4, 4, CV_32F);
        Rinv.copyTo(GroundToInit.rowRange(0, 3).colRange(0, 3));
        Ow.copyTo(GroundToInit.rowRange(0, 3).col(3));

        bool build_worldframe_on_ground = true;
        if (build_worldframe_on_ground) // transform initial pose and map to ground frame
        {
            pKFini->SetPose(pKFini->GetPose() * GroundToInit);
            pKFcur->SetPose(pKFcur->GetPose() * GroundToInit);

            for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
            {
                if (vpAllMapPoints[iMP])
                {
                    MapPoint *pMP = vpAllMapPoints[iMP];
                    pMP->SetWorldPos(InitToGround.rowRange(0, 3).colRange(0, 3) * pMP->GetWorldPos() + InitToGround.rowRange(0, 3).col(3));
                }
            }
        }
        // [EAO] rotate the world coordinate to the initial frame -----------------------------------------------------------
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    mpMap->mvpKeyFrameOrigins.push_back(pKFini);
    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mState = OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for (int i = 0; i < mLastFrame.N; i++)
    {
        MapPoint *pMP = mLastFrame.mvpMapPoints[i];

        if (pMP)
        {
            MapPoint *pRep = pMP->GetReplaced();
            if (pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7, true);
    vector<MapPoint *> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 10)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    // add plane
    mpMap->AssociatePlanesByBoundary(mCurrentFrame);
    // add plane end

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    // add plane
    int nDisgardPlane = 0;
    for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++)
    {
        if (mCurrentFrame.mvpMapPlanes[i])
        {
            if (mCurrentFrame.mvpMapPlanes[i] != nullptr && mCurrentFrame.mvbPlaneOutlier[i])
            {
                mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(NULL);
                nmatches--;
                nDisgardPlane++;
            }
            else
                nmatchesMap++;
        }
    }

    return nmatchesMap >= 10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    // Step 1：利用参考关键帧更新上一帧在世界坐标系下的位姿
    // 上一普通帧的参考关键帧，注意这里用的是参考关键帧（位姿准）而不是上上一帧的普通帧
    KeyFrame *pRef = mLastFrame.mpReferenceKF;
    // ref_keyframe 到 lastframe的位姿变换
    cv::Mat Tlr = mlRelativeFramePoses.back();

    // 将上一帧的世界坐标系下的位姿计算出来
    // l:last, r:reference, w:world
    // Tlw = Tlr*Trw
    mLastFrame.SetPose(Tlr * pRef->GetPose());

    // 如果上一帧为关键帧，或者单目的情况，则退出
    if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR)
        return;
    if ( mSensor == System::RGBD)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    // 注意这些地图点只是用来跟踪，不加入到地图中，跟踪完后会删除
    vector<pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    // Step 2.1：得到上一帧中具有有效深度值的特征点（不一定是地图点）
    for (int i = 0; i < mLastFrame.N; i++)
    {
        float z = mLastFrame.mvDepth[i];
        if (z > 0)
        {
            vDepthIdx.push_back(make_pair(z, i));
        }
    }

    if (vDepthIdx.empty())
        return;
    // std::cout << "vDepthIdx.size() : " <<vDepthIdx.size() << std::endl;

    sort(vDepthIdx.begin(), vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    // Step 2.2：从中找出不是地图点的部分
    // 如果这个点对应在上一帧中的地图点没有,或者创建后就没有被观测到,那么就生成一个临时的地图点
    int nPoints = 0;
    for (size_t j = 0; j < vDepthIdx.size(); j++)
    {
        // std::cout << "j: "<< j  << std::endl;
        // std::cout << vDepthIdx.size() << std::endl;
        int i = vDepthIdx[j].second;
        // std::cout << "i: " << i << std::endl;

        bool bCreateNew = false;

        MapPoint *pMP = mLastFrame.mvpMapPoints[i];
        if (!pMP)
            bCreateNew = true;
        else if (pMP->Observations() < 1)
        {
            bCreateNew = true;
        }

        if (bCreateNew)
        {
            // Step 2.3：需要创建的点，包装为地图点。只是为了提高双目和RGBD的跟踪成功率，并没有添加复杂属性，因为后面会扔掉
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            if (x3D.empty()) continue;
            MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);
            mLastFrame.mvpMapPoints[i] = pNewMP;
            // 标记为临时添加的MapPoint，之后在CreateNewKeyFrame之前会全部删除
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if (vDepthIdx[j].first > mThDepth && nPoints > 100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9, true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);


    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

    // *****************************
    // STEP 1. construct 2D object *
    // *****************************
    vector<Object_2D *> objs_2d;
    cv::Mat image = mCurrentFrame.mColorImage.clone();
    for (auto &box : mCurrentFrame.boxes)
    {
        Object_2D *obj = new Object_2D;

        // copy object bounding box and initialize the 3D center.
        obj->CopyBoxes(box);
        obj->sum_pos_3d = cv::Mat::zeros(3, 1, CV_32F);

        objs_2d.push_back(obj);
    }
    // construct 2D object END ---------------

    // Project points seen in previous frame
    int th;
    if (mSensor != System::STEREO)
        th = 15;
    else
        th = 7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

    // If few matches, uses a wider window search
    if (nmatches < 20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR);
    }

    if (nmatches < 20)
        return false;

    // ***************************************
    // STEP 2. associate objects with points *
    // ***************************************
    AssociateObjAndPoints(objs_2d);



    // ***************************************************
    // STEP 4. compute the mean and standard of points.*
    // Erase outliers (camera frame) by boxplot.*
    // **************************************************
    for (auto &obj : objs_2d)
    {
        // compute the mean and standard.
        obj->ComputeMeanAndStandardFrame();

        // If the object has too few points, ignore.
        if (obj->Obj_c_MapPonits.size() < 8)
            continue;

        // Erase outliers by boxplot.
        obj->RemoveOutliersByBoxPlot(mCurrentFrame);
    }
    // Erase outliers of obj_2d END ----------------------

    // **************************************************************************
    // STEP 5. construct the bounding box by object feature points in the image.*
    // **************************************************************************
    // bounding box detected by yolo |  bounding box constructed by object points.
    //  _______________                 //  _____________
    // |   *        *  |                // |   *        *|
    // |    *  *       |                // |    *  *     |
    // |*      *  *    |                // |*      *  *  |
    // | *   *    *    |                // | *   *    *  |
    // |   *       *   |                // |___*_______*_|
    // |_______________|
    const cv::Mat Rcw = mCurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
    const cv::Mat tcw = mCurrentFrame.mTcw.rowRange(0, 3).col(3);
    for (auto &obj : objs_2d)
    {
        // object 3D center (world).
        obj->_Pos = obj->sum_pos_3d / obj->Obj_c_MapPonits.size();
        obj->mCountMappoint = obj->Obj_c_MapPonits.size(); // point number.
        // world -> camera.
        cv::Mat x3Dc = Rcw * obj->_Pos + tcw;
        const float xc = x3Dc.at<float>(0);
        const float yc = x3Dc.at<float>(1);
        const float invzc = 1.0 / x3Dc.at<float>(2);
        // camera -> image.
        float u = mCurrentFrame.fx * xc * invzc + mCurrentFrame.cx;
        float v = mCurrentFrame.fy * yc * invzc + mCurrentFrame.cy;
        obj->point_center_2d = cv::Point2f(u, v);    // 3D center project to image. no use in this opensource version.

        // record the coordinates of each point in the xy(uv) directions.
        vector<float> x_pt;
        vector<float> y_pt;
        for (auto &pMP : obj->Obj_c_MapPonits)
        {
            float u = pMP->feature.pt.x;
            float v = pMP->feature.pt.y;

            x_pt.push_back(u);
            y_pt.push_back(v);
        }

        if (x_pt.size() < 4) // ignore.
            continue;

        // extremum in xy(uv) direction
        sort(x_pt.begin(), x_pt.end());
        sort(y_pt.begin(), y_pt.end());
        float x_min = x_pt[0];
        float x_max = x_pt[x_pt.size() - 1];
        float y_min = y_pt[0];
        float y_max = y_pt[y_pt.size() - 1];

        // make insure in the image.
        if (x_min < 0)
            x_min = 0;
        if (y_min < 0)
            y_min = 0;
        if (x_max > image.cols)
            x_max = image.cols;
        if (y_max > image.rows)
            y_max = image.rows;

        // the bounding box constructed by object feature points.
        // notes: 视野范围内的特征点
        obj->mRectFeaturePoints = cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min);
    }
    // construct bounding box by feature points END -----------------------------------

    // **********************************************************************************************
    // STEP 6. remove 2d bad bounding boxes.
    // Due to the complex scene and Yolo error detection, some poor quality objects need to be removed.
    // The strategy can be adjusted and is not unique, such as:
    // 1. objects overlap with too many object;
    // 2. objects with too few points;
    // 3. objects with too few points and on the edge of the image;
    // 4. objects too large and take up more than half of the image;
    // and so on ......
    // **********************************************************************************************
    // overlap with too many objects.
    for (size_t f = 0; f < objs_2d.size(); ++f)
    {
        int num = 0;
        for (size_t l = 0; l < objs_2d.size(); ++l)
        {
            if (f == l)
                continue;

            if (Converter::bboxOverlapratioLatter(objs_2d[f]->mBoxRect, objs_2d[l]->mBoxRect) > 0.05)
                num++;
        }
        // overlap with more than 3 objects.
        if (num > 4)
            objs_2d[f]->bad = true;
    }
    for (size_t f = 0; f < objs_2d.size(); ++f)
    {
        if (objs_2d[f]->bad)
            continue;

        // if (conf.model == "multi_classes"){
        //     // ignore the error detect by yolox.
        //     if ((objs_2d[f]->_class_id == 0) || (objs_2d[f]->_class_id == 63) || (objs_2d[f]->_class_id == 15))
        //         objs_2d[f]->bad = true;
        // }

        // too large in the image.
        if ((float)objs_2d[f]->mBoxRect.area() / (float)(image.cols * image.rows) > 0.5)
            objs_2d[f]->bad = true;

        // too few object points.
        if (objs_2d[f]->Obj_c_MapPonits.size() < 5)
            objs_2d[f]->bad = true;

        // object points too few and the object on the edge of the image.
        else if ((objs_2d[f]->Obj_c_MapPonits.size() >= 5) && (objs_2d[f]->Obj_c_MapPonits.size() < 10))
        {
            if ((objs_2d[f]->mBox.x < 20) || (objs_2d[f]->mBox.y < 20) ||
                (objs_2d[f]->mBox.x + objs_2d[f]->mBox.width > image.cols - 20) ||
                (objs_2d[f]->mBox.y + objs_2d[f]->mBox.height > image.rows - 20))
            {
                objs_2d[f]->bad = true;
            }
        }

        // mark the object that on the edge of the image.
        if (((objs_2d[f]->mBox.x < 5) || (objs_2d[f]->mBox.y < 5) ||
            (objs_2d[f]->mBox.x + objs_2d[f]->mBox.width > image.cols - 5) ||
            (objs_2d[f]->mBox.y + objs_2d[f]->mBox.height > image.rows - 5)))
        {
            objs_2d[f]->bOnEdge = true;
        }

        // when the overlap is large, only one object remains.
        for (size_t l = 0; l < objs_2d.size(); ++l)
        {
            if (objs_2d[l]->bad)
                continue;

            if (f == l)
                continue;

            // retain objects which with high probability.
            if (Converter::bboxOverlapratio(objs_2d[f]->mBoxRect, objs_2d[l]->mBoxRect) > 0.3)
            {
                if (objs_2d[f]->mScore < objs_2d[l]->mScore)
                    objs_2d[f]->bad = true;
                else if (objs_2d[f]->mScore >= objs_2d[l]->mScore)
                    objs_2d[l]->bad = true;
            }
            // if one object surrounds another, keep the larger one.
            if (Converter::bboxOverlapratio(objs_2d[f]->mBoxRect, objs_2d[l]->mBoxRect) > 0.05)
            {
                if (Converter::bboxOverlapratioFormer(objs_2d[f]->mBoxRect, objs_2d[l]->mBoxRect) > 0.85)
                    objs_2d[f]->bad = true;
                if (Converter::bboxOverlapratioLatter(objs_2d[f]->mBoxRect, objs_2d[l]->mBoxRect) > 0.85)
                    objs_2d[l]->bad = true;
            }
        }
    }

    // erase the bad object.
    vector<Object_2D *>::iterator it;
    for (it = objs_2d.begin(); it != objs_2d.end(); )
    {
        if ((*it)->bad == true)
            it = objs_2d.erase(it); // erase.
        else
        {
            // if ((*it)->Obj_c_MapPonits.size() >= 5)
            // {
            //     cv::rectangle(image,
            //                     (*it)->mBoxRect,
            //                     cv::Scalar(100, 100, 256),
            //                     2);
            // }

            // cv::putText(image, to_string((*it)->_class_id),
            //             (*it)->box_center_2d,
            //             cv::FONT_HERSHEY_SIMPLEX, 0.5,
            //             cv::Scalar(0, 255, 255), 2);

            // std::string imname_rect = "./box/" + to_string(mCurrentFrame.mTimeStamp) + ".jpg";
            // cv::imwrite(imname_rect, image);

            ++it;
        }
    }
    // remove 2d bad bounding boxes END ------------------------------------------------------

    // *************************************************************
    // STEP 7. copy objects in the last frame after initialization.*
    // *************************************************************
    if ((mbObjectIni == true) && (mCurrentFrame.mnId > mnObjectIniFrameID))
    {
        // copy objects in the last frame.
        mCurrentFrame.mvLastObjectFrame = mLastFrame.mvObjectFrame;

        // copy objects in the penultimate frame.
        if (!mLastFrame.mvLastObjectFrame.empty())
            mCurrentFrame.mvLastLastObjectFrame = mLastFrame.mvLastObjectFrame;
    }
    // copy objects END -------------------------------------------------------


    // *******************************************************************************
    // STEP 8. Merges objects with 5-10 points  between two adjacent frames.
    // Advantage: Small objects with too few points, can be merged to keep them from being eliminated.
    // (The effect is not very significant.)
    // *******************************************************************************
    bool bMergeTwoObj = true;
    if ((!mCurrentFrame.mvLastObjectFrame.empty()) && bMergeTwoObj)
    {
        // object in current frame.
        for (size_t k = 0; k < objs_2d.size(); ++k)
        {
            // ignore objects with more than 10 points.
            if (objs_2d[k]->Obj_c_MapPonits.size() >= 10)
                continue;

            // object in last frame.
            for (size_t l = 0; l < mCurrentFrame.mvLastObjectFrame.size(); ++l)
            {
                // ignore objects with more than 10 points.
                if (mCurrentFrame.mvLastObjectFrame[l]->Obj_c_MapPonits.size() >= 10)
                    continue;

                // merge two objects.
                if (Converter::bboxOverlapratio(objs_2d[k]->mBoxRect, mCurrentFrame.mvLastObjectFrame[l]->mBoxRect) > 0.5)
                {
                    objs_2d[k]->MergeTwoFrameObj(mCurrentFrame.mvLastObjectFrame[l]);
                    break;
                }
            }
        }
    }
    // merge objects in two frame END -----------------------------------------------

    // ************************************
    // STEP 9. Initialize the object map  *
    // ************************************
    // if ((mCurrentFrame.mnId > mInitialSecendFrame.mnId) && mbObjectIni == false)
    //     InitObjMap(objs_2d);
    if ( mbObjectIni == false){
        InitObjMap(objs_2d);
    }

    // **************************************************************
    // STEP 10. Data association after initializing the object map. *
    // **************************************************************
    if ((mCurrentFrame.mnId > mnObjectIniFrameID) && (mbObjectIni == true))
    {

        // step 10.1 points of the object that appeared in the last 30 frames
        // are projected into the image to form a projection bounding box.
        for (int i = 0; i < (int)mpMap->mvObjectMap.size(); i++)
        {
            if (mpMap->mvObjectMap[i]->bBadErase)
                continue;

            // object appeared in the last 30 frames.
            if (mpMap->mvObjectMap[i]->mnLastAddID > mCurrentFrame.mnId - 30){
                // notes: 物体投影到当前图像上
                mpMap->mvObjectMap[i]->ComputeProjectRectFrame(image, mCurrentFrame);
            }
            else
            {
                mpMap->mvObjectMap[i]->mRectProject = cv::Rect(0, 0, 0, 0);
            }
        }

        // step 10.2 data association.
        for (size_t k = 0; k < objs_2d.size(); ++k)
        {
            // ignore object with less than 5 points.
            if (objs_2d[k]->Obj_c_MapPonits.size() < 5)
            {
                objs_2d[k]->few_mappoint = true;
                objs_2d[k]->current = false;
                continue;
            }
            // note: data association.
            objs_2d[k]->ObjectDataAssociation(mpMap, mCurrentFrame, image, mflag);
        }

        // step 10.3 remove objects with too few observations.
        for (int i = (int)mpMap->mvObjectMap.size() - 1; i >= 0; i--)
        {
            if(mflag == "NA")
                continue;

            if (mpMap->mvObjectMap[i]->bBadErase)
                continue;

            int df = (int)mpMap->mvObjectMap[i]->mObjectFrame.size();
            if (df < 10)
            {
                // not been observed in the last 30 frames.
                if (mpMap->mvObjectMap[i]->mnLastAddID < (mCurrentFrame.mnId - 30))
                {
                    if (df < 5)
                        mpMap->mvObjectMap[i]->bBadErase = true;

                    // if not overlap with other objects, don't remove.
                    else
                    {
                        bool overlap = false;
                        for (int j = (int)mpMap->mvObjectMap.size() - 1; j >= 0; j--)
                        {
                            if (mpMap->mvObjectMap[j]->bBadErase || (i == j))
                                continue;

                            if (mpMap->mvObjectMap[i]->WhetherOverlap(mpMap->mvObjectMap[j]))
                            {
                                overlap = true;
                                break;
                            }
                        }
                        if (overlap)
                            mpMap->mvObjectMap[i]->bBadErase = true;
                    }
                }
            }
        }

        // step 10.4 Update the co-view relationship between objects. (appears in the same frame).
        for (int i = (int)mpMap->mvObjectMap.size() - 1; i >= 0; i--)
        {
            if (mpMap->mvObjectMap[i]->mnLastAddID == mCurrentFrame.mnId)
            {
                for (int j = (int)mpMap->mvObjectMap.size() - 1; j >= 0; j--)
                {
                    if (i == j)
                        continue;

                    if (mpMap->mvObjectMap[j]->mnLastAddID == mCurrentFrame.mnId)
                    {
                        int nObjId = mpMap->mvObjectMap[j]->mnId;

                        map<int, int>::iterator sit;
                        sit = mpMap->mvObjectMap[i]->mmAppearSametime.find(nObjId);

                        if (sit != mpMap->mvObjectMap[i]->mmAppearSametime.end())
                        {
                            int sit_sec = sit->second;
                            mpMap->mvObjectMap[i]->mmAppearSametime.erase(nObjId);
                            mpMap->mvObjectMap[i]->mmAppearSametime.insert(make_pair(nObjId, sit_sec + 1));
                        }
                        else
                            mpMap->mvObjectMap[i]->mmAppearSametime.insert(make_pair(nObjId, 1));   // first co-view.
                    }
                }
            }
        }

        // step 10.5 Merge potential associate objects (see mapping thread).

        // step 10.6 Estimate the orientation of objects.
        for (int i = (int)mpMap->mvObjectMap.size() - 1; i >= 0; i--)
        {
            // map object.
            Object_Map* objMap = mpMap->mvObjectMap[i];

            if (objMap->bBadErase)
                continue;

            if (objMap->mnLastAddID < mCurrentFrame.mnId - 5)
                continue;

            // step 10.7 project quadrics to the image (only for visualization).
            // 除以2得到半轴
            cv::Mat axe = cv::Mat::zeros(3, 1, CV_32F);
            axe.at<float>(0) = mpMap->mvObjectMap[i]->mCuboid3D.lenth / 2;
            axe.at<float>(1) = mpMap->mvObjectMap[i]->mCuboid3D.width / 2;
            axe.at<float>(2) = mpMap->mvObjectMap[i]->mCuboid3D.height / 2;

            // object pose (world).
            cv::Mat Twq = Converter::toCvMat(mpMap->mvObjectMap[i]->mCuboid3D.pose);

            // Projection Matrix K[R|t].
            cv::Mat P(3, 4, CV_32F);
            Rcw.copyTo(P.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(P.rowRange(0, 3).col(3));
            P = mCurrentFrame.mK * P;

            // draw.
            image = DrawQuadricProject( this->mCurrentFrame.mQuadricImage,
                                        P,
                                        axe,
                                        Twq,
                                        mpMap->mvObjectMap[i]->mnClass);
        }
    } // data association END ----------------------------------------------------------------

    // add plane
    mpMap->AssociatePlanesByBoundary(mCurrentFrame);
    // add plane end

    // Optimize frame pose with all matches
    // 疑问：在哪里用的优化？
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    // add plane
    int nDisgardPlane = 0;
    for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
        if (mCurrentFrame.mvpMapPlanes[i]) {
            if (mCurrentFrame.mvpMapPlanes[i]!= nullptr && mCurrentFrame.mvbPlaneOutlier[i]) {
                mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(NULL);
                nmatches--;
                nDisgardPlane++;
            } else
                nmatchesMap++;
        }
    }
    // add plane end

    // if (nDisgardPlane > 0)
    if (mbOnlyTracking)
    {
        mbVO = nmatchesMap < 10;
        return nmatches > 20;
    }

//--------------------------------------------------------------------------//
    // // ! bytetrack and localization
    // std::vector<std::shared_ptr<Object2DInstance>> tmp;
    // for (auto &anchor : track_anchors)
    // {
    //     auto current_anchor = std::make_shared<Anchor>();

    //     current_anchor->class_id = anchor.label;
    //     current_anchor->score = anchor.score;
    //     cv::Rect rect(anchor.tlwh[0], anchor.tlwh[1], anchor.tlwh[2], anchor.tlwh[3]);
    //     current_anchor->rect = rect;
    //     current_anchor->track_id = anchor.track_id;

    //     int flag = -1;
    //     for (size_t i = 0; i < object2DMap.size(); ++i)
    //     {
    //         auto obj2d = object2DMap[i];
    //         if (obj2d->id == anchor.track_id)
    //         {
    //             auto current_object2d = std::make_shared<Object2DInstance>();
    //             current_object2d->id = anchor.track_id;
    //             current_object2d->anchor = current_anchor;
    //             current_object2d->class_id = anchor.label;
    //             current_object2d->detect_flag = 0;

    //             current_object2d->history = object2DMap[i]->history;
    //             current_object2d->history.emplace_back(current_anchor);

    //             current_object2d->track_len = object2DMap[i]->track_len + 1;
    //             current_object2d->frames_list = object2DMap[i]->frames_list;
    //             auto Frame = std::make_shared<SimpleFrame>(mCurrentFrame);
    //             current_object2d->frames_list.emplace_back(Frame);

    //             tmp.emplace_back(current_object2d);
    //             flag = 1;
    //             break;
    //         }
    //     }
    //     if (flag == -1){
    //         auto current_object2d = std::make_shared<Object2DInstance>();
    //         current_object2d->id = anchor.track_id;
    //         current_object2d->anchor = current_anchor;
    //         current_object2d->class_id = anchor.label;
    //         current_object2d->track_len = 0;
    //         current_object2d->detect_flag = 0;
    //         // object2DMap.emplace_back(current_object2d);
    //         current_object2d->history.emplace_back(current_anchor);
    //         auto Frame = std::make_shared<SimpleFrame>(mCurrentFrame);
    //         current_object2d->frames_list.emplace_back(Frame);
    //         tmp.emplace_back(current_object2d);
    //     }
    // }

    // object2DMap = tmp;
    // // for (auto &obj : object2DMap)
    // // {
    // //     std::cout << obj->id << ", ";
    // // }
    // // std::cout << std::endl;

    // // std::cout << "[DEBUG] : track_anchors.size() is " << track_anchors.size() << std::endl;
    // // std::cout << "[DEBUG] : object2DMap.size() is " << object2DMap.size() << std::endl;

    // std::vector<std::shared_ptr<ORB_SLAM2::Object3DInstance>> trackingObject3Ds;
    // for (auto &obj3d : object3DMap)
    // {
    //     // 将3d实例投影到2d空间
    //     obj3d->TrackingObject3D(mCurrentFrame);
    //     // 不在视野范围内进行下一轮
    //     if (!obj3d->object2D->anchor->isInImageBoundary(mCurrentFrame.rgb_))
    //         continue;
    //     trackingObject3Ds.emplace_back(obj3d);
    // }

    // // KM算法进行数据关联
    // // std::cout << "[DEBUG] USE KM! " << std::endl;
    // std::vector<int> u_detection, u_track;

    // Object3DInstance::Association3Dto2D(object2DMap, trackingObject3Ds, u_detection, u_track);
    // // std::cout << "[DEBUG] : u_detection.size() is " << u_detection.size() << std::endl;
    // // std::cout << "[DEBUG] : u_track.size() is " << u_track.size() << std::endl;

    // // 遍历未检测对应的观测量，当追踪到达一定次数时候升级为3D实例
    // for (size_t i=0; i<u_detection.size(); ++i){
    //     // std::cout << "[DEBUG] ID: " << u_detection[i] << std::endl;
    //     auto obj2d = object2DMap[u_detection[i]];
    //     if (obj2d->track_len > 3){
    //         // std::cout << "[INFO] New Object3D Build!" << std::endl;
    //         auto obj3d = std::make_shared<Object3DInstance>();
    //         obj3d->id = obj2d->class_id;
    //         obj3d->object2D = obj2d;
    //         if (obj3d->BuildEllipsoid()){
    //             object3DMap.emplace_back(obj3d);
    //             trackingObject3Ds.emplace_back(obj3d);
    //         }
    //     }
    // }


    // debug ------------------------------------------------
    // std::cout << "[DEBUG] : trackingObject3Ds.size() is " << trackingObject3Ds.size() << std::endl;
    // {
    //     cv::Mat showIMG = mCurrentFrame.rgb_.clone();
    //     for (size_t i = 0; i < trackingObject3Ds.size(); ++i)
    //     {
    //         cv::Rect tmp = trackingObject3Ds[i]->object2D->anchor->rect;
    //         cv::rectangle(showIMG, tmp, cv::Scalar(0, 255, 0), 2);
    //         std::string text;
    //         stringstream ss;
    //         ss << trackingObject3Ds[i]->object2D->anchor->class_id;
    //         ss >> text;
    //         int baseLine = 0;
    //         cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
    //         cv::putText(showIMG, text, cv::Point(tmp.x, tmp.y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    //     }

    //     if (trackingObject3Ds.size() > 0)
    //     {
    //         stringstream ss_;
    //         std::string text_;
    //         ss_ << mCurrentFrame.mnId;
    //         ss_ >> text_;
    //         // cv::imwrite("/home/chen/Datasets/tmp/" + text_ + ".png", showIMG);
    //     }
    //     // cv::imshow("debug", showIMG);
    //     // cv::waitKey(0);
    // }
    // std::cout << "[DEBUG INFO END] ——————————————————————————————————-" << std::endl;

    return nmatchesMap >= 10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    // Step 1：更新局部关键帧 mvpLocalKeyFrames 和局部地图点 mvpLocalMapPoints
    UpdateLocalMap();
    // Step 2：筛选局部地图中新增的在视野范围内的地图点，投影到当前帧搜索匹配，得到更多的匹配关系
    SearchLocalPoints();

    // add plane
    mpMap->AssociatePlanesByBoundary(mCurrentFrame);
    // add plane end

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if (!mbOnlyTracking)
                {
                    if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if (mSensor == System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        }
    }

    // add plane
    int nDisgardPlane = 0;
    for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++)
    {
        if (mCurrentFrame.mvpMapPlanes[i])
        {
            if (mCurrentFrame.mvpMapPlanes[i] != nullptr && mCurrentFrame.mvbPlaneOutlier[i])
            {
                mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(NULL);
                nDisgardPlane++;
            }
            else
                mnMatchesInliers++;
        }
    }
    // add plane end

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
        return false;

    if (mnMatchesInliers < 10)
        return false;
    else
        return true;
}

// note [EAO] Modify: Create keyframes by new object.
int Tracking::NeedNewKeyFrame()
{
    // Step 1：纯VO模式下不插入关键帧
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    // Step 2：如果局部地图线程被闭环检测使用，则不插入关键帧
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;
    // 获取当前地图中的关键帧数目
    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    // mCurrentFrame.mnId是当前帧的ID
    // mnLastRelocFrameId是最近一次重定位帧的ID
    // mMaxFrames等于图像输入的帧率
    //  Step 3：如果距离上一次重定位比较近，并且关键帧数目超出最大限制，不插入关键帧
    if( mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    // Step 4：得到参考关键帧跟踪到的地图点数量
    // UpdateLocalKeyFrames 函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧

    // 地图点的最小观测次数
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    // 参考关键帧地图点中观测的数目>= nMinObs的地图点数目
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    // Step 5：查询局部地图线程是否繁忙，当前能否接受新的关键帧
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    // Step 6：对于双目或RGBD摄像头，统计成功跟踪的近点的数量，如果跟踪到的近点太少，没有跟踪到的近点较多，可以插入关键帧
     int nNonTrackedClose = 0;  //双目或RGB-D中没有跟踪到的近点
    int nTrackedClose= 0;       //双目或RGB-D中成功跟踪的近点（三维点）
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            // 深度值在有效范围内
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    // 双目或RGBD情况下：跟踪到的地图点中近点太少 同时 没有跟踪到的三维点太多，可以插入关键帧了
    // 单目时，为false
    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Step 7：决策是否需要插入关键帧
    // Thresholds
    // Step 7.1：设定比例阈值，当前帧和参考关键帧跟踪到点的比例，比例越大，越倾向于增加关键帧
    float thRefRatio = 0.75f;

    // 关键帧只有一帧，那么插入关键帧的阈值设置的低一点，插入频率较低
    if(nKFs<2)
        thRefRatio = 0.4f;

    //单目情况下插入关键帧的频率很高
    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    // Step 7.2：很长时间没有插入关键帧，可以插入
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;

    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    // Step 7.3：满足插入关键帧的最小间隔并且localMapper处于空闲状态，可以插入
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);

    // Condition 1c: tracking is weak
    // Step 7.4：在双目，RGB-D的情况下当前帧跟踪到的点比参考关键帧的0.25倍还少，或者满足bNeedToInsertClose
    const bool c1c =  mSensor!=System::MONOCULAR &&             //只考虑在双目，RGB-D的情况
                    (mnMatchesInliers<nRefMatches*0.25 ||       //当前帧和地图点匹配的数目非常少
                      bNeedToInsertClose) ;                     //需要插入

    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    // Step 7.5：和参考帧相比当前跟踪到的点太少 或者满足bNeedToInsertClose；同时跟踪到的内点还不能太少
    const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) && mnMatchesInliers > 15);

    // note [EAO] create new keyframe by object.
    bool c1d = false;
    if (mCurrentFrame.AppearNewObject)
        c1d = true;

    if ((c1a || c1b || c1c) && c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if (bLocalMappingIdle)
        {
            return 1;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if (mSensor != System::MONOCULAR)
            {
                if (mpLocalMapper->KeyframesInQueue() < 3)
                    return 1;
                else
                    return 0;
            }
            else
                return 0;
        }
    }

    // add plane
    if (mCurrentFrame.mbNewPlane)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if (bLocalMappingIdle)
        {
            return 1;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if (mSensor != System::MONOCULAR)
            {
                if (mpLocalMapper->KeyframesInQueue() < 3)
                    return 1;
                else
                    return 0;
            }
            else
                return 0;
        }
    }

    // note [EAO] create new keyframe by object.
    if (c1d)
    {
        if (bLocalMappingIdle)
        {
            return 2;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if (mSensor != System::MONOCULAR)
            {
                if (mpLocalMapper->KeyframesInQueue() < 3)
                    return 2;
                else
                    return 0;
            }
            else
                return 0;
        }
    }


    return 0;
}

void Tracking::CreateNewKeyFrame(bool CreateByObjs)
{
    if (!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    // save ovjects to keyframe.
    pKF->objects_kf = mCurrentFrame.mvObjectFrame;

    // keyframe created by objects.
    if (CreateByObjs)
        pKF->mbCreatedByObjs = true;

    if (mSensor != System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        if (!CreateByObjs){
            // add plane
            mpMap->AssociatePlanesByBoundary(mCurrentFrame);

            for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i)
            {
                // TODO: 增加观测计数是为了什么？
                if (mCurrentFrame.mvpMapPlanes[i])
                {
                    mCurrentFrame.mvpMapPlanes[i]->AddObservation(pKF, i);
                    if (!mCurrentFrame.mvpMapPlanes[i]->mbSeen)
                    {
                        mCurrentFrame.mvpMapPlanes[i]->mbSeen = true;
                        // 只有在关键帧中才更新平面系数
                        mpMap->AddMapPlane(mCurrentFrame.mvpMapPlanes[i]);
                    }
                    continue;
                }

                if (mCurrentFrame.mvbPlaneOutlier[i])
                    continue;

                // 为地图增加新的平面
                cv::Mat p3D = mCurrentFrame.ComputePlaneWorldCoeff(i);
                MapPlane *pNewMP = new MapPlane(p3D, pKF, i, mpMap);
                mpMap->AddMapPlane(pNewMP);
                pKF->AddMapPlane(pNewMP, i);

            }
        }
        // add plane end

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float, int>> vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
            {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if (!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(), vDepthIdx.end());

            int nPoints = 0;
            for (size_t j = 0; j < vDepthIdx.size(); j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP)
                    bCreateNew = true;
                else if (pMP->Observations() < 1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }

                if (bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap);
                    pNewMP->AddObservation(pKF, i);
                    pKF->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP)
        {
            if (pMP->isBad())
            {
                *vit = static_cast<MapPoint *>(NULL);
            }
            else
            {
                // 更新能观测到该点的帧数加1(被当前帧观测了)
                pMP->IncreaseVisible();
                // 标记该点被当前帧观测到
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                // 标记该点在后面搜索匹配时不被投影，因为已经有匹配了
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch = 0;

    // Project points in frame and check its visibility
    for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if (pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        // 判断地图点是否在在当前帧视野内
        if (mCurrentFrame.isInFrustum(pMP, 0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }
    // Step 3：如果需要进行投影匹配的点的数目大于0，就进行投影匹配，增加更多的匹配关系
    if (nToMatch > 0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if (mSensor == System::RGBD)
            th = 3;
        // If the camera has been relocalised recently, perform a coarser search
        if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            th = 5;
        matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        KeyFrame *pKF = *itKF;
        const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

        for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
        {
            MapPoint *pMP = *itMP;
            if (!pMP)
                continue;
            if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                continue;
            if (!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame *, int> keyframeCounter;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP->isBad())
            {
                const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i] = NULL;
            }
        }
    }

    if (keyframeCounter.empty())
        return;

    int max = 0;
    KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
    {
        KeyFrame *pKF = it->first;

        if (pKF->isBad())
            continue;

        if (it->second > max)
        {
            max = it->second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if (mvpLocalKeyFrames.size() > 80)
            break;

        KeyFrame *pKF = *itKF;

        const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
        {
            KeyFrame *pNeighKF = *itNeighKF;
            if (!pNeighKF->isBad())
            {
                if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame *> spChilds = pKF->GetChilds();
        for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
        {
            KeyFrame *pChildKF = *sit;
            if (!pChildKF->isBad())
            {
                if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame *pParent = pKF->GetParent();
        if (pParent)
        {
            if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }
        }
    }

    if (pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization(int update)
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if (vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75, true);

    vector<PnPsolver *> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint *>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates = 0;

    for (int i = 0; i < nKFs; i++)
    {
        KeyFrame *pKF = vpCandidateKFs[i];
        if (pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            if (nmatches < 15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9, true);

    while (nCandidates > 0 && !bMatch)
    {
        for (int i = 0; i < nKFs; i++)
        {
            if (vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver *pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if (bNoMore)
            {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if (!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint *> sFound;

                const int np = vbInliers.size();

                for (int j = 0; j < np; j++)
                {
                    if (vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j] = NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if (nGood < 10)
                    continue;

                for (int io = 0; io < mCurrentFrame.N; io++)
                    if (mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if (nGood < 50)
                {
                    int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

                    if (nadditional + nGood >= 50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if (nGood > 30 && nGood < 50)
                        {
                            sFound.clear();
                            for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                if (mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

                            // Final optimization
                            if (nGood + nadditional >= 50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for (int io = 0; io < mCurrentFrame.N; io++)
                                    if (mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io] = NULL;
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue
                if (nGood >= 50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if (!bMatch)
    {
        return false;
    }
    else
    {
        if (update == 0)
            mnLastRelocFrameId = mCurrentFrame.mnId;
        // mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}

void Tracking::Reset()
{
    mpViewer->RequestStop();

    cout << "System Reseting" << endl;
    while (!mpViewer->isStopped())
        usleep(3000);

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    KeyFrame::nNextMappingId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if (mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer *>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    mpViewer->Release();
}


void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

// BRIEF [EAO] associate objects with points.
void Tracking::AssociateObjAndPoints(vector<Object_2D *> objs_2d)
{
    const cv::Mat Rcw = mCurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
    const cv::Mat tcw = mCurrentFrame.mTcw.rowRange(0, 3).col(3);

    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

            if (!pMP->isBad())
            {
                for (size_t k = 0; k < objs_2d.size(); ++k)
                {
                    if (objs_2d[k]->mBoxRect.contains(mCurrentFrame.mvKeysUn[i].pt))// in rect.
                    {
                        cv::Mat PointPosWorld = pMP->GetWorldPos();                 // world frame.
                        cv::Mat PointPosCamera = Rcw * PointPosWorld + tcw;         // camera frame.

                        pMP->object_view = true;                  // the point is associated with an object.
                        pMP->frame_id.insert(mCurrentFrame.mnId); // no use.
                        pMP->feature = mCurrentFrame.mvKeysUn[i]; // coordinate in current frame.

                        // object points.
                        objs_2d[k]->Obj_c_MapPonits.push_back(pMP);

                        // summation the position of points.
                        objs_2d[k]->sum_pos_3d += PointPosWorld;
                    }
                }
            }
        }
    }
} // AssociateObjAndPoints() END -----------------------------------

// BRIEF [EAO] Initialize the object map.
void Tracking::InitObjMap(vector<Object_2D *> objs_2d)
{
    // notes：一个obj对应一个objMAP
    int nGoodObjId = -1;        // object id.
    for (auto &obj : objs_2d)
    {
        // Initialize the object map need enough points.
        // 移除人类这个动态障碍物
        // if (conf.model == "multi_classes" && obj->_class_id == 0)
        // {
        //     continue;
        // }
        if (obj->Obj_c_MapPonits.size() < 10)
        {
            obj->few_mappoint = true;
            obj->current = false;
            continue;
        }

        nGoodObjId++;

        mbObjectIni = true;
        mnObjectIniFrameID = mCurrentFrame.mnId;

        // Create an object in the map.
        Object_Map *ObjectMapSingle = new Object_Map;
        ObjectMapSingle->mObjectFrame.push_back(obj);   // 2D objects in each frame associated with this 3D map object.
        ObjectMapSingle->mnId = nGoodObjId;             // 3d objects in the map.
        ObjectMapSingle->mnClass = obj->_class_id;      // object class.
        ObjectMapSingle->mnConfidence = 1;              // object confidence = mObjectFrame.size().
        ObjectMapSingle->mbFirstObserve = true;                 // the object was observed for the first time.
        ObjectMapSingle->mnAddedID = mCurrentFrame.mnId;        // added id.
        ObjectMapSingle->mnLastAddID = mCurrentFrame.mnId;      // last added id.
        ObjectMapSingle->mnLastLastAddID = mCurrentFrame.mnId;  // last last added id.
        ObjectMapSingle->mLastRect = obj->mBoxRect;             // last rect.
        // ObjectMapSingle->mPredictRect = obj->mBoxRect;       // for iou.
        ObjectMapSingle->msFrameId.insert(mCurrentFrame.mnId);  // no used in this version.
        ObjectMapSingle->mSumPointsPos = obj->sum_pos_3d;       // accumulated coordinates of object points.
        ObjectMapSingle->mCenter3D = obj->_Pos;                 // 3d centre.
        obj->mAssMapObjCenter = obj->_Pos;                      // for optimization, no used in this version.

        ObjectMapSingle->mntrack_id = obj->_track_id; // object track id.

        EllipsoidExtractor e;
        cv::Mat depth = mCurrentFrame.mImDepth.clone();
        Eigen::Vector4d bbox;
        bbox << obj->mBoxRect.x, obj->mBoxRect.y, obj->mBoxRect.x + obj->mBoxRect.width, obj->mBoxRect.y + obj->mBoxRect.height;
        Eigen::Matrix4f pose;
        pose.resize(mCurrentFrame.mTcw.rows, mCurrentFrame.mTcw.cols);
        for (int i = 0; i < mCurrentFrame.mTcw.rows; i++)
            for (int j = 0; j < mCurrentFrame.mTcw.cols; j++)
                pose(i, j) = mCurrentFrame.mTcw.at<float>(i, j);

        CameraIntrinsic camera;
        camera.cx = mCurrentFrame.cx;
        camera.cy = mCurrentFrame.cy;
        camera.fx = mCurrentFrame.fx;
        camera.fy = mCurrentFrame.fy;
        camera.scale = mCurrentFrame.mDepthMapFactor;

        pcl::PointCloud<PointType>::Ptr pCloudPCL = e.ExtractPointCloud(depth, bbox, pose, camera);
        PCAResult data = e.ProcessPCA(pCloudPCL);

        e.AdjustChirality(data);
        e.AlignZAxisToGravity(data);
        Eigen::Matrix3d res_pos = (pose.block<3, 3>(0, 0)).cast<double>().transpose() * data.rotMat;
        Eigen::Vector3d euler = data.rotMat.eulerAngles(0, 1, 2);
        double yaw = euler[2];

        // ObjectMapSingle->mCuboid3D.rotY = yaw;

        // add properties of the point and save it to the object.
        for (size_t i = 0; i < obj->Obj_c_MapPonits.size(); i++)
        {
            MapPoint *pMP = obj->Obj_c_MapPonits[i];

            pMP->object_id = ObjectMapSingle->mnId;
            pMP->object_class = ObjectMapSingle->mnClass;
            pMP->track_id = ObjectMapSingle->mntrack_id;
            pMP->object_id_vector.insert(make_pair(ObjectMapSingle->mnId, 1)); // the point is first observed by the object.

            if (ObjectMapSingle->mbFirstObserve == true)
                pMP->First_obj_view = true;

            // save to the object.
            ObjectMapSingle->mvpMapObjectMappoints.push_back(pMP);
        }

        // 2d object.
        obj->mnId = ObjectMapSingle->mnId;
        obj->mnWhichTime = ObjectMapSingle->mnConfidence;
        obj->current = true;

        // save this 2d object to current frame (associates with a 3d object in the map).
        mCurrentFrame.mvObjectFrame.push_back(obj);
        mCurrentFrame.mvLastObjectFrame.push_back(obj);
        //mCurrentFrame.mvLastLastObjectFrame.push_back(obj);

        // updata map.
        ObjectMapSingle->ComputeMeanAndStandard();
        ObjectMapSingle->mCuboid3D.pose = g2o::SE3Quat(data.rotMat, Eigen::Vector3d(0, 0, 1));
        mpMap->mvObjectMap.push_back(ObjectMapSingle);
    }
} // initialize the object map. END -----------------------------------------------------


// BRIEF [EAO] project points to image.
cv::Point2f Tracking::WorldToImg(cv::Mat &PointPosWorld)
{
    // world.
    const cv::Mat Rcw = mCurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
    const cv::Mat tcw = mCurrentFrame.mTcw.rowRange(0, 3).col(3);

    // camera.
    cv::Mat PointPosCamera = Rcw * PointPosWorld + tcw;

    const float xc = PointPosCamera.at<float>(0);
    const float yc = PointPosCamera.at<float>(1);
    const float invzc = 1.0 / PointPosCamera.at<float>(2);

    // image.
    float u = mCurrentFrame.fx * xc * invzc + mCurrentFrame.cx;
    float v = mCurrentFrame.fy * yc * invzc + mCurrentFrame.cy;

    return cv::Point2f(u, v);
} // WorldToImg(cv::Mat &PointPosWorld) END ------------------------------

// BRIEF [EAO] project quadrics from world to image.
cv::Mat Tracking::DrawQuadricProject(cv::Mat &im,
                                     const cv::Mat &P,   // projection matrix.
                                     const cv::Mat &axe, // axis length.
                                     const cv::Mat &Twq, // object pose.
                                     int nClassid,
                                     bool isGT,
                                     int nLatitudeNum,
                                     int nLongitudeNum)
{
    // color.
    std::vector<cv::Scalar> colors = {  cv::Scalar(135,0,248),
                                        cv::Scalar(255,0,253),
                                        cv::Scalar(4,254,119),
                                        cv::Scalar(255,126,1),
                                        cv::Scalar(0,112,255),
                                        cv::Scalar(0,250,250),
                                        };

    // draw params
    // TODO: 改变配色
    cv::Scalar sc = colors[nClassid % 6];

    int nLineWidth = 2;

    // generate angluar grid -> xyz grid (vertical half sphere)
    vector<float> vfAngularLatitude;  // (-90, 90)
    vector<float> vfAngularLongitude; // [0, 180]
    cv::Mat pointGrid(nLatitudeNum + 2, nLongitudeNum + 1, CV_32FC4);

    for (int i = 0; i < nLatitudeNum + 2; i++)
    {
        float fThetaLatitude = -M_PI_2 + i * M_PI / (nLatitudeNum + 1);
        cv::Vec4f *p = pointGrid.ptr<cv::Vec4f>(i);
        for (int j = 0; j < nLongitudeNum + 1; j++)
        {
            float fThetaLongitude = j * M_PI / nLongitudeNum;
            p[j][0] = axe.at<float>(0, 0) * cos(fThetaLatitude) * cos(fThetaLongitude);
            p[j][1] = axe.at<float>(1, 0) * cos(fThetaLatitude) * sin(fThetaLongitude);
            p[j][2] = axe.at<float>(2, 0) * sin(fThetaLatitude);
            p[j][3] = 1.;
        }
    }

    // draw latitude
    for (int i = 0; i < pointGrid.rows; i++)
    {
        cv::Vec4f *p = pointGrid.ptr<cv::Vec4f>(i);
        // [0, 180]
        for (int j = 0; j < pointGrid.cols - 1; j++)
        {
            cv::Mat spherePt0 = (cv::Mat_<float>(4, 1) << p[j][0], p[j][1], p[j][2], p[j][3]);
            cv::Mat spherePt1 = (cv::Mat_<float>(4, 1) << p[j + 1][0], p[j + 1][1], p[j + 1][2], p[j + 1][3]);
            cv::Mat conicPt0 = P * Twq * spherePt0;
            cv::Mat conicPt1 = P * Twq * spherePt1;
            cv::Point pt0(conicPt0.at<float>(0, 0) / conicPt0.at<float>(2, 0), conicPt0.at<float>(1, 0) / conicPt0.at<float>(2, 0));
            cv::Point pt1(conicPt1.at<float>(0, 0) / conicPt1.at<float>(2, 0), conicPt1.at<float>(1, 0) / conicPt1.at<float>(2, 0));
            cv::line(im, pt0, pt1, sc, nLineWidth); // [0, 180]
        }
        // [180, 360]
        for (int j = 0; j < pointGrid.cols - 1; j++)
        {
            cv::Mat spherePt0 = (cv::Mat_<float>(4, 1) << -p[j][0], -p[j][1], p[j][2], p[j][3]);
            cv::Mat spherePt1 = (cv::Mat_<float>(4, 1) << -p[j + 1][0], -p[j + 1][1], p[j + 1][2], p[j + 1][3]);
            cv::Mat conicPt0 = P * Twq * spherePt0;
            cv::Mat conicPt1 = P * Twq * spherePt1;
            cv::Point pt0(conicPt0.at<float>(0, 0) / conicPt0.at<float>(2, 0), conicPt0.at<float>(1, 0) / conicPt0.at<float>(2, 0));
            cv::Point pt1(conicPt1.at<float>(0, 0) / conicPt1.at<float>(2, 0), conicPt1.at<float>(1, 0) / conicPt1.at<float>(2, 0));
            cv::line(im, pt0, pt1, sc, nLineWidth); // [180, 360]
        }
    }

    // draw longitude
    cv::Mat pointGrid_t = pointGrid.t();
    for (int i = 0; i < pointGrid_t.rows; i++)
    {
        cv::Vec4f *p = pointGrid_t.ptr<cv::Vec4f>(i);
        // [0, 180]
        for (int j = 0; j < pointGrid_t.cols - 1; j++)
        {
            cv::Mat spherePt0 = (cv::Mat_<float>(4, 1) << p[j][0], p[j][1], p[j][2], p[j][3]);
            cv::Mat spherePt1 = (cv::Mat_<float>(4, 1) << p[j + 1][0], p[j + 1][1], p[j + 1][2], p[j + 1][3]);
            cv::Mat conicPt0 = P * Twq * spherePt0;
            cv::Mat conicPt1 = P * Twq * spherePt1;
            cv::Point pt0(conicPt0.at<float>(0, 0) / conicPt0.at<float>(2, 0), conicPt0.at<float>(1, 0) / conicPt0.at<float>(2, 0));
            cv::Point pt1(conicPt1.at<float>(0, 0) / conicPt1.at<float>(2, 0), conicPt1.at<float>(1, 0) / conicPt1.at<float>(2, 0));
            cv::line(im, pt0, pt1, sc, nLineWidth); // [0, 180]
        }
        // [180, 360]
        for (int j = 0; j < pointGrid_t.cols - 1; j++)
        {
            cv::Mat spherePt0 = (cv::Mat_<float>(4, 1) << -p[j][0], -p[j][1], p[j][2], p[j][3]);
            cv::Mat spherePt1 = (cv::Mat_<float>(4, 1) << -p[j + 1][0], -p[j + 1][1], p[j + 1][2], p[j + 1][3]);
            cv::Mat conicPt0 = P * Twq * spherePt0;
            cv::Mat conicPt1 = P * Twq * spherePt1;
            cv::Point pt0(conicPt0.at<float>(0, 0) / conicPt0.at<float>(2, 0), conicPt0.at<float>(1, 0) / conicPt0.at<float>(2, 0));
            cv::Point pt1(conicPt1.at<float>(0, 0) / conicPt1.at<float>(2, 0), conicPt1.at<float>(1, 0) / conicPt1.at<float>(2, 0));
            cv::line(im, pt0, pt1, sc, nLineWidth); // [180, 360]
        }
    }

    return im;
} // DrawQuadricProject() END  -----------------------------------------------------------------------------------------------------

void Tracking::LightTrack()
{
    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
    bool useMotionModel = true; // set true

    if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
    {
        cout << "Light Tracking not working because Tracking is not initialized..." << endl;
        return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;
        {
            // Localization Mode:
            if (mState == LOST)
            {
                bOK = Relocalization(1);
            }
            else
            {
                if (!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map
                    if (!mVelocity.empty() && useMotionModel)
                    {
                        bool _bOK = false;
                        bOK = LightTrackWithMotionModel(_bOK); // TODO: check out!!!
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint *> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    bool lightTracking = false;
                    bool bVO = false;
                    if (!mVelocity.empty() && useMotionModel)
                    {
                        lightTracking = true;
                        bOKMM = LightTrackWithMotionModel(bVO); // TODO: check out!!
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization(1);

                    if (bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if ((lightTracking && bVO) || (!lightTracking && mbVO))
                        {
                            for (int i = 0; i < mCurrentFrame.N; i++)
                            {
                                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        if (!bOK)
        {
            if (mpMap->KeyFramesInMap() <= 5)
            {
                cout << "Light Tracking not working..." << endl;
                return;
            }
        }

        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}
bool Tracking::LightTrackWithMotionModel(bool &bVO)
{
    ORBmatcher matcher(0.9, true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    Frame lastFrameBU = mLastFrame;
    list<MapPoint *> lpTemporalPointsBU = mlpTemporalPoints;
    UpdateLastFrame(); // TODO: check out!

    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL)); // TODO:Checkout

    // Project points seen in previous frame
    int th;
    if (mSensor != System::STEREO)
        th = 15;
    else
        th = 7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR); // TODO:Checkout

    // If few matches, uses a wider window search
    if (nmatches < 20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL)); // TODO:Checkout
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR);    // TODO:Checkout
    }

    if (nmatches < 20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }
    mLastFrame = lastFrameBU;
    mlpTemporalPoints = lpTemporalPointsBU;

    bVO = nmatchesMap < 10;
    return nmatches > 20;
}

} // namespace ORB_SLAM2
