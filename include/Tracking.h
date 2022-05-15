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


#ifndef TRACKING_H
#define TRACKING_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"
#include "Converter.h"
#include "Geometry.h"

#include <mutex>

// 深度滤波
#include "JBF.h"
#include "Kernel.h"
#include "Config.h"

// YOLOX
#include "Global.h"

// lst-ot
#include "ObjectInstance.h"

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;
// class YOLOX;
class BYTETrackerImpl;
class Object2DInstance;
class Object3DInstance;

class Tracking
{

public:
    Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase *pKFDB, const string &strSettingPath, const int sensor, const string &flag, const bool SemanticOnline);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp);
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp, const bool bSemanticOnline);
    cv::Mat GrabImageMonocular( const cv::Mat &im,
                                const double &timestamp,
                                const bool bSemanticOnline);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);


    // void SetSemanticer(YOLOX* detector); //yolox

    void SetSemanticer(BYTETrackerImpl *detector);

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal lenght
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);

    // ] associate objects with points.
    void AssociateObjAndPoints(vector<Object_2D *> objs_2d);

    // [EAO] initialize the object map.
    void InitObjMap(vector<Object_2D *> objs_2d);

    // [EAO] project points from world to image.
    cv::Point2f WorldToImg(cv::Mat &PointPosWorld);

    // [EAO] sample object yaw.
    void SampleObjYaw(Object_Map* objMap);

    // [EAO] project quadrics to the image.
    cv::Mat DrawQuadricProject( cv::Mat &im,
                                const cv::Mat &P,
                                const cv::Mat &axe,
                                const cv::Mat &Twq,
                                int nClassid,
                                bool isGT=true,
                                int nLatitudeNum = 7,
                                int nLongitudeNum = 6);

    // dynaslam
    bool LightTrackWithMotionModel(bool &bVO);
    void LightTrack();

public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // demo.
    string mflag;
    int refine_flag;
    // dynaslam:
    DynaSLAM::Geometry mGeometry;
    // YOLOX* Semanticer;
    BYTETrackerImpl *ByteTracker;


    // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImGray;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;
    Frame mInitialSecendFrame;          // [EAO-SLAM] the second frame when initialization.

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset();

    // object detection.
    bool finish_detected;
    std::vector<BoxSE> boxes_tracker;
    bool have_detected;

    // Groundtruth.
    static bool mbReadedGroundtruth;

    bool mbObjectIni = false;          // initialize the object map.
    int mnObjectIniFrameID;
    int mflag2 = 0;

    // 0: no constraint; 1: use ground truth; 2: use imu
    int miConstraintType = 0;
    bool mbSemanticOnline;

    // lst-ot
    std::vector<BYTE_TRACK::STrack> track_anchors;

    // add obj2d
    std::vector<std::shared_ptr<Object2DInstance>> object2DMap;
    std::vector<std::shared_ptr<Object3DInstance>> object3DMap;

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization();
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();

    bool Relocalization(int update=0);

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    int NeedNewKeyFrame();
    void CreateNewKeyFrame(bool CreateByObjs);

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    Initializer* mpInitializer;

    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;

    // System
    System* mpSystem;

    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    //Map
    Map* mpMap;



    //Calibration matrix
    cv::Mat mK;     // 相机内参.
    cv::Mat mDistCoef;
    float mbf;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    Frame mLastFrame;
    Frame* mRGBDTrackingFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    //Motion Model
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    list<MapPoint*> mlpTemporalPoints;

    vector<vector<double>> mGroundtruth_mat;    // camera groundtruth.
};

} //namespace ORB_SLAM

#endif // TRACKING_H
