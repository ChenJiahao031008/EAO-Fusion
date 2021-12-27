/**
* This file is part of ORB-SLAM2.
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* Modification: EAO-SLAM
* Version: 1.0
* Created: 05/21/2019
* Author: Yanmin Wu
* E-mail: wuyanminmax@gmail.com
*/
#include "Global.h"

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    : mpORBvocabulary(frame.mpORBvocabulary),
      mpORBextractorLeft(frame.mpORBextractorLeft),
      mpORBextractorRight(frame.mpORBextractorRight),
      mTimeStamp(frame.mTimeStamp),
      mK(frame.mK.clone()),
      mDistCoef(frame.mDistCoef.clone()),
      im_(frame.im_.clone()),   //new add.
      rgb_(frame.rgb_.clone()), //new add.
      mbf(frame.mbf),
      mb(frame.mb),
      mThDepth(frame.mThDepth),
      N(frame.N),
      mColorImage(frame.mColorImage.clone()),     // color image.
      mQuadricImage(frame.mQuadricImage.clone()), // quadrics image.
      boxes(frame.boxes),
      boxes_eigen(frame.boxes_eigen),
      mvKeys(frame.mvKeys),
      mvKeysRight(frame.mvKeysRight),
      mvKeysUn(frame.mvKeysUn),
      mvuRight(frame.mvuRight),
      mvDepth(frame.mvDepth),
      mBowVec(frame.mBowVec),
      mFeatVec(frame.mFeatVec),
      mDescriptors(frame.mDescriptors.clone()),
      mDescriptorsRight(frame.mDescriptorsRight.clone()),
      mvpMapPoints(frame.mvpMapPoints),
      mvbOutlier(frame.mvbOutlier),
      mnId(frame.mnId),
      mpReferenceKF(frame.mpReferenceKF),
      mnScaleLevels(frame.mnScaleLevels),
      mfScaleFactor(frame.mfScaleFactor),
      mfLogScaleFactor(frame.mfLogScaleFactor),
      mvScaleFactors(frame.mvScaleFactors),
      mvInvScaleFactors(frame.mvInvScaleFactors),
      mvLevelSigma2(frame.mvLevelSigma2),
      mvInvLevelSigma2(frame.mvInvLevelSigma2),
      //add plane
      mvPlanePoints(frame.mvPlanePoints), //add plane
      mvPlaneCoefficients(frame.mvPlaneCoefficients),
      mbNewPlane(frame.mbNewPlane),
      mvpMapPlanes(frame.mvpMapPlanes),
      mnPlaneNum(frame.mnPlaneNum),
      mvbPlaneOutlier(frame.mvbPlaneOutlier),
      mnRealPlaneNum(frame.mnRealPlaneNum),
      mvBoundaryPoints(frame.mvBoundaryPoints)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);

    // [EAO-SLAM] save groundtruth
    if(!frame.mGroundtruthPose_mat.empty())
    {
        mGroundtruthPose_mat = frame.mGroundtruthPose_mat;
        mGroundtruthPose_eigen = Eigen::Matrix4d::Zero(4, 4);
        // mGroundtruthPose_eigen = frame.mGroundtruthPose_eigen;
    }
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    if(mvKeys.empty())
        return;

    N = mvKeys.size();

    UndistortKeyPoints();

    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

//rgbd
Frame::Frame(const cv::Mat &rawImage, // color image.
             const cv::Mat &imGray,
             const cv::Mat &imDepth,
             const double &timeStamp,
             ORBextractor *extractor,
             ORBVocabulary *voc,
             cv::Mat &K,
             cv::Mat &distCoef,
             const float &bf,
             const float &thDepth,
             cv::Mat &grayimg,
             cv::Mat &rgbimg)
    : mpORBvocabulary(voc),
      mpORBextractorLeft(extractor),
      mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
      mTimeStamp(timeStamp),
      mColorImage(rawImage.clone()),   // color image.
      mQuadricImage(rawImage.clone()), // quadrics image.
      mK(K.clone()),
      mDistCoef(distCoef.clone()),
      im_(grayimg.clone()),
      rgb_(rgbimg.clone()),
      mbf(bf),
      mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double t12 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "[COST TIME] Point Extraction Time is : " << t12 << std::endl;


    // add plane --------------------------
    // 由于特征提取时间和面特征提取时间加在一起都没有目标检测时间耗时大，因此这里并没有采用并行处理的方法
    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();
    // case1 :
    // ComputePlanesFromOrganizedPointCloud(imDepth);
    // case2:
    ComputePlanesFromPEAC(imDepth);
    std::chrono::steady_clock::time_point t6 = std::chrono::steady_clock::now();
    double t56 = std::chrono::duration_cast<std::chrono::duration<double>>(t6 - t5).count();
    std::cout << "[COST TIME] Plane Extraction Time is : " << t56 << std::endl;

    double t16 = std::chrono::duration_cast<std::chrono::duration<double>>(t6 - t1).count();
    std::cout << "[COST TIME] Total Extraction Time is : " << t16 << std::endl;

    mnRealPlaneNum = mvPlanePoints.size();
    mnPlaneNum = mvPlanePoints.size();
    // std::cout << "[INFO] Plane Num is : " << mnPlaneNum << std::endl;
    mvpMapPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
    mvbPlaneOutlier = vector<bool>(mnPlaneNum, false);
}

// add plane -----------------------------

void Frame::ComputePlanesFromOrganizedPointCloud(const cv::Mat &imDepth)
{
    PointCloud::Ptr inputCloud(new PointCloud());

    //TODO: 参数传递
    int cloudDis = 2;
    int min_plane = 500;
    float AngTh = 3.0;
    float DisTh = 0.05;

    // 间隔cloudDis进行采样
    for (int m = 0; m < imDepth.rows; m += cloudDis)
    {
        for (int n = 0; n < imDepth.cols; n += cloudDis)
        {
            float d = imDepth.ptr<float>(m)[n];
            PointT p;
            p.z = d;
            p.x = (n - cx) * p.z / fx;
            p.y = (m - cy) * p.z / fy;
            p.r = 0;
            p.g = 0;
            p.b = 250;

            inputCloud->points.push_back(p);
        }
    }
    inputCloud->height = ceil(imDepth.rows / float(cloudDis));
    inputCloud->width = ceil(imDepth.cols / float(cloudDis));

    //估计法线
    pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(0.05f);
    ne.setNormalSmoothingSize(10.0f);
    ne.setInputCloud(inputCloud);
    //计算特征值
    ne.compute(*cloud_normals);

    vector<pcl::ModelCoefficients> coefficients;
    vector<pcl::PointIndices> inliers;
    pcl::PointCloud<pcl::Label>::Ptr labels(new pcl::PointCloud<pcl::Label>);
    vector<pcl::PointIndices> label_indices;
    vector<pcl::PointIndices> boundary;

    pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
    mps.setMinInliers(min_plane);
    mps.setAngularThreshold(0.017453 * AngTh);
    mps.setDistanceThreshold(DisTh);
    mps.setInputNormals(cloud_normals);
    mps.setInputCloud(inputCloud);
    // 该方法能够一次性提取几个面
    std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT>>> regions;
    mps.segmentAndRefine(regions, coefficients, inliers, labels, label_indices, boundary);

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(inputCloud);
    extract.setNegative(false);

    // srand(time(0));
    // 每次提取获得: 1）平面的系数Mat（mvPlaneCoefficients）
    //             2) 平面的点云（mvPlanePoints）
    //             3）面边界上的点云（mvBoundaryPoints）
    for (int i = 0; i < inliers.size(); ++i)
    {
        PointCloud::Ptr planeCloud(new PointCloud());
        cv::Mat coef = (cv::Mat_<float>(4, 1) << coefficients[i].values[0],
                        coefficients[i].values[1],
                        coefficients[i].values[2],
                        coefficients[i].values[3]);
        // 要求距离d大于0
        if (coef.at<float>(3) < 0)
            coef = -coef;

        if (!PlaneNotSeen(coef))
        {
            continue;
        }
        extract.setIndices(boost::make_shared<pcl::PointIndices>(inliers[i]));
        extract.filter(*planeCloud);

        mvPlanePoints.push_back(*planeCloud);

        PointCloud::Ptr boundaryPoints(new PointCloud());
        // 获得平面的边界点
        boundaryPoints->points = regions[i].getContour();
        mvBoundaryPoints.push_back(*boundaryPoints);
        mvPlaneCoefficients.push_back(coef);
    }
}

bool Frame::PlaneNotSeen(const cv::Mat &coef)
{
    // 现有的平面实例集mvPlaneCoefficients
    for (int j = 0; j < mvPlaneCoefficients.size(); ++j)
    {
        cv::Mat pM = mvPlaneCoefficients[j];
        // 两个平面的距离d和夹角 $ a\cdot b = |a||b| \cos\theta = \cos\theta $
        float d = pM.at<float>(3, 0) - coef.at<float>(3, 0);
        float angle = pM.at<float>(0, 0) * coef.at<float>(0, 0) +
                      pM.at<float>(1, 0) * coef.at<float>(1, 0) +
                      pM.at<float>(2, 0) * coef.at<float>(2, 0);
        // 判断平面实例是否和当前观测平面平行或者重叠
        // 1. 两个平面间距过大
        if (d > 0.2 || d < -0.2)
            continue;
        // 2. 夹角处于[20，160] or [-160, -20]范围时：夹角大于一定角度
        if (angle < 0.9397 && angle > -0.9397)
            continue;
        return false;
    }

    return true;
}

cv::Mat Frame::ComputePlaneWorldCoeff(const int &idx)
{
    cv::Mat temp;
    // 注意这里是 mTwc -> mTcw -> tmp: 相当于先求逆再转置，符合平面转换公式
    cv::transpose(mTcw, temp);
    return temp * mvPlaneCoefficients[idx];
}

void Frame::ComputePlanesFromPEAC(const cv::Mat &imDepth)
{
    int cloudDis = 1;
    int vertex_idx = 0;

    // 间隔cloudDis进行采样
    cloud.vertices.resize(imDepth.rows * imDepth.cols);
    cloud.w = ceil(imDepth.cols / float(cloudDis));
    cloud.h = ceil(imDepth.rows / float(cloudDis));

    for (int m = 0; m < imDepth.rows; m += cloudDis)
    {
        for (int n = 0; n < imDepth.cols; n += cloudDis, vertex_idx++)
        {
            double d = (double)(imDepth.ptr<float>(m)[n]);
            if (_isnan(d) ||  d > 4.0 || d < 0.2 )
            {
                // cloud.vertices[vertex_idx++] = Eigen::Vector3d(0, 0, d);
                continue;
            }
            double x = ((double)n - cx) * d / fx;
            double y = ((double)m - cy) * d / fy;
            cloud.vertices[vertex_idx] = Eigen::Vector3d(x, y, d);
        }
    }

    seg_img_ = cv::Mat(imDepth.rows, imDepth.cols, CV_8UC3);
    plane_filter.run(&cloud, &plane_vertices_, &seg_img_);

    plane_num_ = (int)plane_vertices_.size();
    for (int i = 0; i < plane_num_; i++)
    {
        auto &indices = plane_vertices_[i];
        // 遍历每平面上的点云
        PointCloud::Ptr inputCloud(new PointCloud());
        for (int j : indices)
        {
            PointT p;
            p.x = (float)cloud.vertices[j][0];
            p.y = (float)cloud.vertices[j][1];
            p.z = (float)cloud.vertices[j][2];
            // 插入点云
            inputCloud->points.push_back(p);
        }
        auto extractedPlane = plane_filter.extractedPlanes[i];
        double nx = extractedPlane->normal[0];
        double ny = extractedPlane->normal[1];
        double nz = extractedPlane->normal[2];
        double cx = extractedPlane->center[0];
        double cy = extractedPlane->center[1];
        double cz = extractedPlane->center[2];

        float d = (float)-(nx * cx + ny * cy + nz * cz);

        pcl::VoxelGrid<PointT> voxel;
        voxel.setLeafSize(0.05, 0.05, 0.05);
        PointCloud::Ptr coarseCloud(new PointCloud());
        voxel.setInputCloud(inputCloud);
        voxel.filter(*coarseCloud);

        cv::Mat coef = (cv::Mat_<float>(4, 1) << nx, ny, nz, d);

        // 要求距离d大于0
        if (coef.at<float>(3) < 0)
            coef = -coef;

        if (!PlaneNotSeen(coef))
        {
            continue;
        }

        //  将滤波后的点云放置到mvPlanePoints中，在我的理解可能是以此来代表平面的大小
        mvBoundaryPoints.push_back(*coarseCloud);
        mvPlanePoints.push_back(*coarseCloud);
        mvPlaneCoefficients.push_back(coef);
    }
    cloud.vertices.clear();
    seg_img_.release();
    color_img_.release();
}

// add plane end -----------------------------

Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}


Frame::Frame(   const cv::Mat &rawImage,                // color image.
                const cv::Mat &imGray,
                const double &timeStamp,
                ORBextractor* extractor,
                ORBVocabulary* voc,
                cv::Mat &K,
                cv::Mat &distCoef,
                const float &bf,
                const float &thDepth,
                cv::Mat &grayimg,
                cv::Mat &rgbimg)
                            :   mpORBvocabulary(voc),
                                mpORBextractorLeft(extractor),
                                mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
                                mTimeStamp(timeStamp),
                                mColorImage(rawImage.clone()),      // color image.
                                mQuadricImage(rawImage.clone()),    // quadrics image.
                                mK(K.clone()),
                                mDistCoef(distCoef.clone()),
                                im_(grayimg.clone()),
                                rgb_(rgbimg.clone()),
                                mbf(bf),
                                mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();


}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,mfLogScaleFactor);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

        // make sure it is inside image
        mnMinX = max(mnMinX,0.0f);
        mnMaxX = min(mnMaxX,(float)imLeft.cols);
        mnMinY = max(mnMinY,0.0f);
        mnMaxY = min(mnMaxY,(float)imLeft.rows);

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = -3;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<ORBmatcher::TH_HIGH)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=0 && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
