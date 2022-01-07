/*
 * @Author: Chen Jiahao
 * @Date: 2021-12-29 09:09:58
 * @LastEditors: Chen Jiahao
 * @LastEditTime: 2022-01-07 21:40:28
 * @Description: file content
 * @FilePath: /catkin_ws/src/EAO-SLAM/include/ObjectInstance.h
 */


#ifndef OBJECTINSTANCE_H
#define OBJECTINSTANCE_H

#include <opencv2/core/core.hpp>
#include <mutex>
#include <stdio.h>
#include <memory>
#include <vector>

#include "Frame.h"
#include "MapPlane.h"
#include "MapPoint.h"
#include "Converter.h"
#include "Tools.h"

// Eigen
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Geometry>

typedef Eigen::Matrix<float, 2, 1> Vector2f;
typedef Eigen::Matrix<float, 4, 1> Vector4f;
typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<PointType> PointCloud;

namespace ORB_SLAM2
{
    class Frame;
    class MapPlane;
    class MapPoint;

    class Anchor{
    public:
        Anchor();
        bool isInImageBoundary(const cv::Mat &image);
        bool RemoveSmallSize();

    public:
        int class_id = -1;    // 物体的类别
        float score = 0.0;    // 物体的置信度
        int trackingFlag = 0; // 该目标框是检测还是追踪得到的
        float averageDepth = -1;

        // Eigen::Vector2f center2D;
        cv::Rect rect;
    };

    class Ellipsoid
    {
    public:
        Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
        Eigen::Matrix4d Q_star = Eigen::Matrix4d::Zero();
        Eigen::Matrix<double, 9, 1> minimalVector; // x,y,z,roll,pitch,yaw,a,b,c
        Eigen::Matrix<double, 8, 3> Cube; // 外接立方体包围框
        bool getEllipsoid_flag = false;   // 是否满足椭球体约束（而非仅仅是二次曲面）

    public:
        Ellipsoid();
        bool ComputeQFromQ_star();
        void ComputeMinimalVector();
        void ComputeCube();
        Eigen::Matrix3d GetRotation(){ return rot; };

    private:
        Eigen::VectorXd eigens;
        Eigen::Matrix3d rot;
    };


    class Object2DInstance
    {
    public:
        int id = -1;       // 物体锚框的唯一编号
        int class_id = -1; // 当前物体的稳定类别
        std::shared_ptr<Anchor> anchor; // 当前观测的锚框

        int lastFrame = -1;          // 上一时刻是否被观测到
        int count = 0;               // 连续观测到的次数
        int visibility = 0;          // 被观测到的总次数

        int correspondingObj3D = -1; // 对应的物体3D实例编号
        int candidateID = -1;        // 待关联id号(记得每次清理！)
        int eraseFLAG = 0;

        std::vector<std::shared_ptr<Frame>> objInFrames;  // 观测到该物体的帧的id
        std::vector<std::shared_ptr<Anchor>> obsInFrames; // 每个帧下对应的观测
        std::vector<std::vector<cv::Mat>> features;       // 物体外观的特征点
        std::vector<cv::Mat> currentFeatures;             // 当前的特征点

    public:
        Object2DInstance();

        float ComputeIoU(const Anchor &candidate);

        void AddNewObservation(const int &idx, Frame &frame);
        void AddFuzzyObservation(const int &idx);
        void AddFeatures(Frame &frame);

        bool UpdateAnchor(const Anchor &candidate);
        void UpdateFrameVec(Frame &frame);

        bool Association2Dto2D(const int &i, Frame &frame, const Anchor &candidate);
        void RemoveOutliers(Frame &frame);

        Eigen::Matrix4d GeneratePlanesFromLines(Eigen::Matrix<double, 3, 4> &P, std::shared_ptr<Anchor> &currentAnchor);
    };

    class Object3DInstance
    {
    public:
        int id = -1; // 物体锚框的唯一编号

        std::shared_ptr<Ellipsoid> ellipsoid;       // 物体椭球体的属性
        std::shared_ptr<Object2DInstance> object2D; // 对应的2d实例
        std::shared_ptr<MapPlane> supportedPlane;   // 对应的支撑平面

        pcl::PointCloud<PointType>::Ptr cloudPtr(new pcl::PointCloud<PointType>()); // 内部包含的关键点
        // std::vector<cv::Mat> Descriptor;          // 物体外观的描述子

    public:
        Object3DInstance();

        bool Association3Dto2D();

        void TrackingObject3D(Frame &frame);

        std::shared_ptr<Anchor> SimpleGenerateAnchor(Eigen::Matrix3d &C);

        std::shared_ptr<Anchor> BBoxGenerateAnchor(Eigen::Matrix3d &C, Frame &frame);

        void BuildEllipsoid(Frame &frame);

        void BuildPointCloud();

        double ComputeIoU3D(std::shared_ptr<Object3DInstance> &candidate);

        Eigen::Matrix4d SetMainDirection(pcl::PointcloudPtr<PointType>::Ptr &cloud);
    };
}
#endif //OBJECTINSTANCE_H
