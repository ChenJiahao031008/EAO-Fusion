/*
 * @Author: Chen Jiahao
 * @Date: 2021-12-29 09:09:58
 * @LastEditors: Chen Jiahao
 * @LastEditTime: 2022-01-02 10:48:39
 * @Description: file content
 * @FilePath: /catkin_ws/src/EAO-SLAM/include/ObjectInstance.h
 */


#ifndef OBJECTINSTANCE_H
#define OBJECTINSTANCE_H

#include <opencv2/core/core.hpp>
#include <mutex>
#include <stdio.h>
#include <memory>

#include "Frame.h"
#include "MapPlane.h"
// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
typedef Eigen::Matrix<float, 2, 1> Vector2f;
typedef Eigen::Matrix<float, 4, 1> Vector4f;

namespace ORB_SLAM2
{
    class Frame;
    class MapPlane;

    class Anchor{
    public:
        Anchor();
        bool isInImageBoundary(const cv::Mat &image);
    public:
        int class_id = -1;    // 物体的类别
        float score = 0.0;    // 物体的置信度
        int trackingFlag = 0; // 该目标框是检测还是追踪得到的
        float averageDepth = -1;

        Eigen::Vector2f center2D;
        cv::Rect rect;
    };

    class Ellipsoid
    {
    public:
        Eigen::Matrix4f Q = Eigen::Matrix4f::Zero();
        Eigen::Matrix4f Q_star = Eigen::Matrix4f::Zero();
        Eigen::Matrix<float, 9, 1> minimalVector; // x,y,z,roll,pitch,yaw,a,b,c
        Eigen::Matrix<float, 8, 3> Cube; // 外接立方体包围框
    public:
        Ellipsoid(){};
        ~Ellipsoid(){};
        void QVector2Q_star();
        void Q_star2Q_Matrix();
        void GetCube();
        Eigen::Matrix4f ComputeC_star();
    };


    class Object2DInstance
    {
    public:
        int id = -1;       // 物体锚框的唯一编号
        Anchor anchor;      // 当前观测的锚框

        int lastFrame = -1;          // 上一时刻是否被观测到
        int count = 0;               // 连续观测到的次数
        int visibility = 0;          // 被观测到的总次数

        int correspondingObj3D = -1; // 对应的物体3D实例编号
        int candidateID = -1;        // 待关联id号(记得每次清理！)

        std::vector<std::shared_ptr<Frame>> objInFrames; // 观测到该物体的帧的id
        std::vector<std::vector<cv::Mat>> descriptor;  // 物体外观的描述子
        std::vector<std::vector<cv::Point3f>> features; // 从属于物体的特征点

    public:
        Object2DInstance();
        float ComputeIoU(const Anchor &candidate);
        bool UpdateAnchor(const Anchor &candidate);
        void AddNewObservation(const int &idx, Frame &frame);
        void AddFuzzyObservation();
    };

    class Object3DInstance
    {
    public:
        int id = -1; // 物体锚框的唯一编号

        int class_id = -1;   // 物体的类别
        float score = 0.0;   // 物体的置信度
        std::shared_ptr<Ellipsoid> ellipsoid;       // 物体椭球体的属性
        std::shared_ptr<Object2DInstance> object2D; // 对应的2d实例
        std::shared_ptr<MapPlane> supportedPlane;   // 对应的支撑平面

        std::vector<cv::Point3f> pointsConnected; // 内部包含的关键点
        std::vector<cv::Mat> Descriptor;          // 物体外观的描述子

    public:
        Object3DInstance(){};
    };
}
#endif //OBJECTINSTANCE_H
