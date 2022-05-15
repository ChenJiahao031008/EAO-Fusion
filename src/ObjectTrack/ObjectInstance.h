#ifndef _OBJECTINSTANCE_H_
#define _OBJECTINSTANCE_H_

#include <opencv2/core/core.hpp>
#include <mutex>
#include <stdio.h>
#include <memory>
#include <vector>

// Eigen
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "bytetrack_impl.h"
#include "Tools.h"

typedef Eigen::Matrix<float, 2, 1> Vector2f;
typedef Eigen::Matrix<float, 4, 1> Vector4f;

namespace ORB_SLAM2
{
class Frame;
class MapPoint;

// ------------------------------------------------------------------------
class SimpleFrame
{
public:
    cv::Mat mK;
    cv::Mat mTcw;
    int N;
    std::vector<MapPoint *> mvpMapPoints;
    SimpleFrame(Frame &frame);
};

// -------------------------------------------------------------------------
class Anchor
{
public:
    Anchor();
    bool isInImageBoundary(const cv::Mat &image);
    bool RemoveSmallSize();
    float ComputeIoU(const Anchor &candidate);

public:
    int class_id = -1;          // 物体的类别
    float score = 0.0;          // 物体的置信度
    int track_id = 0;           // 物体的id
    float averageDepth = -1;    // 平均深度
    cv::Rect rect;

};

// -------------------------------------------------------------------------
class Ellipsoid
{
public:
    Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
    Eigen::Matrix4d Q_star = Eigen::Matrix4d::Zero();
    // x,y,z,roll,pitch,yaw,a,b,c
    Eigen::Matrix<double, 9, 1> minimalVector = Eigen::Matrix<double, 9, 1>::Zero();
    // 外接立方体包围框
    Eigen::Matrix<double, 8, 3> Cube = Eigen::Matrix<double, 8, 3>::Zero();
    // 是否满足椭球体约束（而非仅仅是二次曲面）
    bool getEllipsoid_flag = false;
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

// -------------------------------------------------------------------------
class Object2DInstance
{
public:
    int id = -1;       // 物体锚框的唯一编号
    int class_id = -1; // 当前物体的稳定类别
    std::shared_ptr<Anchor> anchor; // 当前观测的锚框
    int track_len = 0;          // 追踪长度
    int lastFrame = -1;          // 上一时刻是否被观测到
    int detect_flag = -1;
    std::deque<std::shared_ptr<Anchor>> history;
    std::deque<std::shared_ptr<SimpleFrame>> frames_list;

public:
    Object2DInstance();

    Eigen::Matrix4d GeneratePlanesFromLines(Eigen::Matrix<double, 3, 4> &P, std::shared_ptr<Anchor> &currentAnchor);
};
// ---------------------------------------------------------------------

class Object3DInstance
{
public:
    int id = -1; // 物体锚框的唯一编号
    std::shared_ptr<Ellipsoid> ellipsoid;        // 物体椭球体的属性
    std::shared_ptr<Object2DInstance> object2D;  // 对应的2d实例
    // std::vector<cv::Mat> Descriptor;          // 物体外观的描述子

public:
    Object3DInstance();

    void TrackingObject3D(Frame &frame);

    std::shared_ptr<Anchor> BBoxGenerateAnchor(Eigen::Matrix3d &C, Frame &frame);

    static void Association3Dto2D(
    std::vector<std::shared_ptr<Object2DInstance>> &obj2ds, std::vector<std::shared_ptr<Object3DInstance>> &obj3ds,
    std::vector<int> &u_detection,
    std::vector<int> &u_track);

    void Update(std::shared_ptr<Object2DInstance> &obj2ds);

    bool BuildEllipsoid();
};
// ------------------------------------------------------------------------
}
#endif //OBJECTINSTANCE_H
