/*
 * @Author: Chen Jiahao
 * @Date: 2021-10-29 10:08:18
 * @LastEditors: Chen Jiahao
 * @LastEditTime: 2021-12-21 11:51:48
 * @Description: file content
 * @FilePath: /catkin_ws/src/EAO-SLAM/include/Converter.h
 */
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

#ifndef CONVERTER_H
#define CONVERTER_H

// opencv
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include<Eigen/Dense>
#include "types/types_six_dof_expmap.h"
#include "types/types_seven_dof_expmap.h"
// add plane
#include "Plane3D.h"

namespace ORB_SLAM2
{

class Converter
{
public:
    static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

    static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
    static g2o::SE3Quat toSE3Quat(const g2o::Sim3 &gSim3);

    static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
    static cv::Mat toCvMat(const g2o::Sim3 &Sim3);
    static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
    static cv::Mat toCvMat(const Eigen::Matrix3d &m);
    static cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
    static cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);

    static Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);
    static Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);
    static Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);

    static std::vector<float> toQuaternion(const cv::Mat &M);

    // NOTE [EAO-SLAM]
    static Eigen::MatrixXd toEigenMatrixXd(const cv::Mat &cvMat);
    static cv::Mat toCvMatInverse(const cv::Mat &Tcw);
    static cv::Mat toCvMat(const Eigen::Vector3f &PosEigen);
    static Eigen::Vector3f toEigenVector(const cv::Mat &PosMat);

    // NOTE [EAO-SLAM] compute IoU overlap ratio between two rectangles [x y w h]
    static float bboxOverlapratio(const cv::Rect& rect1, const cv::Rect& rect2);
    static float bboxOverlapratioFormer(const cv::Rect& rect1, const cv::Rect& rect2);
    static float bboxOverlapratioLatter(const cv::Rect& rect1, const cv::Rect& rect2);

    // add plane
    static g2o::Plane3D toPlane3D(const cv::Mat &coe);
    static cv::Mat toCvMat(const g2o::Plane3D &plane);
};

}// namespace ORB_SLAM

#endif // CONVERTER_H
