/*
 * @Author: Chen Jiahao
 * @Date: 2022-01-06 19:10:25
 * @LastEditors: Chen Jiahao
 * @LastEditTime: 2022-01-10 11:05:09
 * @Description: file content
 * @FilePath: /catkin_ws/src/EAO-SLAM/include/Tools.h
 */

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <Eigen/Dense>

class Tools
{
public:
    Tools(){};
    ~Tools(){};

    static bool SolveQuadraticEquation(double a, double b, double c, Eigen::Vector2d &x);

    static Eigen::Vector3f RotationMatrix2Euler(Eigen::Matrix3f &rotation);

    static double DistanceBetweenPoints(Eigen::Vector3d p1, Eigen::Vector3d p2);
};


