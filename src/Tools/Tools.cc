/*
 * @Author: Chen Jiahao
 * @Date: 2022-01-06 19:09:38
 * @LastEditors: Chen Jiahao
 * @LastEditTime: 2022-01-07 17:48:37
 * @Description: file content
 * @FilePath: /catkin_ws/src/EAO-SLAM/src/Tools.cc
 */

#include "Tools.h"

bool Tools::SolveQuadraticEquation(double a, double b, double c, Eigen::Vector2d &x)
{
    x.setZero();
    double delta = b * b - 4 * a * c;
    if (delta < 0)
        return false;
    double x1 = (-b + sqrt(delta)) * 1.0 / (2 * a);
    double x2 = (-b - sqrt(delta)) * 1.0 / (2 * a);
    x(0) = x1;
    x(1) = x2;
    return true;
}

Eigen::Vector3f Tools::RotationMatrix2Euler(Eigen::Matrix3f &rotation)
{
    Eigen::Vector3f eulerAngle = rotation.eulerAngles(0, 1, 2);
    return eulerAngle;
}

double Tools::DistanceBetweenPoints(Eigen::Vector3d p1, Eigen::Vector3d p2){
    double dx = p1(0) - p2(0);
    double dy = p1(1) - p2(1);
    double dz = p1(2) - p2(2);
    return sqrt( dx*dx + dy*dy + dz*dz );
}
