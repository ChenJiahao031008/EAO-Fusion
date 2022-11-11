/*
 * @Author: Chen Jiahao
 * @Date: 2021-12-29 15:16:12
 * @LastEditors: Chen Jiahao
 * @LastEditTime: 2022-01-18 17:55:33
 * @Description: file content
 * @FilePath: /EAO-Fusion/src/ObjectInstance.cc
 */

#include "ObjectInstance.h"

#include <iostream>
#include <stdint.h>
#include <random>
#include <vector>
#include <string>

namespace ORB_SLAM2
{

// class SimpleFrame  --------------------------------------------------
SimpleFrame::SimpleFrame(Frame &frame)
    :mK(frame.mK), mTcw(frame.mTcw.clone()), N(frame.N), mvpMapPoints(frame.mvpMapPoints)
{}

// class Anchor ---------------------------------------------------------
Anchor::Anchor(){};

bool Anchor::isInImageBoundary(const cv::Mat &image)
{
    cv::Rect imageBoundary = cv::Rect(0, 0, image.cols, image.rows);
    return (rect == (rect & imageBoundary));
}

bool Anchor::RemoveSmallSize()
{
    const int WidthTH = 20;
    const int HeightTH = 20;
    const int SizeTH = WidthTH * HeightTH;
    if (rect.area() < SizeTH)
        return false;
    if (rect.width < WidthTH)
        return false;
    if (rect.height < HeightTH)
        return false;
    return true;
}

float Anchor::ComputeIoU(const Anchor &candidate)
{
    cv::Rect Union = rect | candidate.rect;
    cv::Rect Intersection = rect & candidate.rect;

    return Intersection.area() * 1.0 / Union.area();
}

// class Object2DInstance -----------------------------------------------
Object2DInstance::Object2DInstance(){}

Eigen::Matrix4d Object2DInstance::GeneratePlanesFromLines(Eigen::Matrix<double, 3, 4> &P, std::shared_ptr<Anchor> &currentAnchor)
{
    Eigen::Vector3d line1(1, 0, -currentAnchor->rect.x);
    Eigen::Vector3d line2(0, 1, -currentAnchor->rect.y);
    Eigen::Vector3d line3(1, 0, -(currentAnchor->rect.x + currentAnchor->rect.width));
    Eigen::Vector3d line4(0, 1, -(currentAnchor->rect.y + currentAnchor->rect.height));

    Eigen::Matrix<double, 3, 4> linesInFrame;
    linesInFrame.setZero();
    linesInFrame.block<3, 1>(0, 0) = line1;
    linesInFrame.block<3, 1>(0, 1) = line2;
    linesInFrame.block<3, 1>(0, 2) = line3;
    linesInFrame.block<3, 1>(0, 3) = line4;

    Eigen::Matrix4d planesForomLines = P.transpose() * linesInFrame;
    return planesForomLines;
}

// class Object3DInstance -----------------------------------------------
Object3DInstance::Object3DInstance(){};

void Object3DInstance::TrackingObject3D(Frame &frame)
{
    // 定义相机射影变换
    Eigen::Matrix3d Rcw = Converter::toEigenMatrixXd(frame.mTcw.rowRange(0, 3).colRange(0, 3));
    Eigen::Vector3d tcw = Converter::toVector3d(frame.mTcw.rowRange(0, 3).col(3));
    Eigen::Matrix4d Tcw = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d K = Converter::toEigenMatrixXd(frame.mK);
    Eigen::Matrix<double, 3, 4> P = K * Tcw.block<3, 4>(0, 0);

    // 投影到二次曲面
    Eigen::Matrix3d C_star = P * ellipsoid->Q_star * P.transpose();
    Eigen::Matrix3d C = C_star.inverse();
    C = C / C(2, 2); // normalize

    // 生成2D锚框, 更新锚框投影
    object2D->anchor = BBoxGenerateAnchor(C, frame);
}

std::shared_ptr<Anchor> Object3DInstance::BBoxGenerateAnchor(Eigen::Matrix3d &C, Frame &frame)
{
    double a = C(0, 0);
    double b = C(0, 1) * 2;
    double c = C(1, 1);
    double d = C(0, 2) * 2;
    double e = C(2, 1) * 2;
    double f = C(2, 2);

    double k1 = c - b * b / (4.0 * a);
    double k2 = e - b * d / (2.0 * a);
    double k3 = f - d * d / (4.0 * a);
    double k4 = a - b * b / (4.0 * c);
    double k5 = d - b * e / (2.0 * c);
    double k6 = f - e * e / (4.0 * c);
    double y_1 = (-k2 + sqrt(k2 * k2 - 4 * k1 * k3)) * 1.0 / (2 * k1);
    double y_2 = (-k2 - sqrt(k2 * k2 - 4 * k1 * k3)) * 1.0 / (2 * k1);
    double y_3 = (-k5 + sqrt(k5 * k5 - 4 * k4 * k6)) * 1.0 / (2 * k4);
    double y_4 = (-k5 - sqrt(k5 * k5 - 4 * k4 * k6)) * 1.0 / (2 * k4);
    double x_1 = -(b * y_1 + d) * 1.0 / (2 * a);
    double x_2 = -(b * y_2 + d) * 1.0 / (2 * a);
    double x_3 = -(b * y_3 + e) * 1.0 / (2 * c);
    double x_4 = -(b * y_4 + e) * 1.0 / (2 * c);

    double w = frame.rgb_.cols;
    double h = frame.rgb_.rows;

    std::vector<Eigen::Vector2d> P;
    P.emplace_back(x_1, y_1);
    P.emplace_back(x_2, y_2);
    P.emplace_back(x_3, y_3);
    P.emplace_back(x_4, y_4);

    Eigen::Vector2d px1, px2, px3, px4, px5, px6, px7, px8;

    bool res_flag_1 = Tools::SolveQuadraticEquation(c, e, f, px1);
    if (res_flag_1)
    {
        P.emplace_back(0.0, px1(0));
        P.emplace_back(0.0, px1(1));
    }

    bool res_flag_2 = Tools::SolveQuadraticEquation(c, (b * w + e), (a * w * w + d * w + f), px2);
    if (res_flag_2)
    {
        P.emplace_back(w, px2(0));
        P.emplace_back(w, px2(1));
    }

    bool res_flag_3 = Tools::SolveQuadraticEquation(a, d, f, px3);
    if (res_flag_3)
    {
        P.emplace_back(px3(0), 0);
        P.emplace_back(px3(1), 0);
    }

    bool res_flag_4 = Tools::SolveQuadraticEquation(a, (b * h + d), (c * h * h + e * h + f), px4);
    if (res_flag_4)
    {
        P.emplace_back(px4(0), h);
        P.emplace_back(px4(1), h);
    }

    double min_x = 99999; double min_y = 99999;
    double max_x = 0; double max_y = 0;
    for (size_t i=0; i<P.size(); ++i){
        Eigen::Vector2d pt = P[i];
        if (pt(0) < 0 || pt(0) > w || pt(1) < 0 || pt(1) > h)
            continue;
        if (pt(0) > max_x) max_x = pt(0);
        if (pt(0) < min_x) min_x = pt(0);
        if (pt(1) > max_y) max_y = pt(1);
        if (pt(1) < min_y) min_y = pt(1);
    }

    std::shared_ptr<Anchor> AnchorPtr = std::make_shared<Anchor>();
    cv::Rect tmpRect(min_x, min_y, (max_x - min_x), (max_y - min_y));
    AnchorPtr->rect = tmpRect;
    AnchorPtr->class_id = object2D->class_id;
    AnchorPtr->track_id = id;

    Eigen::Matrix3d Rcw = Converter::toEigenMatrixXd(frame.mTcw.rowRange(0, 3).colRange(0, 3));
    Eigen::Vector3d tcw = Converter::toVector3d(frame.mTcw.rowRange(0, 3).col(3));

    Eigen::Vector3d CenterPointInCamera =
        Rcw * ellipsoid->minimalVector.block<3, 1>(0, 0) + tcw;
    // 平均深度为椭球体中心投影
    AnchorPtr->averageDepth = CenterPointInCamera(2, 0);

    return std::move(AnchorPtr);
}

void Object3DInstance::Association3Dto2D(
    std::vector<std::shared_ptr<Object2DInstance>> &obj2ds, std::vector<std::shared_ptr<Object3DInstance>> &obj3ds,
    std::vector<int> &u_detection,
    std::vector<int> &u_track)
{
    const int M = obj2ds.size();
    const int N = obj3ds.size();
    std::vector<std::vector<int>> matches;
    std::vector<std::vector<float>> cost_matrix;

    if (N == 0)
    {
        for (int i = 0; i < M; i++)
        {
            u_detection.push_back(i);
        }
        for (int i = 0; i < N; i++)
        {
            u_track.push_back(i);
        }
        return;
    }
    std::cout << "[DEBUG] M is " << M << std::endl;
    std::cout << "[DEBUG] N is " << N << std::endl;

    cost_matrix.resize(M);
    for (size_t i = 0; i < cost_matrix.size(); ++i)
    {
        std::vector<float> tmp(N, 0.0);
        cost_matrix[i] = tmp;
    }
    std::cout << "[DEBUG] INIT COST MATRIX." << std::endl;

    // Step1：根据2d框序列的交并比，建立权重矩阵
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < M; ++j)
        {
            auto iou = obj2ds[j]->anchor->ComputeIoU(*(obj3ds[i]->object2D->anchor));
            cost_matrix[j][i] = iou;
        }
    }
    std::cout << "[DEBUG] COMPUTE COST MATRIX." << std::endl;

    // Step2：KM算法匹配
    std::vector<int> rowsol;
    std::vector<int> colsol;
    if (cost_matrix.empty()) return;
    // float c = BYTE_TRACK::BYTETracker::lapjv(cost_matrix, rowsol, colsol, true, 0.8);
    float c = 0;
    std::cout << "[DEBUG] COMPUTE KM." << std::endl;

    for (int i = 0; i < rowsol.size(); i++)
    {
        if (rowsol[i] >= 0)
        {
            std::vector<int> match;
            match.push_back(i);
            match.push_back(rowsol[i]);
            matches.push_back(match);
        }
        else
        {
            u_detection.push_back(i);
        }
    }

    for (int i = 0; i < colsol.size(); i++)
    {
        if (colsol[i] < 0)
        {
            u_track.push_back(i);
        }
    }

    // 根据匹配进行更新
    for (size_t i = 0; i < matches.size(); ++i)
    {
        int idx_obs = matches[i][0];
        int idx_obj = matches[i][1];
        obj3ds[idx_obj]->Update(obj2ds[idx_obs]);
    }
}

void Object3DInstance::Update(
    std::shared_ptr<Object2DInstance> &obj2ds)
{
    obj2ds->id = id;
    obj2ds->detect_flag = 1;
    // history.emplace_back(obj2ds);
    object2D = obj2ds;
}

bool Object3DInstance::BuildEllipsoid(){
    auto ep = std::make_shared<Ellipsoid>();
    // ----------------------------------- //
    // Case1: 包围盒加入约束
    // 1. 去除范围内外点,
    // 2. 聚类，将内点放入点云
    // 3. 设置主方向

    // ----------------------------------- //
    // Case2: 传统算法
    int tmpCols = object2D->history.size();
    tmpCols = tmpCols > 5 ? tmpCols : 5;
    int lineRows = 4 * tmpCols;

    Eigen::MatrixXd allPlanes = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>();
    allPlanes.resize(4, lineRows);
    allPlanes.setZero();

    // 1. 将包围框转化为平面, 并投影到世界坐标系下
    Eigen::MatrixXd K = Converter::toEigenMatrixXd(object2D->frames_list[0]->mK);
    size_t j=0;
    // std::cout << "[DEBUG] History Size: " << object2D->history.size() << std::endl;
    // std::cout << "[DEBUG] Frame Size: " << object2D->frames_list.size() << std::endl;
    // std::cout << "[DEBUG] tmpCols : " << tmpCols << std::endl;
    // std::cout << "[DEBUG] lineRows : " << lineRows << std::endl;
    for (size_t i=tmpCols-1; i > 0; i--)
    {
        // std::cout << "[DEBUG] i : " << i << std::endl;
        // std::cout << "[DEBUG] j : " << j << std::endl;
        auto currentAnchor = object2D->history[i];
        Eigen::Matrix4d currentPose = Converter::toEigenMatrixXd(object2D->frames_list[i]->mTcw);
        Eigen::Matrix<double, 3, 4> P = K * currentPose.block<3, 4>(0, 0);
        // 3*4 x 4*4 = 3*4
        allPlanes.block<4, 4>(0, j * 4) = object2D->GeneratePlanesFromLines(P, currentAnchor);
        j++;
    }

    // 2. 将平面系数转换为求解方程系数形式
    Eigen::MatrixXd planesCoeffcient = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>();
    planesCoeffcient.resize(10, allPlanes.cols());
    planesCoeffcient.setZero();
    std::cout << "[DEBUG] AllPlane is: " << allPlanes.cols() << std::endl;
    for (size_t i = 0; i < allPlanes.cols(); ++i)
    {
        double pi_1 = allPlanes(0, i);
        double pi_2 = allPlanes(1, i);
        double pi_3 = allPlanes(2, i);
        double pi_4 = allPlanes(3, i);
        Eigen::Matrix<double, 10, 1> planeVector;
        planeVector <<
            pi_1 * pi_1, 2 * pi_1 * pi_2, 2 * pi_1 * pi_3, 2 * pi_1 * pi_4,
            pi_2 * pi_2, 2 * pi_2 * pi_3, 2 * pi_2 * pi_4,
            pi_3 * pi_3, 2 * pi_3 * pi_4,
            pi_4 * pi_4;
        planesCoeffcient.block<10, 1>(0, i) = planeVector;
    }
    // std::cout << "[DEBUG] planesCoeffcient!\n"
    //           << planesCoeffcient << std::endl;

    // 3. SVD求解10个系数
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(planesCoeffcient.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd V = svd.matrixV();
    // SVD求解超定方程，取最后一列
    Eigen::VectorXd qj_hat = V.col(V.cols() - 1);
    // 得到Q_star
    ep->Q_star << qj_hat(0), qj_hat(1), qj_hat(2), qj_hat(3),
        qj_hat(1), qj_hat(4), qj_hat(5), qj_hat(6),
        qj_hat(2), qj_hat(5), qj_hat(7), qj_hat(8),
        qj_hat(3), qj_hat(6), qj_hat(8), qj_hat(9);

    // std::cout << "[DEBUG] Q_star :\n " << ep->Q_star << std::endl;

    // 4. 求解Q、9自由度表达、Cube
    bool flag = ep->ComputeQFromQ_star();
    ep->getEllipsoid_flag = flag;
    if (flag)
    {
        ep->ComputeMinimalVector();
        ep->ComputeCube();
        std::cout << "[DEBUG] Ellipsoid Build Success." << std::endl;
        ellipsoid = ep;
        return true;
    }
    return false;

}

// class Ellipsoid -----------------------------------
Ellipsoid::Ellipsoid()
{
}

bool Ellipsoid::ComputeQFromQ_star()
{
    // TODO: cbrt, 查找给定数字的立方根, 不明白为什么这样求Q 为什么需要有立方根
    // Q = Q_star.inverse() * cbrt(Q_star.determinant());
    Q = Q_star.inverse() * Q_star.determinant();
    // 特征值分解方法
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Q); // ascending order by default
    Eigen::MatrixXd D = es.eigenvalues().asDiagonal();
    Eigen::MatrixXd V = es.eigenvectors();

    eigens = es.eigenvalues();

    // For an ellipsoid, the signs of the eigenvalues must be ---+ or +++-
    int num_pos = int(eigens(0) > 0) + int(eigens(1) > 0) + int(eigens(2) > 0) + int(eigens(3) > 0);
    int num_neg = int(eigens(0) < 0) + int(eigens(1) < 0) + int(eigens(2) < 0) + int(eigens(3) < 0);
    if (!(num_pos == 3 && num_neg == 1) && !(num_pos == 1 && num_neg == 3))
    {
        std::cout << "[WARNNING] Not Ellipsoid : pos/neg  " << num_pos << " / " << num_neg << std::endl;
        return false;
    }

    if (eigens(3) > 0) // - - - +
    {
        Q = -Q; // + + + -
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_2(Q);
        D = es_2.eigenvalues().asDiagonal();
        V = es_2.eigenvectors();
        eigens = es_2.eigenvalues();
    }
    return true;
}

void Ellipsoid::ComputeMinimalVector()
{
    // Solve ellipsoid parameters from matrix Q
    Eigen::Vector3d lambda_mat = eigens.head(3).array().inverse();

    Eigen::Matrix3d Q33 = Q.block(0, 0, 3, 3);

    double k = Q.determinant() / Q33.determinant();

    Eigen::Vector3d value = -k * (lambda_mat);
    Eigen::Vector3d s = value.array().abs().sqrt();

    Eigen::Vector4d t = Q_star.col(3);
    t = t / t(3);
    Eigen::Vector3d translation = t.head(3);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es2(Q33);
    Eigen::MatrixXd D_Q33 = es2.eigenvalues().asDiagonal();
    rot = es2.eigenvectors();
    Eigen::Vector3d rpy = rot.eulerAngles(0, 1, 2);

    // generate ellipsoid
    minimalVector << t(0), t(1), t(2), rpy(0), rpy(1), rpy(2), s(0), s(1), s(2);
}

void Ellipsoid::ComputeCube()
{
    float x = minimalVector(0);
    float y = minimalVector(1);
    float z = minimalVector(2);
    // float r = minimalVector(3);
    // float p = minimalVector(4);
    // float y = minimalVector(5);
    float a = minimalVector(6);
    float b = minimalVector(7);
    float c = minimalVector(8);

    Eigen::Matrix3d R_wo = rot;
    Eigen::Vector3d t_wo(x, y, z);

    Eigen::Vector3d p1 = R_wo * Eigen::Vector3d(-a, -b, -c) + t_wo;
    Eigen::Vector3d p2 = R_wo * Eigen::Vector3d(+a, -b, -c) + t_wo;
    Eigen::Vector3d p3 = R_wo * Eigen::Vector3d(+a, +b, -c) + t_wo;
    Eigen::Vector3d p4 = R_wo * Eigen::Vector3d(-a, +b, -c) + t_wo;
    Eigen::Vector3d p5 = R_wo * Eigen::Vector3d(-a, -b, +c) + t_wo;
    Eigen::Vector3d p6 = R_wo * Eigen::Vector3d(+a, -b, +c) + t_wo;
    Eigen::Vector3d p7 = R_wo * Eigen::Vector3d(+a, +b, +c) + t_wo;
    Eigen::Vector3d p8 = R_wo * Eigen::Vector3d(-a, +b, +c) + t_wo;

    Cube.block<1, 3>(0, 0) = p1.transpose();
    Cube.block<1, 3>(1, 0) = p2.transpose();
    Cube.block<1, 3>(2, 0) = p3.transpose();
    Cube.block<1, 3>(3, 0) = p4.transpose();
    Cube.block<1, 3>(4, 0) = p5.transpose();
    Cube.block<1, 3>(5, 0) = p6.transpose();
    Cube.block<1, 3>(6, 0) = p7.transpose();
    Cube.block<1, 3>(7, 0) = p8.transpose();
}

} // namspace ORB_SLAM2 end
