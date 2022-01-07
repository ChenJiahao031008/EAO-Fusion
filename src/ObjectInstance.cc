/*
 * @Author: Chen Jiahao
 * @Date: 2021-12-29 15:16:12
 * @LastEditors: Chen Jiahao
 * @LastEditTime: 2022-01-07 22:38:53
 * @Description: file content
 * @FilePath: /catkin_ws/src/EAO-SLAM/src/ObjectInstance.cc
 */

#include "ObjectInstance.h"

#include <iostream>
#include <stdint.h>
#include <random>
#include <vector>
#include <string>

namespace ORB_SLAM2
{
// class Object2DInstance ------------------------------------------------------
Object2DInstance::Object2DInstance(){};

float Object2DInstance::ComputeIoU(const Anchor &candidate)
{
    cv::Rect Union = anchor->rect | candidate.rect;
    cv::Rect Intersection = anchor->rect & candidate.rect;

    return Intersection.area() * 1.0 / Union.area();
}

bool Object2DInstance::UpdateAnchor(const Anchor &candidate)
{
    if (1.2 * candidate.score < anchor->score){
        return false;
    }else if (candidate.score > 1.2*anchor->score){
        anchor->score =  candidate.score;
        anchor->rect = candidate.rect;
        // TODO: 更新其他数据(如平均深度等)
    }else{
        float ratio = anchor->score * 1.0 / (anchor->score + candidate.score);
        int newX = ratio * anchor->rect.x + (1 - ratio) * anchor->rect.x;
        int newY = ratio * anchor->rect.y + (1 - ratio) * anchor->rect.y;
        int newWidth = ratio * anchor->rect.width + (1 - ratio) * anchor->rect.width;
        int newHeight = ratio * anchor->rect.height + (1 - ratio) * anchor->rect.height;
        cv::Rect newRect(newX, newY, newWidth, newHeight);
        anchor->rect = newRect;
    }
    return true;
}

void Object2DInstance::AddNewObservation(const int &idx, Frame &frame)
{
    count++;
    candidateID = idx;
    visibility++;
    lastFrame = 1;
    UpdateFrameVec(frame);
}

void Object2DInstance::AddFuzzyObservation(const int &idx)
{
    candidateID = idx;
    visibility++;
    lastFrame = 1;
}

void Object2DInstance::UpdateFrameVec(Frame &frame)
{
    // TODO:change threshold
    if (objInFrames.size() < 5){
        objInFrames.push_back(static_cast<std::shared_ptr<Frame>>(&frame));
        obsInFrames.push_back(anchor);
    }else{
        // TODO:后续计算角度误差,保留视角和共视度较大的区域，而非简单的窗口滑动
        objInFrames.erase(objInFrames.begin());
        objInFrames.push_back(static_cast<std::shared_ptr<Frame>>(&frame));
        obsInFrames.push_back(anchor);
    }
}

bool Object2DInstance::Association2Dto2D(const int &i, Frame &frame, const Anchor &candidate)
{
    // 设置阈值
    const float TH_IOU = 0.7;
    // 计算IoU值大小
    float score = ComputeIoU(candidate);
    // Case1 重叠度很高，认为可能形成关联
    bool IoU_flag = score > TH_IOU;
    // TODO: Case2 描述子统计学比较确定是否关联
    bool descriptor_flag = true;
    // Case3 校核类别和置信度，IoU重合但是类别不同时不再进行计数
    bool class_flag = candidate.class_id == anchor->class_id;

    if (!IoU_flag)
        return false;

    // 仅有类别错误的情况下
    if (IoU_flag && descriptor_flag && !class_flag)
    {
        if (candidateID != -1){
            // 在类别错误下更新观测（不稳定性更新）
            AddFuzzyObservation(i);
            return true;
        }else{
            return false;
        }

    }

    // 全部关联成功
    if (IoU_flag && descriptor_flag && class_flag)
    {
        // 认为发生关联，更新当前锚框，并判断锚框是否更新
        bool update_flag = UpdateAnchor(candidate);
        // 多个观测指向同一个obj2D实例
        if (candidateID != -1){
            // 不更新时直接退出
            if (!update_flag)
                return false;
        }

        // 更新obj2d可见性、计数属性,更新当前观测帧
        AddNewObservation(i, frame);
        // 追踪成功
        return true;
    }

    // 关联失败
    return false;
}

void Object2DInstance::AddFeatures(Frame &frame)
{
    const cv::Mat Rcw = frame.mTcw.rowRange(0, 3).colRange(0, 3);
    const cv::Mat tcw = frame.mTcw.rowRange(0, 3).col(3);
    const cv::Mat Rwc = Rcw.t();
    const cv::Mat twc = -Rcw.t() * tcw;

    float sum_depth = 0.0;
    int vaildCount = 0;
    currentFeatures.clear();

    for (size_t i = 0; i < frame.N; i++)
    {
        if (frame.mvpMapPoints[i])
        {
            auto tmpPT = frame.mvKeysUn[i].pt;
            auto tmpDepth = frame.mvDepth[i];

            bool case1 = !(frame.mvpMapPoints[i]->isBad());
            bool case2 = anchor->rect.contains(tmpPT);
            bool case3 = tmpDepth > 0;

            if (case1 && case2 && case3)
            {
                sum_depth += tmpDepth;
                vaildCount++;
                float tmpMP_x = (tmpPT.x - frame.cx) * 1.0 / frame.fx * tmpDepth;
                float tmpMP_y = (tmpPT.y - frame.cy) * 1.0 / frame.fy * tmpDepth;
                float tmpMP_z = tmpDepth;
                cv::Mat currentMP = (cv::Mat_<float>(3, 1) << tmpMP_x, tmpMP_y, tmpMP_z);
                currentMP = Rwc * currentMP + twc;
                currentFeatures.emplace_back(currentMP);
            }
        }
    }
    anchor->averageDepth = sum_depth / vaildCount;

    RemoveOutliers(frame);

}

void Object2DInstance::RemoveOutliers(Frame &frame)
{
    const cv::Mat Rcw = frame.mTcw.rowRange(0, 3).colRange(0, 3);
    const cv::Mat tcw = frame.mTcw.rowRange(0, 3).col(3);
    // world -> camera.
    std::vector<float> x_c, y_c, z_c;
    for (size_t i = 0; i < currentFeatures.size(); i++)
    {
        cv::Mat PointPosWorld = currentFeatures.at(i);
        cv::Mat PointPosCamera = Rcw * PointPosWorld + tcw;

        x_c.push_back(PointPosCamera.at<float>(0));
        y_c.push_back(PointPosCamera.at<float>(1));
        z_c.push_back(PointPosCamera.at<float>(2));
    }

    // sort.
    sort(x_c.begin(), x_c.end());
    sort(y_c.begin(), y_c.end());
    sort(z_c.begin(), z_c.end());

    // notes: 点的数量需要大于4
    if ((z_c.size() / 4 <= 0) || (z_c.size() * 3 / 4 >= z_c.size() - 1))
        return;
    // notes: 取排序1/4和3/4处的深度并对其进行扩展，获取最大最小阈值
    float Q1 = z_c[(int)(z_c.size() / 4)];
    float Q3 = z_c[(int)(z_c.size() * 3 / 4)];
    float IQR = Q3 - Q1;

    float min_th = Q1 - 1.5 * IQR;
    float max_th = Q3 + 1.5 * IQR;

    std::vector<cv::Mat>::iterator iter = currentFeatures.begin();
    for (; iter != currentFeatures.end(); iter++)
    {
        cv::Mat PointPosWorld = (*iter);
        cv::Mat PointPosCamera = Rcw * PointPosWorld + tcw;

        float z = PointPosCamera.at<float>(2);
        // notes: 排除过远和过近处的3D点
        if (z > max_th || z < min_th)
        {
            iter = currentFeatures.erase(iter); // remove.
            iter--;
        }
    }
}

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

// class Anchor ------------------------------------------------------
Anchor::Anchor(){};

bool Anchor::isInImageBoundary(const cv::Mat &image)
{
    cv::Rect imageBoundary = cv::Rect( 0, 0, image.cols, image.rows);
    return (rect == (rect & imageBoundary));
}

bool Anchor::RemoveSmallSize(){
    const int WidthTH = 30;
    const int HeightTH = 30;
    const int SizeTH = WidthTH * HeightTH;
    if (rect.area() < SizeTH) return false;
    if (rect.width()  < WidthTH) return false;
    if (rect.height() < HeightTH) return false;
    return true;
}

// class Object3DInstance -----------------------------------------------
Object3DInstance::Object3DInstance(){};

bool Object3DInstance::Association3Dto2D(){
    // TODO: 3d-2d 数据关联
    return true;
}

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
    // object2D->anchor = SimpleGenerateAnchor(C);
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
        if (pt(0) < 0 || pt(0) > w || pt(1) < 0 || pt(1) > h)
            continue;
        Eigen::Vector2d pt = P[i];
        if (pt(0) > max_x) max_x = pt(0);
        if (pt(0) < min_x) min_x = pt(0);
        if (pt(1) > max_y) max_y = pt(1);
        if (pt(1) < min_y) min_y = pt(1);
    }

    std::shared_ptr<Anchor> AnchorPtr = std::make_shared<Anchor>();
    cv::Rect tmpRect(min_x, min_y, (max_x - min_x), (max_y - min_y));
    AnchorPtr->rect = tmpRect;
    AnchorPtr->class_id = object2D->class_id;
    AnchorPtr->score = count * 1.0 / visibility;
    AnchorPtr->trackingFlag = 1;

    Eigen::Vector3d CenterPointInCamera =
        Rcw * ellipsoid->minimalVector.block<3, 1>(0, 0) + tcw;
    // 平均深度为椭球体中心投影
    AnchorPtr->averageDepth = CenterPointInCamera(2, 0);

    return std::move(AnchorPtr);
}

std::shared_ptr<Anchor> Object3DInstance::SimpleGenerateAnchor(Eigen::Matrix3d &C)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C); // ascending sort by default
    Eigen::VectorXd eigens = es.eigenvalues();

    // If it is an ellipse, the sign of eigen values must be :  1 1 -1
    // Ref book : Multiple View Geometry in Computer Vision
    int num_pos = int(eigens(0) > 0) + int(eigens(1) > 0) + int(eigens(2) > 0);
    int num_neg = int(eigens(0) < 0) + int(eigens(1) < 0) + int(eigens(2) < 0);

    // matrix to equation coefficients: ax^2+bxy+cy^2+dx+ey+f=0
    double a = C(0, 0);
    double b = C(0, 1) * 2;
    double c = C(1, 1);
    double d = C(0, 2) * 2;
    double e = C(2, 1) * 2;
    double f = C(2, 2);

    // get x_c, y_c, theta, axis1, axis2 from coefficients
    double delta = c * c - 4.0 * a * b;
    double k = (a * f - e * e / 4.0) - pow((2 * a * e - c * d), 2) / (4 * (4 * a * b - c * c));
    double theta = 1 / 2.0 * atan2(b, (a - c));
    double x_c = (b * e - 2 * c * d) / (4 * a * c - b * b);
    double y_c = (b * d - 2 * a * e) / (4 * a * c - b * b);
    double a_2 = 2 * (a * x_c * x_c + c * y_c * y_c + b * x_c * y_c - 1) / (a + c + sqrt((a - c) * (a - c) + b * b));
    double b_2 = 2 * (a * x_c * x_c + c * y_c * y_c + b * x_c * y_c - 1) / (a + c - sqrt((a - c) * (a - c) + b * b));

    double axis1 = sqrt(a_2);
    double axis2 = sqrt(b_2);

    double cos_theta_2 = cos(theta) * cos(theta);
    double sin_theta_2 = 1 - cos_theta_2;

    double x_limit = sqrt(axis1* axis1* cos_theta_2 + axis2* axis2* sin_theta_2);
    double y_limit = sqrt(axis1* axis1* sin_theta_2 + axis2* axis2* cos_theta_2);

    std::shared_ptr<Anchor> AnchorPtr = std::make_shared<Anchor>();

    cv::Rect tmpRect(x_c - x_limit, y_c - y_limit, 2 * x_limit, 2 * y_limit);
    AnchorPtr->rect = tmpRect;
    AnchorPtr->class_id = object2D->class_id;
    AnchorPtr->score = count * 1.0 / visibility;
    AnchorPtr->trackingFlag = 1;

    Eigen::Vector3d CenterPointInCamera =
        Rcw * ellipsoid->minimalVector.block<3, 1>(0, 0) + tcw;
    // 平均深度为椭球体中心投影
    AnchorPtr->averageDepth = CenterPointInCamera(2,0);

    return std::move(AnchorPtr);
}

void Object3DInstance::BuildEllipsoid(Frame &frame)
{
    ellipsoid = std::make_shared<Ellipsoid>();
    // ----------------------------------- //
    // Case1: 包围盒加入约束
    // 1. 去除范围内外点,
    // 2. 聚类，将内点放入点云
    // 3. 设置主方向
    // ----------------------------------- //
    // Case2: 传统算法
    const int lineRows = 4 * object2D->objInFrames.size();
    Eigen::MatrixXd allPlanes = Eigen::Matrix<double, Dynamic, Dynamic>();
    allPlanes.resize(4, lineRows);

    // 1. 将包围框转化为平面, 并投影到世界坐标系下
    Eigen::Matrix3d K = Converter::toEigenVector(object2D->objInFrames[0]->mK);
    for (size_t i=0; i<object2D->objInFrames.size(); ++i)
    {
        std::shared_ptr<Anchor> currentAnchor = object2D->obsInFrames[i];
        Eigen::Matrix4d currentPose = Converter::toEigenVector(object2D->objInFrames[i]->mTcw);
        Eigen::Matrix<double, 3, 4> P = K * currentPose.block<3, 4>(0, 0);
        // 3*4 x 4*4 = 3*4
        allPlanes.block<4, 4>(0, i * 4) = object2D->GeneratePlanesFromLines(P, currentAnchor);
    }

    // 2. 将平面系数转换为求解方程系数形式
    const int tmpCols = object2D->objInFrames.size();
    Eigen::MatrixXd planesCoeffcient = Eigen::Matrix<double, Dynamic, Dynamic>();
    planesCoeffcient.resize(10, tmpCols);
    for (size_t i = 0; i< allPlanes.cols(); ++i){
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

    // 3. SVD求解10个系数
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(planesCoeffcient.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV );
    MatrixXd V = svd.matrixV();
    // SVD求解超定方程，取最后一列
    VectorXd qj_hat = V.col(V.cols() - 1);
    // 得到Q_star
    ellipsoid->Q_star <<
        qj_hat(0), qj_hat(1), qj_hat(2), qj_hat(3),
        qj_hat(1), qj_hat(4), qj_hat(5), qj_hat(6),
        qj_hat(2), qj_hat(5), qj_hat(7), qj_hat(8),
        qj_hat(3), qj_hat(6), qj_hat(8), qj_hat(9);

    // 4. 求解Q、9自由度表达、Cube
    bool flag  = ellipsoid->ComputeQFromQ_star();
    ellipsoid->getEllipsoid_flag = flag;
    if (flag){
        ellipsoid->ComputeMinimalVector();
        ellipsoid->ComputeCube();
    }

}

Eigen::Matrix4d Object3DInstance::SetMainDirection(pcl::PointcloudPtr<PointType>::Ptr &cloud)
{
    // 计算质心
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    ellipsoid->minimalVector.head<3>() = (centroid.head<3>()).cast<double>();

    // 计算协方差
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cloud, centroid, covariance);

    // 计算特征值
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
    Eigen::Matrix4f eigenTransform = Eigen::Matrix4f::Identity();
    eigenTransform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
    eigenTransform.block<3, 1>(0, 3) = centroid.head<3>();

    return eigenTransform.cast<double>();
}

void Object3DInstance::BuildPointcloudPtr()
{
    int cloudPointsNum = 0;
    for (size_t i = 0; i < object2D->objInFrames.size(); ++i){
        std::shared_ptr<Frame> pFrame = object2D->objInFrames[i];
        int N = pFrame->N;
        for (int j = 0; j < N; j++)
        {
            std::shared_ptr<MapPoint> pMP = static_cast<std::shared_ptr<MapPoint>>(&pFrame->mvpMapPoints[i]);
            if (pMP)
            {
                PointType P; // float
                cv::Mat cvMatPoint = pMP->GetWorldPos();
                P.x = cvMatPoint.at<float>(0);
                P.y = cvMatPoint.at<float>(1);
                P.z = cvMatPoint.at<float>(2);
                cloudPtr->points.push_back(P);
            }
        }
        cloudPointsNum += N;
    }
    cloudPtr->height = 1;
    cloudPtr->width = cloudPointsNum;
    cloudPtr->is_dense = false;
}

double Object3DInstance::ComputeIoU3D(std::shared_ptr<Object3DInstance> &candidate){

    //TODO: 用一种近似方法替代精准的IoU算法，计算较为复杂，看后续是否有机会替换
    double IoU3D = 0.0;

    Eigen::Vector3d e1Center3D(ellipsoid->minimalVector(0), ellipsoid->minimalVector(1), ellipsoid->minimalVector(3));
    Eigen::Vector3d e2Center3D(candidate->ellipsoid->minimalVector(0), candidate->ellipsoid->minimalVector(1), candidate->ellipsoid->minimalVector(3));

    // TODO: may be max(2a,2b,2c) is better?
    double e1MaxDist = Tools::DistanceBetweenPoints(
        (ellipsoid->Cube.block<1, 3>(0, 0)).transpose(),
        (ellipsoid->Cube.block<1, 3>(6, 0)).transpose() );
    double e2MaxDist = Tools::DistanceBetweenPoints(
        (candidate->ellipsoid->Cube.block<1, 3>(0, 0)).transpose(),
        (candidate->ellipsoid->Cube.block<1, 3>(6, 0)).transpose() );
    double centerDist = Tools::DistanceBetweenPoints(e1Center3D, e2Center3D);

    if (centerDist > 0.5 * (e1MaxDist + e2MaxDist))
        return 0.0;

    double sumArea = 8 * (ellipsoid->minimalVector(6) * ellipsoid->minimalVector(7) * ellipsoid->minimalVector(8) + candidate->ellipsoid->minimalVector(6) * candidate->ellipsoid->minimalVector(7) * candidate->ellipsoid->minimalVector(8));

    // 计算合并后的点云
    pcl::PointcloudPtr<PointType>::Ptr mergeCloud(new pcl::PointcloudPtr<PointType>());
    *mergeCloud = (*cloudPtr) + (*(candidate->cloudPtr));
    // 计算合并点云的主方向
    Eigen::Matrix4d pose = SetMainDirection(mergeCloud);
    //TODO: 设置最小包围盒
    // 求逆,transform表示从相机坐标系到物体坐标系
    Eigen::Matrix4f tm = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f tm_inv = pose;
    tm.block<3, 3>(0, 0) = (pose.block<3, 3>(0, 0)).transpose();
    tm.block<3, 1>(0, 3) = -1.0f * ((pose.block<3, 3>(0, 0)).transpose()) * (pose.block<3, 1>(0, 3));

    pcl::PointCloud<PointType>::Ptr transformedCloud(new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*mergeCloud, *transformedCloud, tm);

    PointType min_p1, max_p1;
    Eigen::Vector3f c1, c;
    pcl::getMinMax3D(*transformedCloud, min_p1, max_p1);
    c1 = 0.5f * (min_p1.getVector3fMap() + max_p1.getVector3fMap());
    Eigen::Affine3f tm_inv_aff(tm_inv);
    pcl::transformPoint(c1, c, tm_inv_aff);

    Eigen::Vector3f whd, whd1;
    whd1 = max_p1.getVector3fMap() - min_p1.getVector3fMap();
    whd = whd1;

    float sc1 = (whd1(0) + whd1(1) + whd1(2)) / 3; //点云平均尺度，用于设置主方向箭头大小

    const Eigen::Quaternionf bboxQ(tm_inv.block<3, 3>(0, 0));
    const Eigen::Vector3f bboxT(c);
    // 详见： /home/chen/桌面/SLAM/tmp/test/main2.cpp
    // viewer.addCube(bboxT, bboxQ, whd(0), whd(1), whd(2), "bbox");

    //TODO: 计算IoU： 总-实际 / 总

}

// class Ellipsoid -----------------------------------
Ellipsoid::Ellipsoid()
{
    minimalVector.setZero();
    Cube.setZero();
}

bool Ellipsoid::ComputeQFromQ_star()
{
    // cbrt：查找给定数字的立方根, 不明白为什么这样求Q 为什么需要有立方根
    // Matrix4d Q = Q_star.inverse() * cbrt(Q_star.determinant());
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

void Ellipsoid::ComputeMinimalVector(){
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

void Ellipsoid::ComputeCube(){
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
