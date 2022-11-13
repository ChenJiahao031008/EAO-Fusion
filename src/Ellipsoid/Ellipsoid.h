#pragma once

#include "Global.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

typedef pcl::PointXYZ PointType;
typedef pcl::PointXYZRGB PointXYZRGB;
typedef pcl::PointCloud<PointType> PointCloud;
typedef pcl::PointCloud<PointXYZRGB> RGBDCloud;
typedef pcl::PointCloud<PointType>::Ptr CloudPtr;

typedef pcl::Normal NormalType;

namespace ORB_SLAM2
{

    struct PCAResult
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        bool result;
        Eigen::Vector3d center;
        Eigen::Matrix3d rotMat;
        Eigen::Vector3d covariance;
        Eigen::Vector3d scale;
    };

    struct CameraIntrinsic
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        double scale;
        double fx;
        double fy;
        double cx;
        double cy;
    };

    class EllipsoidExtractor
    {

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EllipsoidExtractor();

        CloudPtr getPointCloudInRect(cv::Mat &depth, const Eigen::VectorXd &detect, CameraIntrinsic &camera, double range);

        void DownSamplePointCloud(CloudPtr &cloudIn, CloudPtr &cloudOut, double grid);

        CloudPtr ExtractPointCloud(cv::Mat &depth, Eigen::Vector4d &bbox, Eigen::Matrix4f &pose, CameraIntrinsic &camera);

        PCAResult ProcessPCA(CloudPtr &cloud);

        void EstimateLocalEllipsoid(cv::Mat &depth, Eigen::Vector4d &bbox, int label, Eigen::Matrix4f &pose, CameraIntrinsic &camera);

        void AlignZAxisToGravity(PCAResult &data);

        void ApplyGravityPrior(PCAResult &data);

        CloudPtr ApplyEuclideanFilter(CloudPtr pCloud, Eigen::Vector3d &center);

        bool GetCenter(cv::Mat &depth, Eigen::Vector4d &bbox, Eigen::Matrix4f &pose, CameraIntrinsic &camera, Eigen::Vector3d &center);

        double getDistanceFromPointToCloud(Eigen::Vector3d &point, pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud);

        void AdjustChirality(PCAResult &data);

    private:
        bool mResult; // estimation result.
    };

}
