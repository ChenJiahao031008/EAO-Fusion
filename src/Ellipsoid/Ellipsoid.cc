#include "Ellipsoid.h"
// For Euclidean Cluster Extraction
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

namespace ORB_SLAM2
{

    EllipsoidExtractor::EllipsoidExtractor()
    {}

    CloudPtr EllipsoidExtractor::getPointCloudInRect(cv::Mat &depth, const Eigen::VectorXd &detect, CameraIntrinsic &camera, double range)
    {
        CloudPtr cloud(new PointCloud);
        int x1 = int(detect(0));
        int y1 = int(detect(1));
        int x2 = int(detect(2));
        int y2 = int(detect(3));

        for (int y = y1; y < y2; y = y + 3)
        {
            for (int x = x1; x < x2; x = x + 3)
            {
                float *ptd = depth.ptr<float>(y);
                float d = ptd[x];

                PointType p;
                p.z = d / camera.scale / 1000.0;
                // std::cerr << p.z << std::endl;

                if (p.z <= 0.1 || p.z > range) // if the depth is valid
                    continue;
                p.x = (x - camera.cx) * p.z / camera.fx;
                p.y = (y - camera.cy) * p.z / camera.fy;
                cloud->push_back(p);
            }
        }
        return cloud;
    }

    void EllipsoidExtractor::DownSamplePointCloud(CloudPtr &cloudIn, CloudPtr &cloudOut, double grid)
    {
        pcl::VoxelGrid<PointType> voxel;
        double gridsize = grid;
        voxel.setLeafSize(gridsize, gridsize, gridsize);
        voxel.setInputCloud(cloudIn);
        voxel.filter(*cloudOut);
    }

    CloudPtr EllipsoidExtractor::ExtractPointCloud(cv::Mat &depth, Eigen::Vector4d &bbox, Eigen::Matrix4f &pose, CameraIntrinsic &camera)
    {
        double depth_range = 6.0f;
        CloudPtr pPoints_local = getPointCloudInRect(depth, bbox, camera, depth_range);

        std::cout << "Num:  " << pPoints_local->points.size() << std::endl;

        if (pPoints_local->size() < 1)
        {
            std::cout << "No enough point cloud(pPoints_local) after sampling. Num:  " << pPoints_local->size() << std::endl;
            return nullptr;
        }
        CloudPtr pPoints_local_downsample(new PointCloud());
        DownSamplePointCloud(pPoints_local, pPoints_local_downsample, 0.01);

        if (pPoints_local_downsample->size() < 1)
        {
            std::cout << "No enough point cloud(pPoints_local_downsample) after sampling. Num:  " << pPoints_local_downsample->size() << std::endl;
            return nullptr;
        }

        // transform to the world coordinate.
        Eigen::Affine3d transform(pose.cast<double>());
        CloudPtr pPoints_sampled(new PointCloud());
        pcl::transformPointCloud(*pPoints_local_downsample, *pPoints_sampled, transform);

        Eigen::Vector3d center;
        bool bCenter = GetCenter(depth, bbox, pose, camera, center);
        if (!bCenter)
        {
            std::cout << "Can't Find Center. Bbox: " << bbox.transpose() << std::endl;
            return nullptr;
        }

        CloudPtr pPointsEuFiltered = ApplyEuclideanFilter(pPoints_sampled, center);

        return pPointsEuFiltered;
    }

    PCAResult EllipsoidExtractor::ProcessPCA(CloudPtr &cloud)
    {
        Eigen::Vector4d pcaCentroid;
        if (cloud->points.size() < 1)
        {
            std::cout << "No enough point cloud(ProcessPCA cloud) after sampling. Num:  " << cloud->size() << std::endl;
        }
        pcl::compute3DCentroid(*cloud, pcaCentroid);

        Eigen::Matrix3d covariance;
        pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(covariance, Eigen::ComputeEigenvectors);
        Eigen::Matrix3d eigenVectorsPCA = eigen_solver.eigenvectors();
        Eigen::Vector3d eigenValuesPCA = eigen_solver.eigenvalues();

        PointType c;
        c.x = pcaCentroid(0);
        c.y = pcaCentroid(1);
        c.z = pcaCentroid(2);

        PCAResult output;
        output.result = true;
        output.center = Eigen::Vector3d(c.x, c.y, c.z);
        output.rotMat = eigenVectorsPCA;
        output.covariance = Eigen::Vector3d(eigenValuesPCA(0), eigenValuesPCA(1), eigenValuesPCA(2));

        return output;
    }

    void EllipsoidExtractor::AlignZAxisToGravity(PCAResult &data)
    {
        // First, get the z axis
        double max_cos_theta_abs = 0;
        bool max_flag_pos;
        int max_id = -1;

        // TODO: 旋转矩阵提出z轴,区分z轴方向是在2还是3
        Eigen::Vector3d z_axis = INIT_POSE.block<3, 1>(0, 2);
        // z_axis = mpPlane->param.head(3).normalized();

        // find which axis in rotMat has minimum angle difference with z_axis
        for (int i = 0; i < 3; i++)
        {
            Eigen::Vector3d axis = data.rotMat.col(i);
            double cos_theta = axis.dot(z_axis); // a*b = |a||b|cos(theta)

            bool flag_pos = cos_theta > 0;
            double cos_theta_abs = std::abs(cos_theta);

            if (cos_theta_abs > max_cos_theta_abs)
            {
                max_cos_theta_abs = cos_theta_abs;
                max_flag_pos = flag_pos;
                max_id = i;
            }
        }

        assert(max_id >= 0 && "Must find a biggest one.");

        // swap the rotMat to get the correct z axis
        Eigen::Matrix3d rotMatSwap;
        Eigen::Vector3d covarianceSwap;
        Eigen::Vector3d z_axis_vec;

        // invert the direction
        if (max_flag_pos)
            z_axis_vec = data.rotMat.col(max_id);
        else
            z_axis_vec = -data.rotMat.col(max_id);
        rotMatSwap.col(1) = z_axis_vec;
        covarianceSwap(1) = data.covariance[max_id];

        // get other two axes.
        int x_axis_id = (max_id + 1) % 3; // next axis.
        rotMatSwap.col(0) = data.rotMat.col(x_axis_id);
        covarianceSwap(0) = data.covariance[x_axis_id];

        int y_axis_id = (max_id + 2) % 3;
        rotMatSwap.col(2) = rotMatSwap.col(2).cross(rotMatSwap.col(0));
        covarianceSwap(2) = data.covariance[y_axis_id];

        data.rotMat = rotMatSwap;
        data.covariance = covarianceSwap;

        return;
    }

    void EllipsoidExtractor::EstimateLocalEllipsoid(cv::Mat &depth, Eigen::Vector4d &bbox, int label, Eigen::Matrix4f &pose, CameraIntrinsic &camera)
    {
        pcl::PointCloud<PointType>::Ptr pCloudPCL = ExtractPointCloud(depth, bbox, pose, camera);
        PCAResult data = ProcessPCA(pCloudPCL);

        AdjustChirality(data);
        AlignZAxisToGravity(data);
    }

    CloudPtr EllipsoidExtractor::ApplyEuclideanFilter(CloudPtr pCloud, Eigen::Vector3d &center)
    {
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(pCloud);

        int point_num = pCloud->size();

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.02);
        ec.setMinClusterSize(100);
        ec.setMaxClusterSize(point_num);

        ec.setSearchMethod(tree);
        ec.setInputCloud(pCloud);
        ec.extract(cluster_indices);

        bool bFindCluster = false;
        pcl::PointCloud<pcl::PointXYZ>::Ptr pFinalPoints;
        int cluster_size = cluster_indices.size();

        // store the point clouds
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_cluster_vector;
        for (auto it = cluster_indices.begin(); it != cluster_indices.end(); it++)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
                cloud_cluster->points.push_back(pCloud->points[*pit]);
            cloud_cluster_vector.push_back(cloud_cluster);
        }

        for (int i = 0; i < cluster_size; i++)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster = cloud_cluster_vector[i];

            if (cluster_size == 1)
            {
                pFinalPoints = cloud_cluster;
                bFindCluster = true;
            }

            double dis = getDistanceFromPointToCloud(center, cloud_cluster);
            bool c2 = false;
            if (dis < 0.5)
                c2 = true;
            if (c2)
            {
                pFinalPoints = cloud_cluster;
                bFindCluster = true;
                break;
            }
        }
        return pFinalPoints;
    }

    double EllipsoidExtractor::getDistanceFromPointToCloud(Eigen::Vector3d &point, pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud)
    {
        double mini_dis = 999999;
        if (pCloud->size() < 1)
            return -1;

        for (auto p : *pCloud)
        {
            Eigen::Vector3d p_(p.x, p.y, p.z);
            double dis = (point - p_).norm();
            if (dis < mini_dis)
                mini_dis = dis;
        }
        return mini_dis;
    }

    bool EllipsoidExtractor::GetCenter(cv::Mat &depth, Eigen::Vector4d &bbox, Eigen::Matrix4f &pose, CameraIntrinsic &camera, Eigen::Vector3d &center)
    {
        double depth_range = 6.0f;
        // get the center of the bounding box
        int x = int((bbox(0) + bbox(2)) / 2.0);
        int y = int((bbox(1) + bbox(3)) / 2.0);

        int point_num = 10; // sample 10 * 10 points
        int x_delta = std::abs(bbox(0) - bbox(2)) / 4.0 / point_num;
        int y_delta = std::abs(bbox(1) - bbox(3)) / 4.0 / point_num;

        CloudPtr pCloud(new PointCloud());
        // PointCloud &cloud = *pCloud;
        for (int x_id = -point_num / 2; x_id < point_num / 2; x_id++)
        {
            for (int y_id = -point_num / 2; y_id < point_num / 2; y_id++)
            {
                int x_ = x + x_id * x_delta;
                int y_ = y + y_id * y_delta;
                float *ptd = depth.ptr<float>(y_);
                float d = ptd[x_];

                PointType p;
                p.z = d / camera.scale / 1000.0;
                // if the depth value is invalid, ignore this point
                if (p.z <= 0.1 || p.z > depth_range)
                    continue;

                p.x = (x_ - camera.cx) * p.z / camera.fx;
                p.y = (y_ - camera.cy) * p.z / camera.fy;
                pCloud->points.push_back(p);
            }
        }

        if (pCloud->size() < 2)
            return false; // we need at least 2 valid points

        Eigen::Vector4d centroid;
        pcl::compute3DCentroid(*pCloud, centroid); // get their centroid

        PointXYZRGB p;
        p.x = centroid[0];
        p.y = centroid[1];
        p.z = centroid[2];

        // transform to the world coordintate
        Eigen::Matrix4d Twc = pose.cast<double>();

        Eigen::Vector4d Xc;
        Xc << p.x, p.y, p.z, 1.0f;

        center = (Twc * Xc).block<3, 1>(0, 0);

        return true;
    }

    void EllipsoidExtractor::AdjustChirality(PCAResult &data)
    {
        data.rotMat.col(2) = data.rotMat.col(0).cross(data.rotMat.col(1));
        return;
    }

}
