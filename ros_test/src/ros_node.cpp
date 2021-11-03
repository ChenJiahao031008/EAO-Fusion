#include "ros/ros.h"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.h"
#include "geometry_msgs/Twist.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ros/package.h>
#include <opencv2/core/core.hpp>
#include <System.h>

// 多线程相关
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "Global.h"

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,sensor_msgs::Image> SyncPolicy;

using namespace std;
using namespace cv;

std::string WORK_SPACE_PATH = "";

class ImageConverter
{
public:
    ros::NodeHandle nh_; // 节点名
    image_transport::ImageTransport it_; // 图像转换

    message_filters::Subscriber<sensor_msgs::Image>* image_sub_color; // 彩色图像接受
    message_filters::Subscriber<sensor_msgs::Image>* image_sub_depth; // 深度图像接受
    message_filters::Synchronizer<SyncPolicy>* sync_; // 信息同步器

    std::shared_ptr<ORB_SLAM2::System> slam_ptr_;
    bool semanticOnline;
    std::string sensor;

public:
    ImageConverter(): it_(nh_)
    {
        // 初始化
        image_sub_color = new message_filters::Subscriber<sensor_msgs::Image>(nh_, "/camera/color/image_raw", 1);
        image_sub_depth = new message_filters::Subscriber<sensor_msgs::Image>(nh_, "/camera/aligned_depth_to_color/image_raw", 1);
        sync_ = new  message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), *image_sub_color, *image_sub_depth);
        sync_->registerCallback(boost::bind(&ImageConverter::callback,this, _1, _2));

        WORK_SPACE_PATH = ros::package::getPath("ros_evo") + "/../";
        const std::string VocFile  = WORK_SPACE_PATH + "/Vocabulary/ORBvoc.bin";
        const std::string YamlFile = WORK_SPACE_PATH + "/ros_test/config/D435i.yaml";

        ros::param::param<std::string>("~sensor", sensor, "RGBD");
        ros::param::param<bool>("~online", semanticOnline, "true");
        if (sensor == "RGBD"){
            slam_ptr_ = std::make_shared<ORB_SLAM2::System>(VocFile, YamlFile, "Full", ORB_SLAM2::System::RGBD, true, semanticOnline);
        }else if (sensor == "MONO"){
            slam_ptr_ = std::make_shared<ORB_SLAM2::System>(VocFile, YamlFile, "Full", ORB_SLAM2::System::MONOCULAR, true, semanticOnline);
        }else{
            std::cerr << "[ERROR] ONLY SUPPORT RGBD OR MONOCULAR! " << std::endl;
            return;
        }

        ROS_INFO("SUCCESS TO READ PARAM!");
  }

    ~ImageConverter()
    {
        slam_ptr_->Shutdown();
        slam_ptr_->SaveTrajectoryTUM("CameraTrajectory.txt");
    }

    // 回调函数
    // 输入：RBG彩色图像和深度图像
    void callback(const sensor_msgs::ImageConstPtr& msgImg,const sensor_msgs::ImageConstPtr& msgDepth)
    {
        // ros->opencv 的常用套路
        cv_bridge::CvImagePtr cvImgPtr, cvDepthPtr;
        try{
            cvImgPtr   = cv_bridge::toCvCopy(msgImg,sensor_msgs::image_encodings::RGB8);
            cvDepthPtr = cv_bridge::toCvCopy(msgDepth,sensor_msgs::image_encodings::TYPE_16UC1);
        }
        catch(cv_bridge::Exception e){
            ROS_ERROR_STREAM("Cv_bridge Exception:" << e.what());
            return;
        }

        // 数据类型转换，得到彩色图、灰度图和深度图
        cv::Mat cvColorImgMat = cvImgPtr->image;
        cv::Mat cvDepthMat = cvDepthPtr->image;

        // cv::Mat CurrentGray;
        // cv::cvtColor(cvColorImgMat,CurrentGray,CV_BGR2GRAY);

        ros::Time ros_time = ros::Time::now();
        double current_time = ros_time.toSec();
        if (sensor == "RGBD"){
            slam_ptr_->TrackRGBD(cvColorImgMat, cvDepthMat, current_time);
        }else if (sensor == "MONO"){
            slam_ptr_->TrackMonocular(cvColorImgMat, current_time);
        }

    }


};


int main(int argc, char *argv[])
{
    ros::init(argc, argv, "ros_node");

    // 定义类
    ImageConverter ic;
    // 堵塞函数
    ros::spin();

    return 0;
}
