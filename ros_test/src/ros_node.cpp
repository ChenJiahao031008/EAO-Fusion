#include "ros/ros.h"
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

#include "Global.h"
#include "glog/logging.h"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.h"
#include "sensor_msgs/Image.h"

#include "imu_subscriber.h"
#include "image_subscriber.h"

// IMU相关
#include <sensor_msgs/Imu.h>

using namespace std;
using namespace cv;

std::string WORK_SPACE_PATH = "";

class MessageFlow
{
public:
    // 图像数据指针
    std::shared_ptr<IMGSubscriber> image_sub_ptr_;
    // IMU数据指针
    std::shared_ptr<IMUSubscriber> imu_sub_ptr_;
    // ORB-SLAM指针
    std::shared_ptr<ORB_SLAM2::System> slam_ptr_;

private:
    bool semanticOnline;
    std::string sensor;

    std::deque<sensor_msgs::Imu> imu_data_buff_;
    std::deque<sensor_msgs::Image> image_color_data_buff_;
    std::deque<sensor_msgs::Image> image_depth_data_buff_;

    sensor_msgs::Imu current_imu_data_;
    sensor_msgs::Image current_image_color_data_;
    sensor_msgs::Image current_image_depth_data_;
    cv::Mat cvColorImgMat, cvDepthMat;
    // cv::Mat CurrentGray;

public:
    MessageFlow(ros::NodeHandle &nh)
    {
        // 初始化图像
        image_sub_ptr_ = std::make_shared<IMGSubscriber>(nh, "/camera/color/image_raw", "/camera/aligned_depth_to_color/image_raw", 5);
        // 初始化IMU
        imu_sub_ptr_   = std::make_shared<IMUSubscriber>(nh, "/camera/imu", 50);

        // 读取参数文件
        const std::string VocFile  = WORK_SPACE_PATH + "/Vocabulary/ORBvoc.bin";
        const std::string YamlFile = WORK_SPACE_PATH + "/ros_test/config/D435i.yaml";

        // 读取launch文件中的参数
        ros::param::param<std::string>("~sensor", sensor, "RGBD");
        ros::param::param<bool>("~online", semanticOnline, "true");

        // 系统初始化
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

    ~MessageFlow()
    {
        slam_ptr_->Shutdown();
        slam_ptr_->SaveTrajectoryTUM("CameraTrajectory.txt");
    }

    void Run(){
        if (!ReadData())
            return;

        while (HasData())
        {
            if (!ValidData())
                continue;
            double real_time = ros::Time::now().toSec();
            double current_time = current_image_color_data_.header.stamp.toSec();
            double imu_time = current_imu_data_.header.stamp.toSec();
            ROS_INFO("current_time is : %f", current_time);
            ROS_INFO("imu_time is : %f", imu_time);
            if (sensor == "RGBD")
            {
                slam_ptr_->TrackRGBD(cvColorImgMat, cvDepthMat, current_time);
            }else if (sensor == "MONO"){
                slam_ptr_->TrackMonocular(cvColorImgMat, current_time);
            }
        }

    };

    bool ReadData(){
        static bool sensor_inited = false;
        imu_sub_ptr_->ParseData(imu_data_buff_);
        image_sub_ptr_->ParseData(image_color_data_buff_, image_depth_data_buff_);
        if (sensor_inited == false && image_color_data_buff_.size() == 0)
            return false;
        sensor_inited = true;
        return true;
        std::cout << "SIZE " << imu_data_buff_.size() << std::endl;
    };

    bool HasData(){
        if(image_color_data_buff_.size() == 0) return false;
        if(image_depth_data_buff_.size() == 0) return false;
        if(imu_data_buff_.size() == 0) return false;
        return true;
    };

    bool ValidData(){
        current_imu_data_ = imu_data_buff_.front();
        current_image_color_data_ = image_color_data_buff_.front();
        current_image_depth_data_ = image_depth_data_buff_.front();

        double diff_time = current_image_color_data_.header.stamp.toSec() - current_imu_data_.header.stamp.toSec();

        imu_data_buff_.pop_front();
        image_color_data_buff_.pop_front();
        image_depth_data_buff_.pop_front();

        if (diff_time > 0.05)
            return false;

        cv_bridge::CvImagePtr cvImgPtr, cvDepthPtr;
        try{
            cvImgPtr = cv_bridge::toCvCopy(current_image_color_data_, sensor_msgs::image_encodings::BGR8);
            cvDepthPtr = cv_bridge::toCvCopy(current_image_depth_data_, sensor_msgs::image_encodings::TYPE_16UC1);
        }catch (cv_bridge::Exception e){
            ROS_ERROR_STREAM("Cv_bridge Exception:" << e.what());
            return false;
        }

        // cv::cvtColor(cvColorImgMat,CurrentGray,CV_BGR2GRAY);

        cvColorImgMat = cvImgPtr->image;
        cvDepthMat = cvDepthPtr->image;

        return true;

    };
};


int main(int argc, char *argv[])
{
    // google::InitGoogleLogging(argv[0]);
    WORK_SPACE_PATH = ros::package::getPath("ros_evo") + "/../";
    // FLAGS_log_dir = WORK_SPACE_PATH + "/LOG";
    // FLAGS_alsologtostderr = 1;

    ros::init(argc, argv, "ros_node");
    ros::NodeHandle nh;

    // 定义类
    std::shared_ptr<MessageFlow> message_flow_ptr = std::make_shared<MessageFlow>(nh);

    ros::Rate rate(100);
    while (ros::ok())
    {
        ros::spinOnce();

        message_flow_ptr->Run();

        rate.sleep();
    }

    return 0;
}
