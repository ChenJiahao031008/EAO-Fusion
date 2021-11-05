/*
 * @Author: Chen Jiahao
 * @Date: 2021-11-04 14:29:27
 * @LastEditors: Chen Jiahao
 * @LastEditTime: 2021-11-05 20:02:24
 * @Description: file content
 * @FilePath: /catkin_ws/src/EAO-SLAM/ros_test/include/message_flow.h
 */
#ifndef _MESSAGE_FLOW_H_
#define _MESSAGE_FLOW_H_

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
    std::deque<sensor_msgs::Imu> unsynced_imu_data_buff_;
    std::deque<sensor_msgs::Image> image_color_data_buff_;
    std::deque<sensor_msgs::Image> image_depth_data_buff_;

    sensor_msgs::Imu current_imu_data_;
    sensor_msgs::Image current_image_color_data_;
    sensor_msgs::Image current_image_depth_data_;
    cv::Mat cvColorImgMat, cvDepthMat;
    // cv::Mat CurrentGray;

public:
    MessageFlow(ros::NodeHandle &nh);
    ~MessageFlow();

    void Run();
    bool ReadData();
    bool HasData();
    bool ValidData();

    bool IMUSyncData(std::deque<sensor_msgs::Imu> &UnsyncedData, std::deque<sensor_msgs::Imu> &SyncedData, ros::Time sync_time);

};

#endif
