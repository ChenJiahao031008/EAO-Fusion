/*
 * @Author: Chen Jiahao
 * @Date: 2021-11-04 15:02:06
 * @LastEditors: Chen Jiahao
 * @LastEditTime: 2021-11-05 19:52:16
 * @Description: file content
 * @FilePath: /catkin_ws/src/EAO-SLAM/ros_test/include/imu_subscriber.h
 */
#ifndef _IMU_SUBSCRIBER_HPP_
#define _IMU_SUBSCRIBER_HPP_

#include <deque>
#include <mutex>
#include <thread>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>

class IMUSubscriber {
  public:
    IMUSubscriber(ros::NodeHandle& nh, std::string topic_name, size_t buff_size);
    IMUSubscriber() = default;
    void ParseData(std::deque<sensor_msgs::Imu>& deque_imu_data);

  private:
    void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg_ptr);

  private:
    ros::NodeHandle nh_;
    ros::Subscriber subscriber_;
    std::deque<sensor_msgs::Imu> new_imu_data_;

    std::mutex buff_mutex_;
};

#endif
