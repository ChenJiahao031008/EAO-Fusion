/*
 * @Author: Chen Jiahao
 * @Date: 2021-10-29 10:08:20
 * @LastEditors: Chen Jiahao
 * @LastEditTime: 2021-12-23 21:51:31
 * @Description: file content
 * @FilePath: /catkin_ws/src/EAO-SLAM/ros_test/app/ros_rgbd_imu.cpp
 */
#include "message_flow.h"
#include "ros_evo/saveOdometry.h"

using namespace std;
using namespace cv;

std::string WORK_SPACE_PATH = "";

bool save_odometry = false;
bool SaveOdometryCb(ros_evo::saveOdometry::Request &request, ros_evo::saveOdometry::Response &response)
{
    save_odometry = true;
    response.succeed = true;
    return response.succeed;
}

int main(int argc, char *argv[])
{
    WORK_SPACE_PATH = ros::package::getPath("ros_evo") + "/../";

    ros::init(argc, argv, "ros_node");
    ros::NodeHandle nh;

    // register service for optimized trajectory save:
    ros::ServiceServer service = nh.advertiseService("save_odometry", SaveOdometryCb);
    std::shared_ptr<RGBDIMessageFlow> message_flow_ptr = std::make_shared<RGBDIMessageFlow>(nh);

    ros::Rate rate(100);
    while (ros::ok())
    {
        ros::spinOnce();

        message_flow_ptr->Run();
        if (save_odometry)
        {
            save_odometry = false;
            message_flow_ptr->SaveTrajectory();
        }

        rate.sleep();
    }

    return 0;
}
