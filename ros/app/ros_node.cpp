/*
 * @Author: Chen Jiahao
 * @Date: 2021-10-29 10:08:20
 * @LastEditors: Chen Jiahao
 * @LastEditTime: 2021-11-26 11:45:04
 * @Description: file content
 * @FilePath: /catkin_ws/src/EAO-SLAM/ros/app/ros_node.cpp
 */
#include "message_flow.h"
#include "object_slam/saveOdometry.h"

using namespace std;
using namespace cv;

std::string WORK_SPACE_PATH = "";

bool save_odometry = false;
bool SaveOdometryCb(object_slam::saveOdometry::Request &request, object_slam::saveOdometry::Response &response)
{
    save_odometry = true;
    response.succeed = true;
    return response.succeed;
}

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    WORK_SPACE_PATH = ros::package::getPath("object_slam") + "/../";
    FLAGS_log_dir = WORK_SPACE_PATH + "/LOG";
    FLAGS_alsologtostderr = 1;

    ros::init(argc, argv, "ros_node");
    ros::NodeHandle nh;

    // register service for optimized trajectory save:
    ros::ServiceServer service = nh.advertiseService("save_odometry", SaveOdometryCb);
    std::shared_ptr<MessageFlow> message_flow_ptr = std::make_shared<MessageFlow>(nh);

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
