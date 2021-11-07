/*
 * @Author: Chen Jiahao
 * @Date: 2021-10-29 10:08:20
 * @LastEditors: Chen Jiahao
 * @LastEditTime: 2021-11-07 16:44:42
 * @Description: file content
 * @FilePath: /catkin_ws/src/EAO-SLAM/ros_test/app/ros_node.cpp
 */
#include "message_flow.h"

using namespace std;
using namespace cv;

std::string WORK_SPACE_PATH = "";

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    WORK_SPACE_PATH = ros::package::getPath("ros_evo") + "/../";
    FLAGS_log_dir = WORK_SPACE_PATH + "/LOG";
    FLAGS_alsologtostderr = 1;

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
