#include "message_flow.h"
#include <cassert>

// #define BACKWARD_HAS_DW 1
// #include "backward.hpp"
// namespace backward
// {
//     backward::SignalHandling sh;
// }

Eigen::Matrix4d INIT_POSE = Eigen::Matrix4d::Identity();

// ---------------------------------------------------------------------- //
// ----------------          [WITH IMU INFO]             ---------------- //
// ---------------------------------------------------------------------- //

RGBDIMessageFlow::RGBDIMessageFlow(ros::NodeHandle &nh)
{
    // 初始化图像
    image_sub_ptr_ = std::make_shared<IMGSubscriber>(nh, "/camera/color/image_raw", "/camera/aligned_depth_to_color/image_raw", 1000);
    // image_sub_ptr_ = std::make_shared<IMGSubscriber>(nh, "/camera/rgb/image_color", "/camera/depth/image", 1000);
    // 初始化IMU
    imu_sub_ptr_ = std::make_shared<IMUSubscriber>(nh, "/camera/imu", 500000);

    // 读取参数文件
    const std::string VocFile = WORK_SPACE_PATH + "/Vocabulary/ORBvoc.bin";
    const std::string YamlFile = WORK_SPACE_PATH + "/ros_test/config/D435i.yaml";

    // 读取launch文件中的参数
    ros::param::param<std::string>("~sensor", sensor, "RGBD");
    ros::param::param<bool>("~online", semanticOnline, "true");
    ros::param::param<bool>("~rosbag", rosBagFlag, "false");;
    ros::param::param<int>("~gravity_aixs", gravity_aixs, 3);

    initIMUFlag = false;

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

RGBDIMessageFlow::~RGBDIMessageFlow()
{
}

void RGBDIMessageFlow::Run()
{
    if (!ReadData())
        return;

    while (HasData())
    {
        if (!ValidData())
            continue;
        if (!InitIMU())
            continue;
        double real_time = ros::Time::now().toSec();
        double current_time = current_image_color_data_.header.stamp.toSec();
        double imu_time = synced_imu_data_.header.stamp.toSec();
        double unsynced_imu_time_back = unsynced_imu_data_.back().header.stamp.toSec();
        double unsynced_imu_time_front = unsynced_imu_data_.front().header.stamp.toSec();
        double diff_time = unsynced_imu_time_back - unsynced_imu_time_front;

        // std::cout << synced_imu_data_.linear_acceleration << std::endl;

        // ROS_INFO("[DEBUG] current_time is : %f", current_time);
        // ROS_INFO("[DEBUG] imu_time is : %f", imu_time);
        // ROS_INFO("[DEBUG] unsynced_imu_time_back is : %f", unsynced_imu_time_back);
        // ROS_INFO("[DEBUG] unsynced_imu_time_front is : %f", unsynced_imu_time_front);
        // ROS_INFO("[DEBUG] diff_time is : %f", diff_time);

        if (sensor == "RGBD"){
            slam_ptr_->TrackRGBD(cvColorImgMat, cvDepthMat, current_time);
        }else if (sensor == "MONO"){
            slam_ptr_->TrackMonocular(cvColorImgMat, current_time);
        }

        unsynced_imu_data_.clear();
    }
}

bool RGBDIMessageFlow::ReadData()
{
    image_sub_ptr_->ParseData(image_color_data_buff_, image_depth_data_buff_);
    // int beforeBuffSize = unsynced_imu_data_buff_.size();
    imu_sub_ptr_->ParseData(unsynced_imu_data_buff_);
    // int afterBuffSize = unsynced_imu_data_buff_.size();
    // int diffBuffSize = afterBuffSize - beforeBuffSize;

    if (image_color_data_buff_.size() == 0 || image_depth_data_buff_.size() == 0)
        return false;

    ros::Time image_time;
    if (rosBagFlag)
        image_time = image_color_data_buff_.front().header.stamp;
    else
        image_time = image_color_data_buff_.back().header.stamp;


    bool valid_imu = IMUSyncData(
        unsynced_imu_data_buff_, unsynced_imu_data_, synced_imu_data_buff_, image_time);

    // only mark lidar as 'inited' when all the three sensors are synced:
    static bool sensor_inited = false;
    if (!sensor_inited)
    {
        if (!valid_imu)
        {
            std::cerr << "[WARNNIGN] FAIL TO START." << std::endl;
            if (rosBagFlag){
                image_color_data_buff_.pop_front();
                image_depth_data_buff_.pop_front();
            }else{
                image_color_data_buff_.pop_back();
                image_depth_data_buff_.pop_back();
            }
            return false;
        }
        sensor_inited = true;
    }

    return true;
}

bool RGBDIMessageFlow::IMUSyncData(
    std::deque<sensor_msgs::Imu> &UnsyncedDataBuff,
    std::deque<sensor_msgs::Imu> &UnsyncedData,
    std::deque<sensor_msgs::Imu> &SyncedData,
    ros::Time sync_time)
{
    // 传感器数据按时间序列排列，在传感器数据中为同步的时间点找到合适的时间位置
    // 即找到与同步时间相邻的左右两个数据
    // 需要注意的是，如果左右相邻数据有一个离同步时间差值比较大，则说明数据有丢失，时间离得太远不适合做差值
    while (UnsyncedDataBuff.size() >= 2)
    {
        // UnsyncedDataBuff.front().time should be <= sync_time:
        // IMU起始时间晚于图像时间时，删除图像数据
        if (UnsyncedDataBuff.front().header.stamp.toSec() > sync_time.toSec()){
            return false;
        }else{
            // 记录最开始的IMU数据
            UnsyncedData.push_back(UnsyncedDataBuff.front());
        }
        // sync_time.toSec() should be <= UnsyncedDataBuff.at(1).header.stamp.toSec():
        if (UnsyncedDataBuff.at(1).header.stamp.toSec() < sync_time.toSec())
        {
            // 记录第一帧和最后一帧之间的图像数据
            UnsyncedData.push_back(UnsyncedDataBuff.at(1));
            UnsyncedDataBuff.pop_front();
            continue;
        }

        // sync_time.toSec() - UnsyncedDataBuff.front().header.stamp.toSec() should be <= 0.2:
        if (sync_time.toSec() - UnsyncedDataBuff.front().header.stamp.toSec() > 0.2)
        {
            UnsyncedDataBuff.pop_front();
            UnsyncedData.clear();
            return false;
        }
        // UnsyncedDataBuff.at(1).header.stamp.toSec() - sync_time.toSec() should be <= 0.2
        if (UnsyncedDataBuff.at(1).header.stamp.toSec() - sync_time.toSec() > 0.2)
        {
            UnsyncedDataBuff.pop_front();
            UnsyncedData.clear();
            return false;
        }
        break;
    }
    if (UnsyncedDataBuff.size() < 2)
    {
        return false;
    }

    sensor_msgs::Imu front_data = UnsyncedDataBuff.at(0);
    sensor_msgs::Imu back_data = UnsyncedDataBuff.at(1);
    sensor_msgs::Imu synced_data;

    double front_scale = (back_data.header.stamp.toSec() - sync_time.toSec()) / (back_data.header.stamp.toSec() - front_data.header.stamp.toSec());
    double back_scale = (sync_time.toSec() - front_data.header.stamp.toSec()) / (back_data.header.stamp.toSec() - front_data.header.stamp.toSec());

    synced_data.header.stamp = sync_time;
    synced_data.linear_acceleration.x = front_data.linear_acceleration.x * front_scale + back_data.linear_acceleration.x * back_scale;
    synced_data.linear_acceleration.y = front_data.linear_acceleration.y * front_scale + back_data.linear_acceleration.y * back_scale;
    synced_data.linear_acceleration.z = front_data.linear_acceleration.z * front_scale + back_data.linear_acceleration.z * back_scale;
    synced_data.angular_velocity.x = front_data.angular_velocity.x * front_scale + back_data.angular_velocity.x * back_scale;
    synced_data.angular_velocity.y = front_data.angular_velocity.y * front_scale + back_data.angular_velocity.y * back_scale;
    synced_data.angular_velocity.z = front_data.angular_velocity.z * front_scale + back_data.angular_velocity.z * back_scale;

    SyncedData.push_back(synced_data);
    UnsyncedData.push_back(UnsyncedDataBuff.at(1));
    UnsyncedData.push_back(synced_data);
    // // debug : 好像没有弹出front_data 和 back_data
    // UnsyncedDataBuff.pop_front();
    // UnsyncedDataBuff.pop_front();
    return true;
}

bool RGBDIMessageFlow::HasData()
{
    if (image_color_data_buff_.size() == 0)
        return false;
    if (image_depth_data_buff_.size() == 0)
        return false;
    if (synced_imu_data_buff_.size() == 0)
        return false;
    return true;
}

bool RGBDIMessageFlow::ValidData()
{
    double image_time;
    if (rosBagFlag){
        image_time = image_color_data_buff_.front().header.stamp.toSec();
        current_image_color_data_ = image_color_data_buff_.front();
        current_image_depth_data_ = image_depth_data_buff_.front();
        image_color_data_buff_.pop_front();
        image_depth_data_buff_.pop_front();
    }else{
        image_time = image_color_data_buff_.back().header.stamp.toSec();
        current_image_color_data_ = image_color_data_buff_.back();
        current_image_depth_data_ = image_depth_data_buff_.back();
        image_color_data_buff_.clear();
        image_depth_data_buff_.clear();
    }


    synced_imu_data_ = synced_imu_data_buff_.front();
    synced_imu_data_buff_.pop_front();

    double diff_time = current_image_color_data_.header.stamp.toSec() - synced_imu_data_.header.stamp.toSec();

    if (diff_time > 0.05)
        return false;

    // // 初始时刻不准，扔掉
    // if (count < 30){
    //     count++;
    //     return false;
    // }


    cv_bridge::CvImagePtr cvImgPtr, cvDepthPtr;
    try
    {
        cvImgPtr = cv_bridge::toCvCopy(current_image_color_data_, sensor_msgs::image_encodings::BGR8);
        cvDepthPtr = cv_bridge::toCvCopy(current_image_depth_data_, sensor_msgs::image_encodings::TYPE_16UC1);
    }
    catch (cv_bridge::Exception e)
    {
        ROS_ERROR_STREAM("CV_bridge Exception:" << e.what());
        return false;
    }

    // cv::cvtColor(cvColorImgMat,CurrentGray,CV_BGR2GRAY);

    cvColorImgMat = cvImgPtr->image;
    cvDepthMat = cvDepthPtr->image;

    return true;
}

bool RGBDIMessageFlow::InitIMU(){
    if (initIMUFlag)
        return true;
    Eigen::Vector3d initIMU(synced_imu_data_.linear_acceleration.x, synced_imu_data_.linear_acceleration.y, synced_imu_data_.linear_acceleration.z);

    Eigen::Vector3d alpha_1 = initIMU;
    Eigen::Vector3d alpha_2, alpha_3;
    if (gravity_aixs == 2)
    {
        alpha_2 = Eigen::Vector3d(1, 0, 0);
        alpha_3 = Eigen::Vector3d(0, 0, 1);
    }else if (gravity_aixs == 3){
        alpha_2 = Eigen::Vector3d(1, 0, 0);
        alpha_3 = Eigen::Vector3d(0, 1, 0);
    }
    Eigen::Vector3d beta_1 = alpha_1.normalized();
    Eigen::Vector3d beta_2 = (alpha_2 - beta_1.dot(alpha_2) * 1.0 / beta_1.dot(beta_1) * beta_1).normalized();
    Eigen::Vector3d beta_3 = (alpha_3 - beta_1.dot(alpha_3) * 1.0 / beta_1.dot(beta_1) * beta_1 - beta_2.dot(alpha_3) * 1.0 / beta_2.dot(beta_2) * beta_2).normalized();
    Eigen::Matrix3d InitR = Eigen::Matrix3d::Identity();
    if (gravity_aixs == 2)
    {
        InitR.block<3, 1>(0, 0) = beta_2;
        InitR.block<3, 1>(0, 1) = beta_1;
        InitR.block<3, 1>(0, 2) = beta_3;
    }else if (gravity_aixs == 3){
        InitR.block<3, 1>(0, 0) = beta_2;
        InitR.block<3, 1>(0, 1) = beta_3;
        InitR.block<3, 1>(0, 2) = beta_1;
    }
    assert(InitR.determinant() == 1 || InitR.determinant() == -1);
    if (InitR.determinant() == -1){
        InitR = -1 * InitR;
        assert(InitR.determinant() == 1);
    }
    INIT_POSE.block<3, 3>(0, 0) = InitR;
    // initIMUFlag = true;
    // std::cout << "[INFO] INIT_POSE IS: \n" << INIT_POSE << std::endl;
    return true;
}

void RGBDIMessageFlow::SaveTrajectory(){
    slam_ptr_->SaveTrajectoryTUM("CameraTrajectory.txt");
    slam_ptr_->Shutdown();
}

// ---------------------------------------------------------------------- //
// ----------------        [WITHOUT IMU INFO]            ---------------- //
// ---------------------------------------------------------------------- //
RGBDMessageFlow::RGBDMessageFlow(ros::NodeHandle &nh)
{
    // 初始化图像
    image_sub_ptr_ = std::make_shared<IMGSubscriber>(nh, "/camera/color/image_raw", "/camera/aligned_depth_to_color/image_raw", 1000);
    // image_sub_ptr_ = std::make_shared<IMGSubscriber>(nh, "/camera/rgb/image_color", "/camera/depth/image", 1000);

    // 读取参数文件
    const std::string VocFile = WORK_SPACE_PATH + "/Vocabulary/ORBvoc.bin";
    // const std::string YamlFile = WORK_SPACE_PATH + "/ros_test/config/D435i.yaml";
    const std::string YamlFile = WORK_SPACE_PATH + "/ros_test/config/TUM2.yaml";

    // 读取launch文件中的参数
    ros::param::param<std::string>("~sensor", sensor, "RGBD");
    ros::param::param<bool>("~online", semanticOnline, "true");
    ros::param::param<bool>("~rosbag", rosBagFlag, "false");


    // 系统初始化
    if (sensor == "RGBD")
        slam_ptr_ = std::make_shared<ORB_SLAM2::System>(VocFile, YamlFile, "Full", ORB_SLAM2::System::RGBD, true, semanticOnline);
    else if (sensor == "MONO")
        slam_ptr_ = std::make_shared<ORB_SLAM2::System>(VocFile, YamlFile, "Full", ORB_SLAM2::System::MONOCULAR, true, semanticOnline);
    else{
        std::cerr << "[ERROR] ONLY SUPPORT RGBD OR MONOCULAR! " << std::endl;
        return;
    }

    ROS_INFO("SUCCESS TO READ PARAM!");
}

RGBDMessageFlow::~RGBDMessageFlow()
{
}

void RGBDMessageFlow::Run()
{
    if (!ReadData())
        return;

    while (HasData())
    {
        if (!ValidData())
            continue;
        double current_time = ros::Time::now().toSec();
        // double current_time = current_image_color_data_.header.stamp.toSec();

        if (sensor == "RGBD")
            slam_ptr_->TrackRGBD(cvColorImgMat, cvDepthMat, current_time);
        else if (sensor == "MONO")
            slam_ptr_->TrackMonocular(cvColorImgMat, current_time);
    }
}

bool RGBDMessageFlow::ReadData()
{
    image_sub_ptr_->ParseData(image_color_data_buff_, image_depth_data_buff_);
    if (image_color_data_buff_.size() == 0 || image_depth_data_buff_.size() == 0)
        return false;
    return true;
}

bool RGBDMessageFlow::HasData()
{
    if (image_color_data_buff_.size() == 0)
        return false;
    if (image_depth_data_buff_.size() == 0)
        return false;
    return true;
}

bool RGBDMessageFlow::ValidData()
{
    double image_time;
    if (rosBagFlag)
    {
        current_image_color_data_ = image_color_data_buff_.front();
        current_image_depth_data_ = image_depth_data_buff_.front();
        image_color_data_buff_.pop_front();
        image_depth_data_buff_.pop_front();
    }
    else
    {
        current_image_color_data_ = image_color_data_buff_.back();
        current_image_depth_data_ = image_depth_data_buff_.back();
        image_color_data_buff_.clear();
        image_depth_data_buff_.clear();
    }

    // // 初始时刻不准，扔掉
    // if (count < 30){
    //     count++;
    //     return false;
    // }

    cv_bridge::CvImagePtr cvImgPtr, cvDepthPtr;
    try
    {
        cvImgPtr = cv_bridge::toCvCopy(current_image_color_data_, sensor_msgs::image_encodings::BGR8);
        cvDepthPtr = cv_bridge::toCvCopy(current_image_depth_data_, sensor_msgs::image_encodings::TYPE_16UC1);
    }
    catch (cv_bridge::Exception e)
    {
        ROS_ERROR_STREAM("CV_bridge Exception:" << e.what());
        return false;
    }

    // cv::cvtColor(cvColorImgMat,CurrentGray,CV_BGR2GRAY);

    cvColorImgMat = cvImgPtr->image;
    cvDepthMat = cvDepthPtr->image;

    return true;
}

void RGBDMessageFlow::SaveTrajectory()
{
    slam_ptr_->SaveTrajectoryTUM("CameraTrajectory.txt");
    slam_ptr_->Shutdown();
}
