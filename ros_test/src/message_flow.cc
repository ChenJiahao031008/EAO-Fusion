#include "message_flow.h"

MessageFlow::MessageFlow(ros::NodeHandle &nh)
{
    // 初始化图像
    image_sub_ptr_ = std::make_shared<IMGSubscriber>(nh, "/camera/color/image_raw", "/camera/aligned_depth_to_color/image_raw", 100);
    // 初始化IMU
    imu_sub_ptr_ = std::make_shared<IMUSubscriber>(nh, "/camera/imu", 5000);

    // 读取参数文件
    const std::string VocFile = WORK_SPACE_PATH + "/Vocabulary/ORBvoc.bin";
    const std::string YamlFile = WORK_SPACE_PATH + "/ros_test/config/D435i.yaml";

    // 读取launch文件中的参数
    ros::param::param<std::string>("~sensor", sensor, "RGBD");
    ros::param::param<bool>("~online", semanticOnline, "true");

    // 系统初始化
    if (sensor == "RGBD")
    {
        slam_ptr_ = std::make_shared<ORB_SLAM2::System>(VocFile, YamlFile, "Full", ORB_SLAM2::System::RGBD, true, semanticOnline);
    }
    else if (sensor == "MONO")
    {
        slam_ptr_ = std::make_shared<ORB_SLAM2::System>(VocFile, YamlFile, "Full", ORB_SLAM2::System::MONOCULAR, true, semanticOnline);
    }
    else
    {
        std::cerr << "[ERROR] ONLY SUPPORT RGBD OR MONOCULAR! " << std::endl;
        return;
    }

    ROS_INFO("SUCCESS TO READ PARAM!");
}

MessageFlow::~MessageFlow()
{
    slam_ptr_->Shutdown();
    slam_ptr_->SaveTrajectoryTUM("CameraTrajectory.txt");
}

void MessageFlow::Run()
{
    if (!ReadData())
        return;

    while (HasData())
    {
        if (!ValidData())
            continue;
        double real_time = ros::Time::now().toSec();
        double current_time = current_image_color_data_.header.stamp.toSec();
        double imu_time = current_imu_data_.header.stamp.toSec();

        ROS_INFO("[DEBUG] current_time is : %f", current_time);
        ROS_INFO("[DEBUG] imu_time is : %f", imu_time);

        if (sensor == "RGBD"){
            slam_ptr_->TrackRGBD(cvColorImgMat, cvDepthMat, current_time);
        }else if (sensor == "MONO"){
            slam_ptr_->TrackMonocular(cvColorImgMat, current_time);
        }
    }
}

bool MessageFlow::ReadData()
{
    static std::deque<sensor_msgs::Imu> unsynced_imu_;

    image_sub_ptr_->ParseData(image_color_data_buff_, image_depth_data_buff_);
    imu_sub_ptr_->ParseData(unsynced_imu_);

    if (image_color_data_buff_.size() == 0)
        return false;

    ros::Time image_time = image_color_data_buff_.front().header.stamp;
    bool valid_imu = IMUSyncData(unsynced_imu_, imu_data_buff_, image_time);

    // only mark lidar as 'inited' when all the three sensors are synced:
    static bool sensor_inited = false;
    if (!sensor_inited)
    {
        if (!valid_imu)
        {
            std::cerr << "[WARNNIGN] FAIL TO START." << std::endl;
            image_color_data_buff_.pop_front();
            image_depth_data_buff_.pop_front();
            return false;
        }
        sensor_inited = true;
    }

    return true;
}

bool MessageFlow::IMUSyncData(std::deque<sensor_msgs::Imu> &UnsyncedData, std::deque<sensor_msgs::Imu> &SyncedData, ros::Time sync_time)
{
    // 传感器数据按时间序列排列，在传感器数据中为同步的时间点找到合适的时间位置
    // 即找到与同步时间相邻的左右两个数据
    // 需要注意的是，如果左右相邻数据有一个离同步时间差值比较大，则说明数据有丢失，时间离得太远不适合做差值
    while (UnsyncedData.size() >= 2)
    {
        // UnsyncedData.front().time should be <= sync_time:
        // IMU起始时间晚于图像时间时删除图像数据
        if (UnsyncedData.front().header.stamp.toSec() > sync_time.toSec())
            return false;
        // sync_time.toSec() should be <= UnsyncedData.at(1).header.stamp.toSec():
        if (UnsyncedData.at(1).header.stamp.toSec() < sync_time.toSec())
        {
            UnsyncedData.pop_front();
            continue;
        }

        // sync_time.toSec() - UnsyncedData.front().header.stamp.toSec() should be <= 0.2:
        if (sync_time.toSec() - UnsyncedData.front().header.stamp.toSec() > 0.2)
        {
            UnsyncedData.pop_front();
            return false;
        }
        // UnsyncedData.at(1).header.stamp.toSec() - sync_time.toSec() should be <= 0.2
        if (UnsyncedData.at(1).header.stamp.toSec() - sync_time.toSec() > 0.2)
        {
            UnsyncedData.pop_front();
            return false;
        }
        break;
    }
    if (UnsyncedData.size() < 2)
    {
        return false;
    }

    sensor_msgs::Imu front_data = UnsyncedData.at(0);
    sensor_msgs::Imu back_data = UnsyncedData.at(1);
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
    return true;
}

bool MessageFlow::HasData()
{
    if (image_color_data_buff_.size() == 0)
        return false;
    if (image_depth_data_buff_.size() == 0)
        return false;
    if (imu_data_buff_.size() == 0)
        return false;
    return true;
}

bool MessageFlow::ValidData()
{
    double image_time = image_color_data_buff_.front().header.stamp.toSec();
    current_image_color_data_ = image_color_data_buff_.front();
    current_image_depth_data_ = image_depth_data_buff_.front();

    current_imu_data_ = imu_data_buff_.front();

    double diff_time = current_image_color_data_.header.stamp.toSec() - current_imu_data_.header.stamp.toSec();

    imu_data_buff_.clear();

    image_color_data_buff_.pop_front();
    image_depth_data_buff_.pop_front();

    // if (diff_time > 0.05)
    //     return false;

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