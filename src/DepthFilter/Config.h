/*
 * @Author: your name
 * @Date: 2021-03-26 11:25:51
 * @LastEditTime: 2021-12-27 16:45:33
 * @LastEditors: Chen Jiahao
 * @Description: In User Settings Edit
 * @FilePath: /catkin_ws/src/EAO-SLAM/src/DepthFilter/Config.h
 */
#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cassert>

class Config
{
public:
    struct AppSettings
    {
        std::string fillMode;
        std::string blurType;
        int extrapolate;
        int resize;
        float maxDepth;
        float minDepth;
    };


public:
    cv::FileStorage SettingsFile;
    AppSettings app;


public:
    Config(cv::FileStorage &fsSettings):SettingsFile(fsSettings)
    {
        AppSettingsInit();
    };

    void AppSettingsInit(){
        app.fillMode = static_cast<std::string>(SettingsFile["FillType"]);
        app.blurType = static_cast<std::string>(SettingsFile["BlurType"]);
        app.extrapolate = SettingsFile["Extrapolate"];
        app.resize = SettingsFile["Resize"];
        app.maxDepth = SettingsFile["maxDepth"];
        app.minDepth = SettingsFile["minDepth"];
    };
};

#endif //CONFIG_H

