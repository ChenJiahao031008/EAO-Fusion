/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef SYSTEM_H
#define SYSTEM_H

#include<string>
#include<thread>
#include<opencv2/core/core.hpp>

#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"

#include "YOLOX.h"

#include "Global.h"
#include "config/project_config.h"
#include <interpreter.hpp>
// #include "bytetrack_impl.h"

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class Tracking;
class LocalMapping;
class LoopClosing;
class YOLOX;
// class BYTETrackerImpl;

class System
{
public:
    // Input sensor
    enum eSensor{
        MONOCULAR=0,
        STEREO=1,
        RGBD=2
    };

public:
    System( const string &strVocFile, const string &strSettingsFile, const string &flag,
            const eSensor sensor, const bool bUseViewer = true, const bool SemanticOnline = false);

    cv::Mat TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp);

    void ActivateLocalizationMode();
    void DeactivateLocalizationMode();
    void Reset();
    void Shutdown();
    void SaveTrajectoryTUM(const string &filename);
    void SaveKeyFrameTrajectoryTUM(const string &filename);
    void SaveTrajectoryKITTI(const string &filename);

public:
    Tracking* mpTracker; // temp

private:

    // Input sensor
    eSensor mSensor;

    ORBVocabulary* mpVocabulary;
    KeyFrameDatabase* mpKeyFrameDatabase;
    Map* mpMap;

    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopCloser;
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    std::thread* mptLocalMapping;
    std::thread* mptLoopClosing;
    std::thread* mptViewer;

    YOLOX *mpSemanticer;         // yolo online.
    std::thread* mptSemanticer;  // yolo online.
    // BYTETrackerImpl *mpByteTrack;
    std::thread *mptByteTrack;

    // Reset flag
    std::mutex mMutexReset;
    bool mbReset;

    // Change mode flags
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;
    bool mbDeactivateLocalizationMode;

    bool bSemanticOnline;    // yolo whether online.
    std::mutex mMutexState;
};

}// namespace ORB_SLAM

#endif // SYSTEM_H
