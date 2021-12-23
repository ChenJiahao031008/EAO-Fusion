/*
 * @Author: Chen Jiahao
 * @Date: 2021-10-29 10:08:18
 * @LastEditors: Chen Jiahao
 * @LastEditTime: 2021-12-23 21:08:38
 * @Description: file content
 * @FilePath: /catkin_ws/src/EAO-SLAM/include/Map.h
 */
/**
* This file is part of ORB-SLAM2.
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* Modification: EAO-SLAM
* Version: 1.0
* Created: 07/18/2019
* Author: Yanmin Wu
* E-mail: wuyanminmax@gmail.com
*/

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "MapPlane.h"
#include "KeyFrame.h"
#include "Frame.h"
#include <set>

#include <mutex>
#include <set>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/ModelCoefficients.h>

namespace ORB_SLAM2
{

class Frame;
class MapPoint;
class MapPlane;
class KeyFrame;
class Object_Map;

class Map
{
public:
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    Map();

    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);
    void EraseMapPoint(MapPoint* pMP);
    void EraseMapPlane(MapPlane *pMP);
    void EraseKeyFrame(KeyFrame* pKF);
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);

    // void AddObjectMapPoints(MapPoint* pMP);


    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();

    std::vector<Object_Map*> GetObjects();
    std::vector<cv::Mat> GetCubeCenters();

    long unsigned int MapPointsInMap();
    long unsigned  KeyFramesInMap();

    long unsigned int GetMaxKFid();

    void clear();

    vector<KeyFrame*> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;

    vector<Object_Map*> mvObjectMap;    // objects in the map.
    vector<cv::Mat> cube_center;

    // add plane -------------------------------
    void AddMapPlane(MapPlane *pMP);

    std::vector<MapPlane*> GetAllMapPlanes();

    void AssociatePlanesByBoundary(Frame &pF, bool out=false);

    double PointDistanceFromPlane(const cv::Mat &plane, PointCloud::Ptr boundry, bool out=false);


    void SearchMatchedPlanes(KeyFrame *pKF, cv::Mat Scw, const vector<MapPlane *> &vpPlanes, vector<MapPlane *> &vpMatched, bool out=false);

    std::vector<long unsigned int> GetRemovedPlanes();

    // add plane end ----------------------------------

protected : std::set<MapPoint *> mspMapPoints;
    std::set<KeyFrame*> mspKeyFrames;

    std::vector<MapPoint*> mvpReferenceMapPoints;

    // add plane -------------------------------
    float mfDisTh;
    float mfAngleTh;
    std::set<MapPlane*> mspMapPlanes;
    std::vector<long unsigned int> mvnRemovedPlanes;

    std::set<MapPoint*> mvpObjectMapPoints;

    long unsigned int mnMaxKFid;


    std::mutex mMutexMap;
};

} //namespace ORB_SLAM

#endif // MAP_H
