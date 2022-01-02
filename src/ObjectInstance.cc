/*
 * @Author: Chen Jiahao
 * @Date: 2021-12-29 15:16:12
 * @LastEditors: Chen Jiahao
 * @LastEditTime: 2022-01-02 10:42:31
 * @Description: file content
 * @FilePath: /catkin_ws/src/EAO-SLAM/src/ObjectInstance.cc
 */

#include "ObjectInstance.h"

#include <iostream>
#include <stdint.h>
#include <random>
#include <vector>
#include <string>

namespace ORB_SLAM2
{
// class Object2DInstance ------------------------------------------------------

Object2DInstance::Object2DInstance(){};

float Object2DInstance::ComputeIoU(const Anchor &candidate)
{
    cv::Rect Union = anchor.rect | candidate.rect;
    cv::Rect Intersection = anchor.rect & candidate.rect;

    return Intersection.area() * 1.0 / Union.area();
}

bool Object2DInstance::UpdateAnchor(const Anchor &candidate)
{
    if (1.2 * candidate.score < anchor.score){
        return false;
    }else if (candidate.score > 1.2*anchor.score){
        anchor.score =  candidate.score;
        anchor.rect = candidate.rect;
        // TODO: 更新其他数据(如平均深度等)
    }else{
        float ratio = anchor.score * 1.0 / (anchor.score + candidate.score);
        int newX = ratio * anchor.rect.x + (1 - ratio) * anchor.rect.x;
        int newY = ratio * anchor.rect.y + (1 - ratio) * anchor.rect.y;
        int newWidth = ratio * anchor.rect.width + (1 - ratio) * anchor.rect.width;
        int newHeight = ratio * anchor.rect.height + (1 - ratio) * anchor.rect.height;
        cv::Rect newRect(newX, newY, newWidth, newHeight);
        anchor.rect = newRect;
    }
    return true;
}

void Object2DInstance::AddNewObservation(const int &idx, Frame &frame)
{
    count++;
    objInFrames.emplace_back(static_cast<std::shared_ptr<Frame>>(&frame));
    candidateID = idx;
    visibility++;
}

void Object2DInstance::AddFuzzyObservation(){
    visibility++;
}

// class Anchor ------------------------------------------------------
Anchor::Anchor(){};

bool Anchor::isInImageBoundary(const cv::Mat &image)
{
    cv::Rect imageBoundary = cv::Rect( 0, 0, image.cols, image.rows);
    return (rect == (rect & imageBoundary));
}

}
