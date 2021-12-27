/**
* This file is part of ORB-SLAM2.
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* Modification: EAO-SLAM
* Version: 1.0
* Created: 11/23/2019
* Author: Yanmin Wu
* E-mail: wuyanminmax@gmail.com
*/

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>

#include "Object.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

// #include "FrameDrawer.h"

// #define BACKWARD_HAS_DW 1
// #include "backward.hpp"
// namespace backward
// {
//     backward::SignalHandling sh;
// }

namespace ORB_SLAM2
{


MapDrawer::MapDrawer(Map* pMap, const string &strSettingPath):mpMap(pMap)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
    mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
    mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
    mPointSize = fSettings["Viewer.PointSize"];
    mCameraSize = fSettings["Viewer.CameraSize"];
    mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];
    // frontPath = "/home/chen/catkin_ws/src/EAO-SLAM/ros_test/config/Anonymous-Pro.ttf";
    // frontSize = 20.0;
}

void MapDrawer::DrawMapPoints()
{
    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
    const vector<MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())
        return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);
    //glColor3f(1.0,1.0,1.0);

    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);
    //glColor3f(0.5,0.5,0.5);

    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        if((*sit)->isBad())
            continue;
        cv::Mat pos = (*sit)->GetWorldPos();
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));

    }
    glEnd();
}

void MapDrawer::DrawFrame()
{
    const vector<KeyFrame*> &vpKf = mpMap->GetAllKeyFrames();
    if(vpKf.empty()) return;

    // get the most recent reconstructed keyframe to texture
    KeyFrame* kfToTexture = NULL;
    KeyFrame* prevKf = NULL;
    for(size_t i = 0; i < vpKf.size();++i) {
        KeyFrame *kf = vpKf[i];
        kf->SetNotEraseDrawer();
        if (kf->isBad() || !kf->semidense_flag_ || !kf->interKF_depth_flag_) {
            kf->SetEraseDrawer();
            continue;
        }
        if (prevKf == NULL){
            kfToTexture = kf;
            prevKf = kf;
        } else if (kf->mnId > prevKf->mnId){
            kfToTexture = kf;
            prevKf->SetEraseDrawer();
            prevKf = kf;
        }
    }
    if (kfToTexture == NULL) return;


    cv::Size imSize = kfToTexture->rgb_.size();

    pangolin::GlTexture imageTexture(imSize.width, imSize.height, GL_RGB, false, 0, GL_BGR,
                                     GL_UNSIGNED_BYTE);

    imageTexture.Upload(kfToTexture->rgb_.data, GL_BGR, GL_UNSIGNED_BYTE);

    imageTexture.RenderToViewportFlipY();


    kfToTexture->SetEraseDrawer();
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            cv::Mat Twc = pKF->GetPoseInverse().t();

            glPushMatrix();

            glMultMatrixf(Twc.ptr<GLfloat>(0));

            glLineWidth(mKeyFrameLineWidth);

            // [EAO] created by objects.
            if(pKF->mbCreatedByObjs)
                glColor3f(1.0f,0.0f,0.0f);
            else
                glColor3f(0.0f,0.0f,1.0f);

            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();
        }
    }

    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            cv::Mat Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
                    cv::Mat Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                }
            }

            // Spanning tree
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
                cv::Mat Owp = pParent->GetCameraCenter();
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owp.at<float>(0),Owp.at<float>(1),Owp.at<float>(2));
            }

            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                cv::Mat Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owl.at<float>(0),Owl.at<float>(1),Owl.at<float>(2));
            }
        }

        glEnd();
    }
}

// BRIEF [EAO-SLAM] draw objects.
void MapDrawer::DrawObject(const bool QuadricObj,
                           const string &flag)
{
    const vector<Object_Map*> &vObjs = mpMap->GetObjects();

    vector<cv::Mat> object_cen;

    int id = -1;
    for(size_t i = 0; i < vObjs.size(); ++i)
    {
        Object_Map* Obj = vObjs[i];

        if((Obj->mObjectFrame.size() < 5) && (flag != "NA"))
            continue;

        if((Obj->mvpMapObjectMappoints.size() < 10) || (Obj->bBadErase == true))
        {
            continue;
        }

        id ++;

        // color.
        glColor3f(0.5,0,0.5);
        glLineWidth(2);

        // *************************************
        //    STEP 1. [EAO-SLAM] Draw cubes.   *
        // *************************************
        // draw cubes END ----------------------------------------------------------------------------

        // ****************************************
        //    STEP 2. [EAO-SLAM] Draw quadrics.   *
        // ****************************************
        if(QuadricObj)
        {
            // half axial length.
            float lenth = Obj->mCuboid3D.lenth/2;
            float width = Obj->mCuboid3D.width/2;
            float height = Obj->mCuboid3D.height/2;

            // tvmonitor, fixed scale, for better visulazation.
            if(Obj->mnClass == 62)
            {
                lenth = 0.13;
                width = 0.035;
                height = 0.08;
            }

            cv::Mat axe = cv::Mat::zeros(3,1,CV_32F);
            axe.at<float>(0) = lenth;
            axe.at<float>(1) = width;
            axe.at<float>(2) = height;

            // quadrcis pose.
            cv::Mat Twq = cv::Mat::zeros(4,4,CV_32F);
            Twq.at<float>(0, 0) = 1;
            Twq.at<float>(0, 1) = 0;
            Twq.at<float>(0, 2) = 0;
            Twq.at<float>(0, 3) = Obj->mCuboid3D.cuboidCenter[0];
            Twq.at<float>(1, 0) = 0;
            Twq.at<float>(1, 1) = 1;
            Twq.at<float>(1, 2) = 0;
            Twq.at<float>(1, 3) = Obj->mCuboid3D.cuboidCenter[1];
            Twq.at<float>(2, 0) = 0;
            Twq.at<float>(2, 1) = 0;
            Twq.at<float>(2, 2) = 1;
            Twq.at<float>(2, 3) = Obj->mCuboid3D.cuboidCenter[2];
            Twq.at<float>(3, 0) = 0;
            Twq.at<float>(3, 1) = 0;
            Twq.at<float>(3, 2) = 0;
            Twq.at<float>(3, 3) = 1;

            // create a quadric.
             //初始化二次曲面并创建一个指向二次曲面的指针
            GLUquadricObj *pObj = gluNewQuadric();
            cv::Mat Twq_t = Twq.t();

            // color
            cv::Scalar sc;
            sc = cv::Scalar(0, 255, 0);

            // front
            std::string str = class_names[Obj->mnClass];

            // case 1
            pangolin::GlText m_gltext = pangolin::GlFont::I().Text(str.c_str());
            m_gltext.Draw(
                (GLfloat)(Twq.at<float>(0, 3)),
                (GLfloat)(Twq.at<float>(1, 3)),
                (GLfloat)(Twq.at<float>(2, 3) + height/2.0));


            // case 2
            // // pangolin::GlFont *text_font = new pangolin::GlFont(frontPath, frontSize);
            // auto text_font = std::make_unique<pangolin::GlFont>(frontPath, frontSize);
            // text_font->Text(str.c_str()).Draw(
            //     (GLfloat)(Twq.at<float>(0, 3)),
            //     (GLfloat)(Twq.at<float>(1, 3) + width/2.0),
            //     (GLfloat)(Twq.at<float>(2, 3) + height/2.0)); //参数为xyz坐标

            // add to display list
            // glPushMatrix、glPopMatrix操作事实上就相当于栈里的入栈和出栈
            // 将本次须要运行的缩放、平移等操作放在glPushMatrix和glPopMatrix之间。
            // glPushMatrix()和glPopMatrix()的配对使用能够消除上一次的变换对本次变换的影响。
            // 使本次变换是以世界坐标系的原点为參考点进行。
            glPushMatrix();
            glMultMatrixf(Twq_t.ptr<GLfloat >(0));
            glScalef(
                    (GLfloat)(axe.at<float>(0,0)),
                    (GLfloat)(axe.at<float>(0,1)),
                    (GLfloat)(axe.at<float>(0,2))
                    );
            // 二次曲面绘制风格：GLU_LINE直线模拟
            gluQuadricDrawStyle(pObj, GLU_LINE);
            // 在二次曲面的表面创建平滑的法向量:GLU_NONE(否)
            gluQuadricNormals(pObj, GLU_NONE);
            // 采用立即模式绘制的数据
            glBegin(GL_COMPILE);
            // 指针，半径，纬线细分面，经线细分面（数字越大越平滑）
            gluSphere(pObj, 1., 15, 10);
            glEnd();
            glPopMatrix();

            // _______________________________ //
            // 外包围框
            glColor3f(1.0, 0, 0.0);
            glLineWidth(2);
            glPushMatrix();
            glMultMatrixf(Twq_t.ptr<GLfloat>(0));
            // draw object center.
            glPointSize(8);

            {
                glBegin(GL_LINES);

                glVertex3f(-lenth, -width, -height); // 1
                glVertex3f(lenth, -width, -height);  // 2

                glVertex3f(lenth, -width, -height); // 2
                glVertex3f(lenth, width, -height);  // 3

                glVertex3f(lenth, width, -height);  // 3
                glVertex3f(-lenth, width, -height); // 4

                glVertex3f(-lenth, width, -height);  // 4
                glVertex3f(-lenth, -width, -height); // 1

                glVertex3f(-lenth, -width, height); // 5
                glVertex3f(lenth, -width, height);  // 6

                glVertex3f(lenth, -width, height); // 6
                glVertex3f(lenth, width, height);  // 7

                glVertex3f(lenth, width, height);  // 7
                glVertex3f(-lenth, width, height); // 8

                glVertex3f(-lenth, width, height);  // 8
                glVertex3f(-lenth, -width, height); // 5

                glVertex3f(-lenth, -width, -height); // 1
                glVertex3f(-lenth, -width, height);  // 5

                glVertex3f(lenth, -width, -height); // 2
                glVertex3f(lenth, -width, height);  // 6

                glVertex3f(lenth, width, -height); // 3
                glVertex3f(lenth, width, height);  // 7

                glVertex3f(-lenth, width, -height); // 4
                glVertex3f(-lenth, width, height);  // 8

                glEnd();
            }

            {
                float axisLen = 1.0;
                glLineWidth(5);
                glColor3f(1.0, 0.0, 0.0); // red x
                glBegin(GL_LINES);
                glVertex3f(0.0, 0.0f, 0.0f);
                glVertex3f(lenth + axisLen, 0.0f, 0.0f);
                glEnd();

                glColor3f(0.0, 1.0, 0.0); // green y
                glBegin(GL_LINES);
                glVertex3f(0.0, 0.0f, 0.0f);
                glVertex3f(0.0, width + axisLen, 0.0f);

                glEnd();

                glColor3f(0.0, 0.0, 1.0); // blue z
                glBegin(GL_LINES);
                glVertex3f(0.0, 0.0f, 0.0f);
                glVertex3f(0.0, 0.0f, height + axisLen);

                glEnd();
            }

            glPopMatrix();

            // draw quadrics END ---------------------------------------------------------------------
        }
    }
} // draw objects END ----------------------------------------------------------------------------


void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}

void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    if(!mCameraPose.empty())
    {
        cv::Mat Rwc(3,3,CV_32F);
        cv::Mat twc(3,1,CV_32F);
        {
            unique_lock<mutex> lock(mMutexCamera);
            Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
            twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
        }

        M.m[0] = Rwc.at<float>(0,0);
        M.m[1] = Rwc.at<float>(1,0);
        M.m[2] = Rwc.at<float>(2,0);
        M.m[3]  = 0.0;

        M.m[4] = Rwc.at<float>(0,1);
        M.m[5] = Rwc.at<float>(1,1);
        M.m[6] = Rwc.at<float>(2,1);
        M.m[7]  = 0.0;

        M.m[8] = Rwc.at<float>(0,2);
        M.m[9] = Rwc.at<float>(1,2);
        M.m[10] = Rwc.at<float>(2,2);
        M.m[11]  = 0.0;

        M.m[12] = twc.at<float>(0);
        M.m[13] = twc.at<float>(1);
        M.m[14] = twc.at<float>(2);
        M.m[15]  = 1.0;
    }
    else
        M.SetIdentity();
}

void MapDrawer::DrawMapPlanes(){
    pcl::VoxelGrid<PointT> voxel;
    voxel.setLeafSize(0.05, 0.05, 0.05);

    const vector<long unsigned int> &vnRPs = mpMap->GetRemovedPlanes();
    const vector<MapPlane *> &vpMPs = mpMap->GetAllMapPlanes();

    if (vpMPs.empty())
        return;

    glPointSize(mPointSize / 2);
    glBegin(GL_POINTS);

    for (auto pMP : vpMPs)
    {
        if (pMP->isBad())
        {
            continue;
        }

        map<KeyFrame *, int> observations = pMP->GetObservations();
        float ir = pMP->mRed;
        float ig = pMP->mGreen;
        float ib = pMP->mBlue;
        float norm = sqrt(ir * ir + ig * ig + ib * ib);

        PointCloud::Ptr planeCloudPoints(new PointCloud);

        for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *frame = mit->first;
            int id = mit->second;
            if (id >= frame->mnRealPlaneNum)
            {
                continue;
            }
            Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(frame->GetPose());
            PointCloud::Ptr cloud(new PointCloud);
            pcl::transformPointCloud(frame->mvPlanePoints[id], *cloud, T.inverse().matrix());
            *planeCloudPoints += *cloud;
        }
        PointCloud::Ptr tmp(new PointCloud());
        voxel.setInputCloud(planeCloudPoints);
        voxel.filter(*tmp);

        // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color(tmp, ir, ig, ib);

        for (auto &p : tmp->points)
        {
            glColor3f(ir / norm, ig / norm, ib / norm);
            glVertex3f(p.x, p.y, p.z);
        }
    }

    glEnd();
}

void MapDrawer::DrawMapPlanesOld()
{
    const vector<MapPlane *> &vpMPs = mpMap->GetAllMapPlanes();
    if (vpMPs.empty())
        return;
    glPointSize(mPointSize / 2);
    glBegin(GL_POINTS);
    pcl::VoxelGrid<PointT> voxel;
    voxel.setLeafSize(0.02, 0.02, 0.02);
    for (auto pMP : vpMPs)
    {
        map<KeyFrame *, int> observations = pMP->GetObservations();
        float ir = pMP->mRed;
        float ig = pMP->mGreen;
        float ib = pMP->mBlue;
        float norm = sqrt(ir * ir + ig * ig + ib * ib);
        glColor3f(ir / norm, ig / norm, ib / norm);
        PointCloud::Ptr allCloudPoints(new PointCloud);
        for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *frame = mit->first;
            int id = mit->second;
            if (id >= frame->mnRealPlaneNum)
            {
                continue;
            }
            Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(frame->GetPose());
            PointCloud::Ptr cloud(new PointCloud);
            pcl::transformPointCloud(frame->mvPlanePoints[id], *cloud, T.inverse().matrix());
            *allCloudPoints += *cloud;
        }
        PointCloud::Ptr tmp(new PointCloud());
        voxel.setInputCloud(allCloudPoints);
        voxel.filter(*tmp);

        for (auto &p : tmp->points)
        {
            glColor3f(ir / norm, ig / norm, ib / norm);
            glVertex3f(p.x, p.y, p.z);
        }
    }
    glEnd();
}

} //namespace ORB_SLAM
