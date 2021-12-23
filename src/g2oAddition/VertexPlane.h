/*
 * @Author: Chen Jiahao
 * @Date: 2021-12-21 09:56:32
 * @LastEditors: Chen Jiahao
 * @LastEditTime: 2021-12-21 10:31:41
 * @Description: file content
 * @FilePath: /catkin_ws/src/EAO-SLAM/src/g2oAddition/VertexPlane.h
 */
//
// Created by fishmarch on 19-5-28.
//

#ifndef ORB_SLAM2_VERTEXPLANE_H
#define ORB_SLAM2_VERTEXPLANE_H

#include "core/base_vertex.h"
#include "core/hyper_graph_action.h"
#include "core/eigen_types.h"
#include "stuff/misc.h"
#include "Plane3D.h"


namespace g2o {
    class  VertexPlane : public BaseVertex<3, Plane3D>
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexPlane();

    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    virtual void setToOriginImpl() { _estimate = Plane3D(); }

    virtual void oplusImpl(const double* update_) {
        Eigen::Map<const Vector3D> update(update_);
        _estimate.oplus(update);
    }

    virtual bool setEstimateDataImpl(const double* est){
        Eigen::Map<const Vector4D> _est(est);
        _estimate.fromVector(_est);
        return true;
    }

    virtual bool getEstimateData(double* est) const{
        Eigen::Map<Vector4D> _est(est);
        _est = _estimate.toVector();
        return true;
    }

    virtual int estimateDimension() const {
        return 4;
    }

};
}


#endif //ORB_SLAM2_VERTEXPLANE_H
