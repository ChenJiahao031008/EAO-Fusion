//
// Created by fishmarch on 19-5-28.
//

#ifndef ORB_SLAM2_PLANE3D_H
#define ORB_SLAM2_PLANE3D_H

#include "stuff/misc.h"
#include "core/eigen_types.h"
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace g2o {
    class  Plane3D {
            public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            friend Plane3D operator*(const Isometry3D& t, const Plane3D& plane);

            Plane3D(){
                Vector4D v;
                v << 1., 0., 0., -1.;
                fromVector(v);
            }

            Plane3D(const Vector4D& v){
                fromVector(v);
            }


            inline Vector4D toVector() const {
                return _coeffs;
            }

            inline const Vector4D& coeffs() const {return _coeffs;}

            inline void fromVector(const Vector4D& coeffs_) {
                _coeffs=coeffs_;
                normalize(_coeffs);
            }

            static double azimuth(const Vector3D& v) {
                return std::atan2(v(1),v(0));
            }

            static  double elevation(const Vector3D& v) {
                // 本质上就是 arcsin(v(2))
                return std::atan2(v(2), v.head<2>().norm());
            }

            double distance() const {
                return -_coeffs(3);
            }

            Vector3D normal() const {
                return _coeffs.head<3>();
            }


            static Matrix3D rotation(const Vector3D& v)  {
                double _azimuth = azimuth(v);
                double _elevation = elevation(v);
                Eigen::AngleAxisd azimuth_v(_azimuth, Vector3D::UnitZ());
                Eigen::AngleAxisd elevation_v(- _elevation, Vector3D::UnitY());
                return (azimuth_v * elevation_v).toRotationMatrix();
            }

            inline void oplus(const Vector3D& v){
                //construct a normal from azimuth and evelation;
                double _azimuth=v[0];
                double _elevation=v[1];
                double s=std::sin(_elevation), c=std::cos(_elevation);
                // [cos\phi * cos\theta, cos\phi * sin\theta, \sin\phi]
                Vector3D n (c*std::cos(_azimuth), c*std::sin(_azimuth), s) ;

                // rotate the normal
                Matrix3D R = rotation(normal());
                double d = distance()+v[2];
                _coeffs.head<3>() =  R * n;
                // 注意：d取负
                _coeffs(3) = -d;
                normalize(_coeffs);
            }

            inline Vector3D ominus(const Plane3D& plane){
                //construct the rotation that would bring the plane normal in (1 0 0)
                // 法向量转换为旋转向量 再转置，与顶点的平面旋转部分相乘
                Matrix3D R = rotation(normal()).transpose();
                Vector3D n = R * plane.normal();
                // 距离直接相减即可
                double d=distance()-plane.distance();
                return Vector3D(azimuth(n), elevation(n), d);
            }

            inline Vector2D ominus_ver(const Plane3D& plane){
                // construct the rotation that would bring the plane normal in (1 0 0)
                // 确定垂直轴，旋转90度
                Vector3D v = normal().cross(plane.normal());
                Eigen::AngleAxisd ver(M_PI/2, v/v.norm());
                Vector3D b = ver * normal();

                Matrix3D R = rotation(b).transpose();
                Vector3D n = R * plane.normal();
                return Vector2D(azimuth(n), elevation(n));
            }

            inline Vector2D ominus_par(const Plane3D& plane){
                // construct the rotation that would bring the plane normal in (1 0 0)
                Vector3D nor = normal();
                if(plane.normal().dot(nor) < 0)
                    nor = -nor;
                // 只计算旋转向量的差值
                Matrix3D R = rotation(nor).transpose();
                Vector3D n = R * plane.normal();

                return Vector2D(azimuth(n), elevation(n));
            }
            //protected:

            static inline void normalize(Vector4D& coeffs) {
                double n=coeffs.head<3>().norm();
                coeffs = coeffs * (1./n);
                if(coeffs(3)<0.0)
                    coeffs = -coeffs;
            }

            Vector4D _coeffs;
    };
    // input t : transform matrix applying to the point
    // 由于平面的投影并不是简单的相乘，因此需要对乘号进行重载
    inline Plane3D operator*(const Isometry3D& t, const Plane3D& plane){
        Vector4D v = plane._coeffs;
        Vector4D v2;
        Matrix3D R = t.rotation();
        // 其实这部分本质就是在做H^{-T}的工作，只不过拆解开了而已
        // 法向量部分做一个旋转即可
        v2.head<3>() = R * v.head<3>();
        v2(3) = v(3) - t.translation().dot(v2.head<3>());
        if(v2(3) < 0.0)
            v2 = -v2;
        return Plane3D(v2);
    };
}


#endif //ORB_SLAM2_PLANE3D_H
