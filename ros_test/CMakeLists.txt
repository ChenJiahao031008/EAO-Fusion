cmake_minimum_required(VERSION 3.10.2)
project(ros_evo)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE release)
ENDIF()

# yolox
ADD_DEFINITIONS("-w -O2 -pthread")
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native  -Wno-unused-function  -Wno-return-type")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native  -Wno-unused-function  -Wno-return-type")

# set carv needed flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__STDC_CONSTANT_MACROS -DGL_GLEXT_PROTOTYPES -D_LINUX -D_REENTRANT -DCGAL_USE_F2C -DCGAL_DISABLE_ROUNDING_MATH_CHECK")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_CONSTANT_MACROS -DGL_GLEXT_PROTOTYPES -D_LINUX -D_REENTRANT -DCGAL_USE_F2C -DCGAL_DISABLE_ROUNDING_MATH_CHECK")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake_modules)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()
message(STATUS "Using opencv version ${OpenCV_VERSION}")
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

# # glog
# include(../cmake_modules/glog.cmake)

# find openmp
find_package(OpenMP)
if (OPENMP_FOUND)
    message("-- OpenMP found")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Find CGAL
set(CMAKE_BUILD_TYPE_BACKUP ${CMAKE_BUILD_TYPE})
set(CMAKE_BUILD_TYPE "Release")
find_package(CGAL REQUIRED COMPONENTS Core)
include( ${CGAL_USE_FILE} )
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE_BACKUP})

# Find Boost
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

# Find PCL
find_package( PCL 1.7 REQUIRED)
add_definitions( ${PCL_DEFINITIONS} )
link_directories( ${PCL_LIBRARY_DIRS} )

# BLAS and LAPACK
find_package(LAPACK REQUIRED)
link_directories(${LAPACK_LIBRARY_DIR})

## CUDA及TensorRT相关
find_package(CUDA REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/include)
# cuda
include_directories(/usr/local/cuda-11.1/include)
link_directories(/usr/local/cuda-11.1/lib64)
# cudnn
include_directories(/home/chen/cuda/include)
link_directories(/home/chen/cuda/lib64)
# tensorrt
include_directories(/home/chen/DeepLearning/settings/TensorRT-7.2.1.6/include)
link_directories(/home/chen/DeepLearning/settings/TensorRT-7.2.1.6/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


find_package(
    catkin REQUIRED COMPONENTS
    roscpp
    rospy
    rosbag
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    message_generation
    std_srvs
    cv_bridge
    image_transport
    roslib
)


## Generate services in the 'srv' folder
add_service_files(FILES
  saveOdometry.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
)

include_directories(
    include ${catkin_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/../include
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/../Thirdparty/DBoW2/DBoW2
    ${PROJECT_SOURCE_DIR}/../Thirdparty/DBoW2/
    ${PROJECT_SOURCE_DIR}/../Thirdparty/g2o/g2o
    ${PROJECT_SOURCE_DIR}/../src/g2oAddition
    ${PROJECT_SOURCE_DIR}/../include/PEAC
    ${PROJECT_SOURCE_DIR}/../src/DepthFilter
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../lib)

add_library(${PROJECT_NAME}_lib SHARED
    # yolox
    ../src/YOLOX.cc
    # ros
    src/message_flow.cc
    src/imu_subscriber.cc
    src/image_subscriber.cc
    # base
    ../src/System.cc
    ../src/Tracking.cc
    ../src/LocalMapping.cc
    ../src/LoopClosing.cc
    ../src/ORBextractor.cc
    ../src/ORBmatcher.cc
    ../src/FrameDrawer.cc
    ../src/Converter.cc
    ../src/MapPoint.cc
    ../src/KeyFrame.cc
    ../src/Map.cc
    ../src/MapDrawer.cc
    ../src/Optimizer.cc
    ../src/PnPsolver.cc
    ../src/Frame.cc
    ../src/KeyFrameDatabase.cc
    ../src/Sim3Solver.cc
    ../src/Initializer.cc
    ../src/Viewer.cc
    ../src/Object.cc      # NOTE [EAO-SLAM]

    # plane
    ../src/MapPlane.cc
    ../src/g2oAddition/EdgePlane.cc
    ../src/g2oAddition/VertexPlane.cc
    # depth filter
    ../src/DepthFilter/JBF.cc
    ../src/DepthFilter/Kernel.cc
)


target_link_libraries(${PROJECT_NAME}_lib
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/../Thirdparty/DBoW2/lib/libDBoW2.so
    ${PROJECT_SOURCE_DIR}/../Thirdparty/g2o/lib/libg2o.so
    boost_system boost_filesystem boost_serialization
    lapack blas
    nvinfer
    cudart
    # dw
)


add_executable(ros_rgbd_imu app/ros_rgbd_imu.cpp)
add_dependencies(ros_rgbd_imu ${catkin_EXPORTED_TARGETS})
target_link_libraries(ros_rgbd_imu ${catkin_LIBRARIES} ${PROJECT_NAME}_lib)

add_executable(ros_rgbd app/ros_rgbd.cpp)
add_dependencies(ros_rgbd ${catkin_EXPORTED_TARGETS})
target_link_libraries(ros_rgbd ${catkin_LIBRARIES} ${PROJECT_NAME}_lib)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../tools)
add_executable(bin_vocabulary
 ../tools/bin_vocabulary.cc)
target_link_libraries(bin_vocabulary ${PROJECT_NAME}_lib)



#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html

