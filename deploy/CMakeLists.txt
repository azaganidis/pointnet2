cmake_minimum_required(VERSION 2.8.3)
project(ros_pointnet LANGUAGES CXX CUDA)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

##set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ")
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import sys;print(sys.path)"
    OUTPUT_VARIABLE PYTHON_SITE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "'" "" PYTHON_SITE ${PYTHON_SITE})
string(REPLACE "[" "" PYTHON_SITE ${PYTHON_SITE})
string(REPLACE "]" "" PYTHON_SITE ${PYTHON_SITE})
string(REPLACE ", " ";" PYTHON_SITE ${PYTHON_SITE})
set(TENSORFLOW_PATH "")
FOREACH(f ${PYTHON_SITE})
    if(EXISTS "${f}/tensorflow")
        LIST(APPEND TENSORFLOW_PATH "${f}/tensorflow")
    endif(EXISTS "${f}/tensorflow")
endforeach(f)
set(TENSORFLOW_INCLUDE_DIRS
    "${TENSORFLOW_PATH}/include;${TENSORFLOW_PATH}/include/external/nsync/public")
message(STATUS "Tensorflow path: ${TENSORFLOW_PATH}")
find_package(Boost REQUIRED COMPONENTS program_options system serialization)
find_package(CUDA REQUIRED)
include_directories(
    ${TENSORFLOW_INCLUDE_DIRS}
  "../tf_ops"
  	${Boost_INCLUDE_DIRS}
  	${PCL_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
  #	${EIGEN3_INCLUDE_DIR}
  # ${ROS_REQUIRED_DIRS}
  #	${PCL_INCLUDE_DIRS})
  )
set(LIB_SRCS 
    "../tf_ops/3d_interpolation/tf_interpolate.cpp"
    "../tf_ops/grouping/tf_grouping.cpp"
    "../tf_ops/sampling/tf_sampling.cpp"
    )
set(CUDA_OPS_SRCS 
    "../tf_ops/grouping/tf_grouping_g.cu"
    "../tf_ops/sampling/tf_sampling_g.cu"
    )
link_directories ( ${Boost_LIBRARY_DIRS}  ${TENSORFLOW_PATH})

find_package(roscpp REQUIRED)
find_package(rosconsole REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(VTK REQUIRED)

set(ROS_REQUIRED_LIBS
    ${roscpp_LIBRARIES}
    ${rosconsole_LIBRARIES}
    ${tf2_ros_LIBRARIES}
    ${pcl_ros_LIBRARIES}
    ${VTK_LIBRARIES}
    )

set(ROS_REQUIRED_DIRS
    ${roscpp_INCLUDE_DIRS}
    ${rosconsole_INCLUDE_DIRS}
    ${tf2_ros_INCLUDE_DIRS}
    ${pcl_ros_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    )

add_library(cuda_ops ${CUDA_OPS_SRCS})
add_library(tf_ops ${LIB_SRCS})
target_link_libraries(tf_ops cuda_ops)
add_executable(deploy  deploy.cpp)
target_link_libraries(deploy 
        ${ROS_REQUIRED_LIBS}
        ${Boost_LIBRARIES}
        tf_ops
    tensorflow tensorflow_framework)
