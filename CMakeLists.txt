# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2019 Intel Corporation. All Rights Reserved.
#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(weight-detection)

find_package(realsense2 REQUIRED) 
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} weight-detection.cpp)
#add_executable(${PROJECT_NAME} weight-detection-2cameras.cpp)
target_link_libraries(${PROJECT_NAME} realsense2::realsense2 ${OpenCV_LIBS})

