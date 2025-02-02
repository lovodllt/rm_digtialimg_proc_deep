cmake_minimum_required(VERSION 3.0.2)
project(rm_digtialimg_proc_deep)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)


set(CMAKE_CXX_STANDARD 20)
set(ngraph_DIR "/opt/intel/openvino_2023.3.0/runtime/cmake")
set(InferenceEngine_DIR "/opt/intel/openvino_2023.3.0/runtime/cmake")
set(OpenVINO_DIR "/opt/intel/openvino_2023.3.0/runtime/cmake")
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake")

set(INC_DIR "/opt/intel/openvino_2023.3.0/runtime/include")
set(LINK_DIR "/opt/intel/openvino_2023.3.0/runtime/lib/intel64")
set(OpenVINO_DIR_TEST "/opt/intel/openvino_2023.3.0/runtime/lib/intel64/libopenvino.so")

add_compile_options(-std=c++17)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=1)
set(CMAKE_CXX_FLAGS"-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        roscpp
        tf
        rm_vision
        message_generation
        dynamic_reconfigure
        std_msgs
        geometry_msgs
        visualization_msgs
        image_transport
        cv_bridge
        image_geometry
        nodelet
        tf2
        tf2_geometry_msgs
        rm_msgs
)
find_package(OpenCV 4 REQUIRED)
#find_package(OpenVINO REQUIRED)

generate_dynamic_reconfigure_options(
        cfg/Inference.cfg
)

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES rm_digtialimg_proc_deep
        CATKIN_DEPENDS
        rm_vision
        roscpp
        tf
        message_generation
        dynamic_reconfigure
        std_msgs
        geometry_msgs
        visualization_msgs
        image_transport
        cv_bridge
        image_geometry
        nodelet
        tf2
        tf2_geometry_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${INC_DIR}
        ${PROJECT_SOURCE_DIR}/include/
        #OpenVINO推理引擎的头文件
        "/opt/intel/openvino_2023.3.0/runtime/include/ie/"
        "/opt/intel/openvino_2023/runtime/include/ngraph/"
        "/opt/intel/openvino_2023.3.0/runtime/include/openvino/"

)

FILE(GLOB_RECURSE SOURCE_FILES "src/*.cpp")

add_library(rm_digtialimg_proc_deep
        src/video_detect_test.cpp
        src/inference.cpp
        src/number_classifier.cpp
)

link_directories(
        ${LINK_DIR}
)

link_libraries(libopenvino.so)


#add_executable(video_detector_node
#        src/video_detect_test.cpp
#        src/tradition.cpp
#        src/inference.cpp
#        src/number_classifier.cpp
#
#)
target_link_libraries(
        rm_digtialimg_proc_deep
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
        ${OpenVINO_DIR_TEST}
#        ${ngraph_LIBRARIES}
)
