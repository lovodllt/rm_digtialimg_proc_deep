//
// Created by gx on 24-1-27.
//

#include <common.h>
#ifndef V8_INFERENCE_H
#include <algorithm>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>    //opencv header file
#include <openvino/openvino.hpp> //openvino header file
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <rm_digtialimg_proc_deep/InferenceConfig.h>

/*#define model_path                                                             \
  "/home/haomo/catkin_ws/src/rm_digtialimg_proc_deep/model/"                   \
  "mobilenetv3_last_int_all_new/last.xml"*/

#define model_path                                                             \
  "/home/lovod/rm_code/src/rm_visplugin/rm_digtialimg_proc_deep/model/"                   \
  "mobilenetv3_last_int_all_new/last.xml"

#define XML_SIZE 416

/*
 * 整体重写的方向为：
 * 1、成为一个独立头文件
 * 2、方便其他文件调用此文件各种函数（安全就不考虑了）
 * 3、分为推理+输出+可视化 ，三部分
 * 4、提供更多的宏定义，便于调整参数（此处就无所谓滥用宏了）
 */

//******************推理**************
//模型加载（懒狗不想写类，全用静态函数了）
struct dataImg{
    float scale; //缩放比例
    cv::Mat blob; //处理后的图像blob
    cv::Mat input; //原始输入图像
};
struct Armor {
  std::vector<float> class_scores;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<float>> objects_keypoints;
  int class_ids;
};
struct OneArmor {
  float class_scores;
  cv::Rect box;
  cv::Point2f objects_keypoints[4];
  int class_ids; // color
  cv::Mat number_img;
  std::string number;
};

static std::once_flag flag;
static ov::Core core;
static ov::CompiledModel compiled_model;
static ov::InferRequest infer_request;
static ov::Output<const ov::Node> input_port;
extern rm_digtialimg_proc_deep::TargetColor target_color_;
inline rm_digtialimg_proc_deep::TargetColor target_color_;

dataImg preprocessImage(const cv::Mat &img,
                        cv::Size new_shape = cv::Size(XML_SIZE, XML_SIZE),
                        cv::Scalar color = cv::Scalar(114, 114,
                                                      114)); // 图片预处理
std::vector<OneArmor> startInferAndNMS(dataImg img_data, double score_threshold_,double nms_threshold_); // 开始推理并返回结果

void show_points_result(cv::Mat &img, std::vector<OneArmor> armors_data);
void show_box_result(cv::Mat &img, std::vector<OneArmor> armors_data);
void show_number_result(cv::Mat &img, std::vector<OneArmor> armors_data);

#define V8_INFERENCE_H

#endif //V8_INFERENCE_H
