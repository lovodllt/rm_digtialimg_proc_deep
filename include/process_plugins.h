//
// Created by yichenghe on 2021/11/13.
//

#pragma once

#include <iostream>
#include <vector>
#include <dirent.h>
#include <cmath>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <XmlRpcValue.h>
// #include <inference_engine.hpp>
#include <rm_vision/vision_base/processor_interface.h>
#include <dynamic_reconfigure/server.h>
//#include <rm_digtialimg_proc_test/ArmorConfig.h>
//#include <rm_digtialimg_proc_test/PreprocessConfig.h>
//#include <rm_digtialimg_proc_test/DrawConfig.h>
//#include <rm_digtialimg_proc_test/MakedatasetConfig.h>
#include "../include/config/ArmorConfig.h"
#include "../include/config/PreprocessConfig.h"
#include "../include/config/DrawConfig.h"
#include "../include/config/MakedatasetConfig.h"
#include <mutex>
#include <thread>
#include <nodelet/nodelet.h>
#include <pluginlib/class_loader.h>
#include <rm_msgs/TargetDetectionArray.h>
#include <rm_msgs/TargetDetection.h>
#include <rm_msgs/StatusChange.h>
#include <rm_msgs/TrackData.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "common.h"

using cv::Mat;
using std::vector;
namespace rm_digtialimg_proc_test
{
class Bar
{
public:
  cv::RotatedRect bar_rect_;
  double length_len_;  /// long side
  double width_len_;   /// short side
  double lw_ratio_;
  cv::Point2f points_[4];
  cv::Point2f center_point_;
  double angle_;  /// the angle between short side of bar and x axis
  double pixel_contained_ratio_;
  ArmorColor color_;
  bool is_clockwise_;
  int used_num_;
  vector<cv::Point> contour_;

  Bar(cv::RotatedRect bar_rect, const vector<cv::Point>& contour, ArmorColor color) : bar_rect_(std::move(bar_rect))
  {
    length_len_ = std::max(bar_rect_.size.width, bar_rect_.size.height);
    width_len_ = std::min(bar_rect_.size.width, bar_rect_.size.height);
    pixel_contained_ratio_ = cv::contourArea(contour) / bar_rect_.size.area();
    lw_ratio_ = length_len_ / width_len_;
    bar_rect_.points(points_);
    center_point_ = bar_rect_.center;
    contour_ = contour;

    if (length_len_ == bar_rect_.size.width)
    {
      angle_ = bar_rect_.angle + 90.0;
      is_clockwise_ = true;
    }
    else
    {
      angle_ = bar_rect_.angle;  ////////////////
      is_clockwise_ = false;
    }
    color_ = color;
  };
};

class Armor
{
public:
  Bar* bar_left_;
  Bar* bar_right_;
  vector<cv::Point2d> bars_4points_;        /// bl, tl, tr, br ; center point
  vector<cv::Point2d> bars_inter_4points_;  /// bl, tl, tr, br ; inter point
  vector<cv::Point2d> bars_PCA_4points_;
  float warp_white_ratio_;
  double length;
  double width;
  cv::Point2d center_;
  double lw_rate_;
  int id_;
  double confidence_;
  double negative_confidence_;
  double area_;
  double parallel_dist_;
  bool is_large_armor_;

  Armor(Bar& bar_left, Bar& bat_right)
  {
    bar_left_ = &bar_left;
    bar_right_ = &bat_right;
    getBarsPoints();
    getArmorArea();
    getCenterPoint();
    getParallelDist();
    id_ = 0;
    confidence_ = 0;
    is_large_armor_ = false;
  };

  void getBarsPoints()
  {
    if (bar_left_->is_clockwise_)
    {
      bars_4points_.emplace_back((bar_left_->points_[0] + bar_left_->points_[1]) * 0.5);
      bars_4points_.emplace_back((bar_left_->points_[2] + bar_left_->points_[3]) * 0.5);
      bars_inter_4points_.emplace_back(bar_left_->points_[0]);
      bars_inter_4points_.emplace_back(bar_left_->points_[3]);
    }
    else
    {
      bars_4points_.emplace_back((bar_left_->points_[3] + bar_left_->points_[0]) * 0.5);
      bars_4points_.emplace_back((bar_left_->points_[2] + bar_left_->points_[1]) * 0.5);
      bars_inter_4points_.emplace_back(bar_left_->points_[3]);
      bars_inter_4points_.emplace_back(bar_left_->points_[2]);
    }

    if (bar_right_->is_clockwise_)
    {
      bars_4points_.emplace_back((bar_right_->points_[3] + bar_right_->points_[2]) * 0.5);
      bars_4points_.emplace_back((bar_right_->points_[1] + bar_right_->points_[0]) * 0.5);
      bars_inter_4points_.emplace_back(bar_right_->points_[2]);
      bars_inter_4points_.emplace_back(bar_right_->points_[1]);
    }
    else
    {
      bars_4points_.emplace_back((bar_right_->points_[2] + bar_right_->points_[1]) * 0.5);
      bars_4points_.emplace_back((bar_right_->points_[3] + bar_right_->points_[0]) * 0.5);
      bars_inter_4points_.emplace_back(bar_right_->points_[1]);
      bars_inter_4points_.emplace_back(bar_right_->points_[0]);
    }
  };

  void getArmorArea()
  {
    double len1 =
        pow(pow(bars_4points_[0].x - bars_4points_[1].x, 2) + pow(bars_4points_[0].y - bars_4points_[1].y, 2), 0.5);
    double len2 =
        pow(pow(bars_4points_[1].x - bars_4points_[2].x, 2) + pow(bars_4points_[1].y - bars_4points_[2].y, 2), 0.5);
    double len3 =
        pow(pow(bars_4points_[0].x - bars_4points_[2].x, 2) + pow(bars_4points_[0].y - bars_4points_[2].y, 2), 0.5);
    double len4 =
        pow(pow(bars_4points_[2].x - bars_4points_[3].x, 2) + pow(bars_4points_[2].y - bars_4points_[3].y, 2), 0.5);
    double len5 =
        pow(pow(bars_4points_[0].x - bars_4points_[3].x, 2) + pow(bars_4points_[0].y - bars_4points_[3].y, 2), 0.5);

    double half_cir1 = (len1 + len2 + len3) / 2.0;
    double half_cir2 = (len3 + len4 + len5) / 2.0;

    double area1 = pow((half_cir1 * (half_cir1 - len1) * (half_cir1 - len2) * (half_cir1 - len3)), 0.5);
    double area2 = pow((half_cir2 * (half_cir2 - len3) * (half_cir2 - len4) * (half_cir2 - len5)), 0.5);

    area_ = area1 + area2;
  };

  void getCenterPoint()
  {
    double p1_x = bars_4points_[0].x;
    double p1_y = bars_4points_[0].y;
    double p2_x = bars_4points_[1].x;
    double p2_y = bars_4points_[1].y;
    double p3_x = bars_4points_[2].x;
    double p3_y = bars_4points_[2].y;
    double p4_x = bars_4points_[3].x;
    double p4_y = bars_4points_[3].y;

    double line1_k = (p1_y - p3_y) / (p1_x - p3_x + 0.000000001);
    double line2_k = (p2_y - p4_y) / (p2_x - p4_x + 0.000000001);
    double line1_b = ((p1_y + p3_y) - line1_k * (p1_x + p3_x)) / 2;
    double line2_b = ((p2_y + p4_y) - line2_k * (p2_x + p4_x)) / 2;

    double cross_point_x = (line2_b - line1_b) / (line1_k - line2_k + 0.000000001);
    double cross_point_y = ((line1_k + line2_k) * cross_point_x + line1_b + line2_b) * 0.5;

    center_.x = int(cross_point_x);
    center_.y = int(cross_point_y);
  }

  void getParallelDist()
  {
    double p1_x = bars_4points_[0].x;
    double p1_y = bars_4points_[0].y;
    double p2_x = bars_4points_[1].x;
    double p2_y = bars_4points_[1].y;
    double p3_x = bars_4points_[2].x;
    double p3_y = bars_4points_[2].y;
    double p4_x = bars_4points_[3].x;
    double p4_y = bars_4points_[3].y;

    double line1_k = (p1_y - p2_y) / (p1_x - p2_x + 0.000000001);
    double line2_k = (p3_y - p4_y) / (p3_x - p4_x + 0.000000001);
    double line1_b = ((p1_y + p2_y) - line1_k * (p1_x + p2_x)) / 2;
    double line2_b = ((p3_y + p4_y) - line2_k * (p3_x + p4_x)) / 2;

    double dist = pow(pow(bar_left_->center_point_.x - bar_right_->center_point_.x, 2) +
                          pow(bar_left_->center_point_.y - bar_right_->center_point_.y, 2),
                      0.5);
    double vertical_dist1 = fabs(line1_k * bar_right_->center_point_.x - bar_right_->center_point_.y + line1_b) /
                            pow(pow(line1_k, 2) + 1, 0.5);
    double parallel_dist1 = pow(pow(dist, 2) - pow(vertical_dist1, 2), 0.5);

    double vertical_dist2 = fabs(line2_k * bar_left_->center_point_.x - bar_left_->center_point_.y + line2_b) /
                            pow(pow(line2_k, 2) + 1, 0.5);
    double parallel_dist2 = pow(pow(dist, 2) - pow(vertical_dist2, 2), 0.5);

    parallel_dist_ = (parallel_dist1 + parallel_dist2) * 0.5;
  }
};

/**
 * @brief
 *
 */
class Processor : public rm_vision::ProcessorInterface, public nodelet::Nodelet
{
public:
  Processor()
  {
  }
  ~Processor() override
  {
    if (this->my_thread_.joinable())
      my_thread_.join();
  }
  void initialize(ros::NodeHandle& nh) override;

  void hsvToBinary();

  void bgrToBinary();

  Mat setElement();

  void imageProcess(cv_bridge::CvImagePtr& cv_image) override;

  ArmorColor getBarColor(const cv::RotatedRect& rect);

  void findBars();

  void paramReconfig() override;
  bool changeStatusCB(rm_msgs::StatusChange::Request& change, rm_msgs::StatusChange::Response& res);
  ros::ServiceServer status_change_srv_;

  bool isLargeArmor(const Armor& armor);

  bool isValidBar(const Bar& bar);

  bool isValidArmor(Bar& left_bar, Bar& right_bar);

  bool warp(const vector<cv::Point2d>& points, bool is_large, Mat& dst);

  void findArmor() override;

  void armorPrograming(const vector<vector<int>>& armor_programme);

  vector<cv::Point> calcPCAPoints(vector<cv::Point>& pts);

  ///////////////////////////////////////////////////////

  void cnnInfer(Mat& src);

  vector<double> recalibrateScores(const vector<double>& activate_vector);

  template <typename T>
  std::vector<int> argsort(const std::vector<T>& array);

  vector<double> weibullWscore(const vector<double>& distance, const vector<double>& alpha_weights);

  vector<double> computeDistance(const vector<double>& activate_vector);
  ////////////////////////////////////////////////////////

  Object getObj() override;

  void putObj() override;

  std::mutex obj_locker_;

  ////////////////////////////////////////////////////////

  void draw() override;

  void drawBars(Mat& image);

  void drawArmors(Mat& image);

  void drawArmorsVertexes(Mat& image);

  bool drawWarp();

  void makeDigitalDataset(Armor armor);
  int g_num = 0;

  dynamic_reconfigure::Server<ArmorConfig>* armor_cfg_srv_;
  dynamic_reconfigure::Server<PreprocessConfig>* preprocess_cfg_srv_;
  dynamic_reconfigure::Server<DrawConfig>* draw_cfg_srv_;
  dynamic_reconfigure::Server<MakedatasetConfig>* make_dataset_cfg_srv_;
  dynamic_reconfigure::Server<ArmorConfig>::CallbackType armor_cfg_cb_;
  dynamic_reconfigure::Server<PreprocessConfig>::CallbackType preprocess_cfg_cb_;
  dynamic_reconfigure::Server<DrawConfig>::CallbackType draw_cfg_cb_;
  dynamic_reconfigure::Server<MakedatasetConfig>::CallbackType make_dataset_cfg_cb_;

  void armorconfigCB(ArmorConfig& config, uint32_t level);
  bool armor_dynamic_reconfig_initialized_ = false;

  void drawconfigCB(DrawConfig& config, uint32_t level);

  void preProcessconfigCB(PreprocessConfig& config, uint32_t level);
  bool pre_process_dynamic_reconfig_initialized_ = false;

  void datasetconfigCB(MakedatasetConfig& config, uint32_t level);

  void onInit() override;

  // 定义回调函数处理接收到的图像
  void callback2(const sensor_msgs::ImageConstPtr& msg);
  void callback2(const sensor_msgs::ImageConstPtr& img, const sensor_msgs::CameraInfoConstPtr& info);

private:
//  rm_msgs::TargetDetectionArray target_array_;
//  ros::Publisher target_pub_;
//  ros::Subscriber track_sub_;
//  ros::Subscriber detection_sub_;
//  ros::Subscriber compute_sub_;
//  ros::NodeHandle nh_;
  std::shared_ptr<image_transport::ImageTransport> it_;
  image_transport::CameraSubscriber cam_sub_;

  image_transport::Publisher target_pub_;

//  void callback(const sensor_msgs::ImageConstPtr& img, const sensor_msgs::CameraInfoConstPtr& info)
//  {
//    if (!target_is_armor_)
//    {
//      //      ROS_INFO("not armor");
//      return;
//    }
//    camera_info_ = info;
//    target_array_.header = info->header;
//    boost::shared_ptr<cv_bridge::CvImage> temp =
//        boost::const_pointer_cast<cv_bridge::CvImage>(cv_bridge::toCvShare(img, "bgr8"));
//    imageProcess(temp);
//    findArmor();
//    draw();
//    for (auto& target : target_array_.detections)
//    {
//      target.pose.position.x = info->roi.x_offset;
//      target.pose.position.y = info->roi.y_offset;
//    }
//    target_array_.is_red = target_is_red_;
//    target_pub_.publish(target_array_);
//  }


  void trackCB(const rm_msgs::TrackData track_data)
  {
    all_points_.clear();
    double yaw = track_data.yaw, r1 = track_data.radius_1, r2 = track_data.radius_2;
    double xc = track_data.position.x, yc = track_data.position.y, zc = track_data.position.z;
    double dz = track_data.dz;
    int a_n = track_data.armors_num;
    geometry_msgs::PointStamped p_a;
    double r = 0;
    bool is_current_pair = true;
    // 请求变换
    geometry_msgs::TransformStamped transformStamped = tf_buffer_->lookupTransform(
        "camera_optical_frame", track_data.header.frame_id, track_data.header.stamp, ros::Duration(1));
    for (int i = 0; i < 4; i++)
    {
      double tmp_yaw = yaw + i * (2 * M_PI / a_n);
      if (a_n == 4)
      {
        r = is_current_pair ? r1 : r2;
        p_a.point.z = zc + (is_current_pair ? 0 : dz);
        is_current_pair = !is_current_pair;
      }
      else
      {
        r = r1;
        p_a.point.z = zc;
      }
      p_a.point.x = xc - r * cos(tmp_yaw);
      p_a.point.y = yc - r * sin(tmp_yaw);
      p_a.header = track_data.header;

      try
      {
        // 应用变换到点上
        tf2::doTransform(p_a, p_a, transformStamped);
      }
      catch (tf2::TransformException& ex)
      {
        ROS_WARN("%s", ex.what());
        return;
      }

      all_points_.emplace_back(p_a);
    }
    p_a.point.x = xc;
    p_a.point.y = yc;
    p_a.point.z = zc;

    try
    {
      // 应用变换到点上
      tf2::doTransform(p_a, p_a, transformStamped);
    }
    catch (tf2::TransformException& ex)
    {
      ROS_WARN("%s", ex.what());
      return;
    }

    all_points_.emplace_back(p_a);
  }

  void computeCB(const rm_msgs::TrackData track_data)
  {
    compute_point_.point = track_data.position;
    try
    {
      // 请求变换
      geometry_msgs::TransformStamped transformStamped = tf_buffer_->lookupTransform(
          "camera_optical_frame", track_data.header.frame_id, compute_point_.header.stamp, ros::Duration(1));
      // 应用变换到点上
      tf2::doTransform(compute_point_, compute_point_, transformStamped);
    }
    catch (tf2::TransformException& ex)
    {
      ROS_WARN("%s", ex.what());
      return;
    }
  }

  void detectionCB(const rm_msgs::TargetDetectionArrayConstPtr& msg)
  {
    if (!msg->detections.empty())
    {  // 确保安全
       //                      ROS_INFO("test");
      detection_first_ = msg->detections.front();
      auto q = detection_first_.pose.orientation;
      double sinThetaOver2 = sqrt(1 - q.w * q.w);

      // 当sin(theta/2)接近于0时，即无旋转或全旋转，处理为特殊情况
      if (sinThetaOver2 > 0.0001)
      {
        double theta = 2 * acos(q.w);  // 旋转角度
        r_vec_(0) = q.x / sinThetaOver2 * theta;
        r_vec_(1) = q.y / sinThetaOver2 * theta;
        r_vec_(2) = q.z / sinThetaOver2 * theta;
      }
      else
      {
        // 当四元数几乎或完全表示无旋转时，旋转向量为0
        r_vec_(0) = r_vec_(1) = r_vec_(2) = 0;
      }

      t_vec_(0) = detection_first_.pose.position.x;
      t_vec_(1) = detection_first_.pose.position.y;
      t_vec_(2) = detection_first_.pose.position.z;
    }
  };

  vector<geometry_msgs::PointStamped> all_points_;
  geometry_msgs::PointStamped compute_point_;
  sensor_msgs::CameraInfoConstPtr camera_info_{};
  rm_msgs::TargetDetection detection_first_;
  cv::Mat_<double> r_vec_ = cv::Mat_<double>(3, 1), t_vec_ = cv::Mat_<double>(3, 1);
  // transform
  tf2_ros::TransformListener* tf_listener_;
  tf2_ros::Buffer* tf_buffer_;

  cv::Mat_<uchar> resizeAndPadding(cv::Mat& src);

  std::thread my_thread_;
  image_transport::Publisher image_pub_;

  bool target_is_armor_ = true;
  int target_option_{};
  int preprocess_method_{};
  int target_is_red_{};
  bool is_bar_debug_{};
  bool is_armor_debug_{};
  bool is_classfy_debug_{};

  /// HSV
  int red_h_min_low_{};
  int red_h_max_low_{};
  int red_h_min_high_{};
  int red_h_max_high_{};
  int red_s_min_{};
  int red_s_max_{};
  int red_v_min_{};
  int red_v_max_{};
  int blue_h_min_{};
  int blue_h_max_{};
  int blue_s_min_{};
  int blue_s_max_{};
  int blue_v_min_{};
  int blue_v_max_{};

  /// RGB(single channel)
  int binary_thresh_{};

  /// morphology
  int morph_type_{};
  int binary_element_{};

  /// bar morphology
  vector<vector<cv::Point>> contours_{};

  /// bar compute
  // int bar_br_thresh_{};
  int select_bar_{};
  int max_angle_{};
  float max_lw_ratio_{};
  float min_lw_ratio_{};
  double min_pixel_contained_ratio_{};
  double max_bars_ratio_{};

  /// armor compute
  double min_bars_distance_{};
  double max_bars_distance_{};
  double max_bars_angle_{};
  double max_bars_y_dis_{};
  float warp_white_ratio_{};
  bool select_by_last_{};

  /// armor warp
  double large_armor_ratio_{};
  vector<cv::Point2d> warp_reference_;
  bool gamma_{};
  cv::Mat look_up_table_ = cv::Mat::ones(1, 256, CV_8U);
  double contrast_alpha_{};
  double contrast_beta_{};
  double gamma_y_{};
  int warp_thresh_{};
  bool rotate_{};
  /// warp : 32 * 28
  int warp_height_;
  int warp_width_;
  /// cut img (roi)
  int roi_height_{};
  int roi_width_{};
  double top_light_y_{};
  double bottom_light_y_{};
  double bar_length_in_warp_{};
  bool is_large_armor_;
  bool is_large_armor_store_{};
  int armor_temp_[6]{};

  /// id classification
  std::string xml_path_{};
  std::string bin_path_{};
  //  InferenceEngine::ExecutableNetwork network_{};
  //  InferenceEngine::CNNNetwork cnn_network_{};
  //  InferenceEngine::InferRequest infer_request_{};
  //  std::string input_name_{};
  //  InferenceEngine::OutputsDataMap output_info_{};
  std::pair<int, float> result_{};
  double negative_confidence_{};
  std::vector<double> expand_ratio_{};
  float id_confidence_{};
  std::vector<int> input_shape_{};
  bool use_id_cls_{};
  float min_id_white_ratio_{};
  float max_id_white_ratio_{};
  double firstnet_score_;
  vector<double> softmax_score_;
  /// fc
  int fc_result_;
  int fcRecognition(Mat& image);
  cv::dnn::Net net_;
  int class_num_;
  /// weibull model
  std::vector<std::vector<double>> weibull_model_{};
  std::vector<std::vector<double>> weibull_mean_{};

  ///
  int image_x_center_{};
  int image_y_center_{};

  /// draw
  DrawImage draw_type_{};
  int line_width_{};

  vector<Bar> bars_{};
  vector<Armor> armors_{};
  vector<Armor> last_frame_armors_{};

  vector<vector<cv::Point2d>> points_{};
  vector<int> labels_{};
  vector<float> probs_{};
  Object object_{};

  Mat raw_image_{};
  Mat binary_image_{};
  Mat morpro_image_{};
  Mat warp_image_{};
  /// make dataset
  bool make_dataset_{};  ////////////////////
  bool collected_completely_ = false;
  int image_num_{};
  int image_index_ = 0;  // range 0~image_num_ * 9
  int image_id_{};       /////////////////
  bool dataset_split_{};
  bool continute_{};
  std::string dataset_path_{};
  std::map<int, std::pair<int, std::string>> last_img_index_{};  // (image_id, (last img index, last img path))
};

}  // namespace rm_digtialimg_proc
