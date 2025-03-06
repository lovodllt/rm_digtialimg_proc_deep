/*
#include "../include/inference.hpp"
#include "../include/tradition.hpp"
#include "../include/number_classifier.hpp"
#define VIDEO_PATH "../video_test/test.mp4"

int main(int argc, char* argv[])
{


    cv::VideoCapture capture(VIDEO_PATH);

    if (!capture.isOpened())
    {
        std::cout << "无法读取视频：" << argv[1] << std::endl;
        return -1;
    }
    // 读取视频帧，使用Mat类型的frame存储返回的帧
    cv::Mat frame;

    int fps = 0;
    while (true){


        capture >> frame;
        if (frame.empty())
        {
            std::cout << "视频读取完毕" << std::endl;
            break;
        }
        cv::TickMeter tm;
        tm.start();
        dataImg blob = preprocessImage(frame);
      //  std::cout<<"start infer"<<std::endl;
        auto armors_data = startInferAndNMS(blob);
        //传统处理获得角点以提高角点精度（实测非必须）
        //auto armors_data_ = tradition(frame, armors_data);
        //获得数字
        auto armors_data_ =  classify(frame,armors_data);
        tm.stop();
        std::cout << "time cost: " << tm.getTimeMilli() << "ms" << std::endl;
        show_number_result(frame, armors_data_);
        show_points_result(frame, armors_data_);
       // show_box_result(frame, armors_data_);

        // 按下ESC键退出
        int k = cv::waitKey(10);
        if (k == 27)
        {
            std:: cout << "退出" << std::endl;
            break;
        }
        cv::imshow("result", frame);
       // cv::destroyAllWindows();

    }
    return 0;
}
*/

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include "../include/inference.hpp"
#include "../include/tradition.hpp"
#include "../include/number_classifier.hpp"
#include "nodelet/nodelet.h"
#include "pluginlib/class_list_macros.h"
#include "../include/process_plugins.h"
#include <ros/callback_queue.h>

using namespace cv;
using namespace std;


namespace rm_digtialimg_proc_deep {

void Processor::onInit()
{
  ros::NodeHandle& nh = getMTPrivateNodeHandle();
  static ros::CallbackQueue my_queue;
  nh.setCallbackQueue(&my_queue);
  initialize(nh);
  my_thread_ = std::thread([]() {
    ros::SingleThreadedSpinner spinner;
    spinner.spin(&my_queue);
  });

  //cv::namedWindow("result", cv::WINDOW_NORMAL); // 确保窗口名称唯一且可见
}

void Processor::initialize(ros::NodeHandle &nh) {
  nh_ = ros::NodeHandle(nh, "digtialimg_proc_deep");
  auto inference_params_init = [this, &nh]() { // lambda表达式，
                                                           // this：捕获当前对象的所有成员。
                                                           // &nh：按引用捕获 nh 变量
    ROS_INFO("reading inference param");

    score_threshold_ = nh.param("score_threshold", decltype(score_threshold_){});
    nms_threshold_ = nh.param("nms_threshold", decltype(nms_threshold_){});

    ROS_INFO("inference params reading done");
  };

  inference_params_init(); // 调用对应的lambda函数

  inference_cfg_srv_ = new dynamic_reconfigure::Server<rm_digtialimg_proc_deep::InferenceConfig>(ros::NodeHandle(nh_, "inference_condition")); // 创建动态配置服务器 inference_cfg_cb__cfg_srv_
  inference_cfg_cb_ = boost::bind(&Processor::inferenceconfigCB, this, _1, _2); // 将回调函数绑定并存入 inference_cfg_cb__cfg_cb_
  inference_cfg_srv_->setCallback(inference_cfg_cb_); // 设置回调函数，将之前绑定好的回调函数 armor_cfg_cb_ 设置到动态配置服务器 inference_cfg_cb__cfg_srv_ 上

  it_ = make_shared<image_transport::ImageTransport>(nh_);
  image_pub_ = it_->advertise("debug_image", 1);
  cam_sub_ = it_->subscribeCamera("/hk_camera/image_raw", 10, &Processor::callback, this);
  tf_buffer_ = new tf2_ros::Buffer(ros::Duration(10));
  tf_listener_ = new tf2_ros::TransformListener(*tf_buffer_);
  target_pub_ = it_->advertise("/processor/result_msg", 10);
}

void Processor::inferenceconfigCB(rm_digtialimg_proc_deep::InferenceConfig &config, uint32_t level)
{
  score_threshold_ = config.score_threshold;
  nms_threshold_ = config.nms_threshold;
  target_color_ = static_cast<TargetColor>(config.target_color);
}

void Processor::callback(const sensor_msgs::ImageConstPtr& img, const sensor_msgs::CameraInfoConstPtr& info)
{
  try {
    
    cv::Mat frame =
        cv_bridge::toCvShare(img, "bgr8")->image; // 将ros图像消息转为opencv格式

    cv::TickMeter tm; // tm:用于测量后续操作的时间
    tm.start();       // 开始计时
    dataImg blob = preprocessImage(frame);            // 图像预处理
    auto armors_data = startInferAndNMS(blob, score_threshold_, nms_threshold_); // 深度推理
    auto armors_data_ = classify(frame, armors_data); // 数字分类
    tm.stop();                                        // 停止计时
    std::cout << "time cost: " << tm.getTimeMilli() << "ms"
              << std::endl; // 输出整个处理过程所花费的时间
    show_number_result(frame, armors_data_);
    show_points_result(frame, armors_data_);

    sensor_msgs::ImagePtr msg_out = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
    target_pub_.publish(msg_out);


  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }
}

void Processor::imageProcess(cv_bridge::CvImagePtr &cv_image) {}
void Processor::paramReconfig() {}
void Processor::findArmor() {}
rm_vision::ProcessorInterface::Object Processor::getObj() {
  return rm_vision::ProcessorInterface::Object();
}
void Processor::putObj() {}
void Processor::draw() {}

}

PLUGINLIB_EXPORT_CLASS(rm_digtialimg_proc_deep::Processor, nodelet::Nodelet)