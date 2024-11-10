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

// 定义回调函数处理接收到的图像
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;

    cv::TickMeter tm;
    tm.start();
    dataImg blob = preprocessImage(frame);
    auto armors_data = startInferAndNMS(blob);
    auto armors_data_ = classify(frame, armors_data);
    tm.stop();
    std::cout << "time cost: " << tm.getTimeMilli() << "ms" << std::endl;
    show_number_result(frame, armors_data_);
    show_points_result(frame, armors_data_);

    // 显示结果
    cv::imshow("result", frame);
    cv::waitKey(1);  // 必须有等待按键，否则窗口会卡住
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "video_detector_node");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("camera/image_raw", 1, imageCallback);

  ros::spin();
  return 0;
}
