#include "../include/inference.hpp"
#include "../include/tradition.hpp"
#include "../include/number_classifier.hpp"
#define VIDEO_PATH "../video_test/test.mp4"

void show_points_result(cv::Mat& img,std::vector<OneArmor> armors_data ) { //展示装甲板的关键点

    //定义颜色映射
    std::map<int , std::string>colors ;
    colors[1] = "red";
    colors[0] = "blue";

    //遍历每个装甲板
    for (auto i: armors_data) {
       // cv::rectangle(img, i.box, cv::Scalar(0, 255, 0), 2);
       for(int j=0;j<4;j++){ //绘制装甲板轮廓
           cv::line(img, i.objects_keypoints[j], i.objects_keypoints[(j + 1) % 4], cv::Scalar(0, 255, 0), 1);
       }
       for(int j=0;j<4;j++){ //绘制关键点
           if(j == 0){
               cv::circle(img, i.objects_keypoints[j], 2, cv::Scalar(0, 0, 255), -1); //第一个关键点(j==0)用红色(cv::Scalar(0,0,255)
           }else if(j==1){     cv::circle(img, i.objects_keypoints[j], 2, cv::Scalar(0, 0, 0), -1);} //第二个关键点(j==1)用黑色(cv::Scalar(0,0,0))
           else if(j==2){     cv::circle(img, i.objects_keypoints[j], 2, cv::Scalar(0, 255, 0), -1);} //第三个关键点(j==2)用绿色(cv::Scalar(0,255,0))
           else if(j==3){     cv::circle(img, i.objects_keypoints[j], 2, cv::Scalar(255, 0, 0), -1);} //第四个关键点(j==3)用蓝色(cv::Scalar(255,0,0))
       }
       //cv::rectangle(img, i.box, cv::Scalar(0, 255, 0), 2);

       //绘制类别标签
       cv::putText(img, colors[i.class_ids], i.objects_keypoints[0], cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
}
void show_box_result(cv::Mat& img,std::vector<OneArmor> armors_data ) {
    for(auto i: armors_data){
        cv::rectangle(img, i.box, cv::Scalar(0, 255, 0), 2);
    }
}
void show_number_result(cv::Mat& img,std::vector<OneArmor> armors_data){ //展示识别出来的数字
    for(auto i: armors_data){
        cv::putText(img, i.number, i.objects_keypoints[3], cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
}
int main(int argc, char* argv[])
{

    //视频捕获初始化
    cv::VideoCapture capture(VIDEO_PATH); // VideoCapture:用于从视频文件、摄像头或其他视频流中捕获视频

    //检查视频是否成功打开
    if (!capture.isOpened())
    {
        std::cout << "无法读取视频：" << argv[1] << std::endl; //argv[1]:显示尝试打开的视频文件名
        return -1;
    }

    // 读取视频帧，使用Mat类型的frame存储返回的帧
    cv::Mat frame;

    int fps = 0;
    while (true){


        capture >> frame; //从视频流中读取下一帧图像(capture中存储视频，一帧一帧写入frame中)
        if (frame.empty())
        {
            std::cout << "视频读取完毕" << std::endl;
            break;
        }
        //时间测量
        cv::TickMeter tm; //TickMeter:测量代码块的执行时间。它通过记录 CPU 的时钟周期来实现高精度的时间测量
        tm.start(); //开始计时

        //图像预处理与推理
        dataImg blob = preprocessImage(frame); //对输入的图像帧进行预处理
      //  std::cout<<"start infer"<<std::endl;
        auto armors_data = startInferAndNMS(blob); //深度推理
        //传统处理获得角点以提高角点精度（实测非必须）
        //auto armors_data_ = tradition(frame, armors_data);

        //后处理与结果展示
        auto armors_data_ =  classify(frame,armors_data); //获得数字并进行数字分类
        tm.stop(); //停止计时
        std::cout << "time cost: " << tm.getTimeMilli() << "ms" << std::endl; //getTimeMilli():获取并输出整个处理过程的时间消耗（以毫秒为单位）
        show_number_result(frame, armors_data_); //绘制数字
        show_points_result(frame, armors_data_); //绘制装甲板的框和关键点
       // show_box_result(frame, armors_data_);

        //用户交互与窗口管理
        int k = cv::waitKey(10);
        if (k == 27) // 按下ESC键退出(ASCII码为27)
        {
            std:: cout << "退出" << std::endl;
            break;
        }
        cv::imshow("result", frame);
       // cv::destroyAllWindows();

    }
    return 0;
}
