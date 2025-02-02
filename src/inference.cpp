//
// Created by gx on 24-1-27.
//
#include "../include/inference.hpp"
#include "../include/process_plugins.h"
#include <ie_core.hpp>
static void using_once();
static void Initialize()
{
    std::cout<<"initialize"<<std::endl;

    //编译模型
    compiled_model = core.compile_model(model_path,"CPU"); //使用 OpenVINO 的 Core 对象编译模型
//    core.set_property({ { CONFIG_KEY(CPU_BIND_THREAD), "NO" } });
//    compiled_model= core.compile_model(model_path, "CPU",ov::inference_num_threads(16));

    //创建一个推理请求对象
    infer_request = compiled_model.create_infer_request();
    //获取编译模型的输入端口
    input_port = compiled_model.input();

}

static void using_once() //初始化模型
{
    std::call_once(flag, Initialize); //call_once：确保某个初始化操作只被执行一次

}
//我们发现神经网络推理的点有时顺序会有误，推断是数据集问题，此函数用于纠正点的顺序
static void sort_keypoints(cv::Point2f keypoints[4]) {
    // Sort points based on their y-coordinates (ascending)
    std::sort(keypoints, keypoints + 4, [](const cv::Point& a, const cv::Point& b) {
        return a.y < b.y;
    });

    // Top points will be the first two, bottom points will be the last two
    cv::Point top_points[2] = { keypoints[0], keypoints[1] };
    cv::Point bottom_points[2] = { keypoints[2], keypoints[3] };

    // Sort the top points by their x-coordinates to distinguish left and right
    std::sort(top_points, top_points + 2, [](const cv::Point& a, const cv::Point& b) {
        return a.x < b.x;
    });

    // Sort the bottom points by their x-coordinates to distinguish left and right
    std::sort(bottom_points, bottom_points + 2, [](const cv::Point& a, const cv::Point& b) {
        return a.x < b.x;
    });

    // Assign sorted points back to the keypoints array
    keypoints[0] = top_points[0];     // top-left
    keypoints[1] = bottom_points[0];  // bottom-left
    keypoints[2] = bottom_points[1];  // bottom-right
    keypoints[3] = top_points[1];     // top-right
}
dataImg preprocessImage(const cv::Mat& img, cv::Size new_shape, cv::Scalar color) //图像预处理
{
    //1.获取当前图像尺寸 Get current shape [height, width]
    cv::Size shape = img.size();

    //2.计算缩放比例 Scale ratio (new / old)
    double r = std::min((double)new_shape.height / shape.height, (double)new_shape.width / shape.width);

    //3.计算新尺寸 Compute padding
    cv::Size new_unpad = cv::Size(int(round(shape.width * r)), int(round(shape.height * r))); //round():四舍五入
    int dw = (new_shape.width - new_unpad.width) ; //计算需要在左右或上下添加多少像素来达到目标尺寸
    int dh = (new_shape.height - new_unpad.height) ;

    //4.调整图像大小 Resize image if necessary
    cv::Mat resized_img;
    if (shape != new_unpad) {
        cv::resize(img, resized_img, new_unpad, 0, 0, cv::INTER_LINEAR); //如果新的尺寸与原始尺寸不同，则使用双线性插值法调整图像大小
    } else {
        resized_img = img;
    }

    //5.添加边界填充 Add border/padding
    int top = 0;
    int bottom = dh;
    int left = 0 ;
    int right = dw;
    cv::Mat bordered_img;
    cv::copyMakeBorder(resized_img, bordered_img, top, bottom, left, right, cv::BORDER_CONSTANT, color); //copyMakeBorder():给图像添加边界

    //6.创建blob并归一化
    dataImg data;
    float scale = 1.0/(std::min( XML_SIZE*1.0/ img.rows,  XML_SIZE*1.0 / img.cols)); //计算缩放比例
    cv :: Mat blob = cv::dnn::blobFromImage(bordered_img, 1.0 / 255.0, cv::Size( XML_SIZE, XML_SIZE), cv::Scalar(), true); //将处理后的图像转换为一个四维的blob，可以直接输入到深度模型中进行推理
                                                                                                                                                                    //1.0 / 255.0:归一化因子，将图像的像素值从0-255缩放到0-1
                                                                                                                                                                    //cv::Size( XML_SIZE, XML_SIZE):目标blob的宽高尺寸
                                                                                                                                                                    //cv::Scalar()：平均减去值，默认情况下是零，表示不做平均减去操作
                                                                                                                                                                    //true: 表示是否交换颜色通道，通常是从BGR（OpenCV默认格式）转换为RGB（许多深度模型的输入格式）

    //7.返回结果
    data.blob = blob;
    data.input = img;
    data.scale = scale;
    return data;
}
#if 0           // openMP
dataImg preprocessImage(cv::Mat imgInput) {
    int col = imgInput.cols;
    int row = imgInput.rows;
    int _max = std::max(col, row);

    Mat result = Mat::zeros(_max, _max, CV_8UC3);
    imgInput.copyTo(result(Rect(0, 0, col, row)));

    dataImg data;
    float scale;

#pragma omp parallel sections
    {
#pragma omp section
        {
            // 计算图像的缩放比例
            scale = result.size[0] / 640;
        }

#pragma omp section
        {
            Mat blob = blobFromImage(result, 1.0 / 255.0, Size(640,640), Scalar(), true); // 图像像素归一化
            data.blob = blob;
        }
    }

    data.scale = scale;
    data.input = imgInput;

    return data;
}
#endif
std::vector<OneArmor> startInferAndNMS(dataImg img_data, double score_threshold_,double nms_threshold_){ //深度推理
    //1.初始化模型
    using_once();
    //2.创建输入张量
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), img_data.blob.ptr(0)); //Tensor: OpenVINO 中表示张量（多维数组）的类
                                                                                                                             //input_port.get_element_type():获取输入端口(输入数据)的元素类型
                                                                                                                             //input_port.get_shape()获取输入端口(输入数据)的形状(维度)
                                                                                                                             //img_data.blob.ptr(0)：获取指向输入数据(图像)的指针
    //3.设置输入张量 Set input tensor for model with one input
    infer_request.set_input_tensor(input_tensor);
    //4.开始推理 -------- Start inference --------
    infer_request.infer();
   // std::cout<<"Run into startInferAndNMS.." << std::endl;
    //5.获取推理结果  -------- Get the inference result --------
    auto output = infer_request.get_output_tensor(0); //获取第一个输出张量
    auto output_shape = output.get_shape(); //获取输出张量的形状

    //6.后处理结果 -------- Postprocess the result --------
    float *data = output.data<float>(); //获取输出张量的数据指针
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data); //将输出数据转换为OpenCV的矩阵格式
    transpose(output_buffer, output_buffer); //对输出矩阵进行转置，变为 [8400, 14]
    //7.定义返回结果容器
    std::vector<OneArmor> qualifiedArmors;

    //8.遍历输出结果
    for (int cls=4 ; cls < 6; ++cls) { //遍历类别ID为4和5的结果
        Armor SingleData;
        for (int i = 0; i < output_buffer.rows; i++) { //遍历每一行输出结果
            float class_score = output_buffer.at<float>(i, cls); //获取当前类别的得分
            //保证当前对应的板子信息匹配
            float max_class_score = 0.0;
            for (int j = 4; j < 6; j++) { //遍历类别4和5的得分，找到最大得分
                if(max_class_score < output_buffer.at<float>(i, j)){
                    max_class_score = output_buffer.at<float>(i, j);
                }
            }
            if (class_score != max_class_score){ //如果当前类别得分不是最大得分，则跳过
                continue;
            }

            if (class_score > score_threshold_) { //如果类别得分大于阈值(置信度)，则继续处理

                SingleData.class_scores.push_back(class_score);

                //提取边界框信息
                float cx = output_buffer.at<float>(i, 0); //获取中心点坐标
                float cy = output_buffer.at<float>(i, 1);
                float w = output_buffer.at<float>(i, 2); //获取宽，高
                float h = output_buffer.at<float>(i, 3);

                // Get the box
                //由于yolov8-pose推理的框不一定满足传统识别取灯条角点所需，可以适当扩大ROI区域
                int left = int((cx - 0.5 * w-2) * img_data.scale);
                int top = int((cy - 0.5 * h-2) * img_data.scale);
                int width = int(w *1.2* img_data.scale);
                int height = int(h *1.2* img_data.scale);

                //提取关键点信息 Get the keypoints
                std::vector<float> keypoints;
                cv::Mat kpts = output_buffer.row(i).colRange(6,14 ); //获取关键点数据
                for (int i = 0; i < 4; i++) { //遍历4个关键点，提取坐标并缩放
                    float x = kpts.at<float>(0, i * 2 + 0) * img_data.scale;
                    float y = kpts.at<float>(0, i * 2 + 1) * img_data.scale;

                    keypoints.push_back(x);
                    keypoints.push_back(y);

                }
                //存储边界框和关键点
                SingleData.boxes.push_back(cv::Rect(left, top, width, height));
                SingleData.objects_keypoints.push_back(keypoints);

            }

        }
        //记录类别ID
        SingleData.class_ids = cls - 4;
       //9.NMS(非极大值抑制)处理
        std::vector<int> indices;
        cv::dnn::NMSBoxes(SingleData.boxes, SingleData.class_scores, score_threshold_, nms_threshold_, indices); //应用NMS去除重叠的边界框

      //  std::cout << "indices: " << indices.size() << std::endl;
        //10.整理结果
        for(auto i:indices) //遍历NMS后的索引
        {
            OneArmor armor; //创建一个新的装甲板对象

            armor.box = SingleData.boxes[i]; //设置边界框
            armor.class_scores = SingleData.class_scores[i]; //设置类别得分
            armor.class_ids = SingleData.class_ids; //设置类别ID

            for (int j = 0; j < 4; j++) { //遍历4个关键点，设置关键点坐标
                int x = std::clamp(int(SingleData.objects_keypoints[i][j * 2 + 0]), 0, img_data.input.cols); //clamp():将一个值限制在给定的最小值和最大值之间
                int y = std::clamp(int(SingleData.objects_keypoints[i][j * 2 + 1]), 0, img_data.input.rows);
                armor.objects_keypoints[j] = cv::Point(x, y);


            }
            sort_keypoints(armor.objects_keypoints); //对关键点进行排序
            qualifiedArmors.push_back(armor); //存储装甲板对象
        }
    }
    return qualifiedArmors;
}

void show_points_result(cv::Mat& img,std::vector<OneArmor> armors_data ) {
  std::map<int , std::string>colors ;
  colors[1] = "red";
  colors[0] = "blue";
  for (auto i: armors_data) {
    // cv::rectangle(img, i.box, cv::Scalar(0, 255, 0), 2);
    for(int j=0;j<4;j++){
      cv::line(img, i.objects_keypoints[j], i.objects_keypoints[(j + 1) % 4], cv::Scalar(0, 255, 0), 1);
    }
    for(int j=0;j<4;j++){
      if(j == 0){
        cv::circle(img, i.objects_keypoints[j], 2, cv::Scalar(0, 0, 255), -1);
      }else if(j==1){     cv::circle(img, i.objects_keypoints[j], 2, cv::Scalar(0, 0, 0), -1);}
      else if(j==2){     cv::circle(img, i.objects_keypoints[j], 2, cv::Scalar(0, 255, 0), -1);}
      else if(j==3){     cv::circle(img, i.objects_keypoints[j], 2, cv::Scalar(255, 0, 0), -1);}
    }
    //cv::rectangle(img, i.box, cv::Scalar(0, 255, 0), 2);
    cv::putText(img, colors[i.class_ids], i.objects_keypoints[0], cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
  }
}
void show_box_result(cv::Mat& img,std::vector<OneArmor> armors_data ) {
  for(auto i: armors_data){
    cv::rectangle(img, i.box, cv::Scalar(0, 255, 0), 2);
  }
}
void show_number_result(cv::Mat& img,std::vector<OneArmor> armors_data){
  for(auto i: armors_data){
    cv::putText(img, i.number, i.objects_keypoints[3], cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
  }
}
