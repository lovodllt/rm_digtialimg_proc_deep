
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

// STL
#include <vector>

#include "../include/inference.hpp"
#include "../include/number_classifier.hpp"

static void using_once();
static void Initialize()
{
    net_ =  cv::dnn::readNetFromONNX(number_classifier_model_path_);
}

static void using_once()
{
    std::call_once(flag_, Initialize);

}
//static cv::Mat perform_opening(const cv::Mat& input_image, int kernel_size) {
//    // Check if the input image is valid
//    if (input_image.empty()) {
//        std::cerr << "Input image is empty!" << std::endl;
//        return cv::Mat();
//    }
//
//    // Create a structuring element (kernel) for the morphological operation
//    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
//
//    // Perform the opening operation
//    cv::Mat opened_image;
//    cv::morphologyEx(input_image, opened_image, cv::MORPH_OPEN, kernel);
//
//    return opened_image;
//}

void extractNumbers(const cv::Mat & src, std::vector<OneArmor> & armors) //用于提取装甲板上的数字
{
    // static int num = 0;
    //常量定义
    // Light length in image
    const int light_length = 12; //装甲板灯条长度
    // Image size after warp
    const int warp_height = 28; //透视变换后的图像高度
    const int small_armor_width = 32; //小装甲板的宽度
    const int large_armor_width = 54; //大装甲板的宽度
    // Number ROI size
    const cv::Size roi_size(20, 28); //数字区域的大小

    //遍历每个装甲板
    for (auto & armor : armors) {
        //透视变换 Warp perspective transform
        cv::Point2f lights_vertices[4] = {armor.objects_keypoints[1], armor.objects_keypoints[0], armor.objects_keypoints[3], armor.objects_keypoints[2]}; //lights_vertices：装甲板上四个关键点的坐标

        const int top_light_y = (warp_height - light_length) / 2 - 1; //计算灯条变换后在图像中的上下边界
        const int bottom_light_y = top_light_y + light_length;
        const int warp_width = small_armor_width;//全按小装甲板处理
        cv::Point2f target_vertices[4] = { //目标图像中四个顶点的坐标
                cv::Point(0, bottom_light_y),
                cv::Point(0, top_light_y),
                cv::Point(warp_width - 1, top_light_y),
                cv::Point(warp_width - 1, bottom_light_y),
        };
        cv::Mat number_image;
        auto rotation_matrix = cv::getPerspectiveTransform(lights_vertices, target_vertices); //计算透视变换矩阵
        cv::warpPerspective(src, number_image, rotation_matrix, cv::Size(warp_width, warp_height)); //将源图像应用透视变换矩阵转换为目标图像

        //提取数字区域 Get ROI
        number_image =
                number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));

        //二值化处理 Binarize
        std::vector<cv::Mat> channels(3);
        cv::split(number_image, channels); //将图像拆分为三个通道

        cv::cvtColor(number_image, number_image, cv::COLOR_RGB2GRAY); //将RGB图像转换为灰度图像
//保存数字图案20*28
        // cv::imwrite("/home/gx/rm_classifier_training-main/补_/"+std::to_string(num++)+".jpg", number_image);
        cv::threshold(number_image, number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU); //使用Otsu方法进行二值化处理
        // cv::Mat number_img_ = perform_opening(number_image,2);

        //存储数字图像
        armor.number_img = number_image;

    }
}

std::vector<OneArmor> classify(cv::Mat src,std::vector<OneArmor> & armors) //对装甲板上的数字进行分类
{
    //提取出数字
    extractNumbers(src,armors);

    //初始化模型
    using_once();
    std::vector<OneArmor> armors_data_;

    //遍历每个装甲板
    for (auto & armor : armors) {

        //图像预处理
        cv::Mat image = armor.number_img.clone();

        // Normalize
        image = image / 255.0; //归一化

        //创建Blob  Create blob from image
        cv::Mat blob;
        cv::dnn::blobFromImage(image, blob); //将图像转换为blob格式

        //设置输入并前向传播 Set the input blob for the neural network
        net_.setInput(blob); //将 Blob 设置为神经网络的输入
        // Forward pass the image blob through the model
        cv::Mat outputs = net_.forward(); //前向传播图像，得到模型的输出

        //Softmax处理
        float max_prob = *std::max_element(outputs.begin<float>(), outputs.end<float>()); //找到输出中的最大值max_prob
        cv::Mat softmax_prob;
        cv::exp(outputs - max_prob, softmax_prob); //计算 Softmax 概率
        float sum = static_cast<float>(cv::sum(softmax_prob)[0]); //计算概率和，并将概率归一化
        softmax_prob /= sum;

        //获取分类结果
        double confidence;
        cv::Point class_id_point;
        minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point); //minMaxLoc函数找到最大概率的位置class_id_point
        int label_id = class_id_point.x; //label_id:分类结果的标签索引

        //更新装甲板信息
        armor.number = class_names_[label_id];

        //过滤负样本
        for(auto i:armors){
            if(i.number != "negative"){ //滤掉number为"negative"的装甲板
                armors_data_.push_back(i);
            }
        }

    }
    return armors_data_;
}


