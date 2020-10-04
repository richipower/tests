//
//  ObjectDetector_YOLO.hpp
//  tests
//
//  Created by Ricardo Rodriguez on 14.11.18.
//  Copyright © 2018 Ricardo Rodriguez. All rights reserved.
//

#ifndef ObjectDetector_YOLO_hpp
#define ObjectDetector_YOLO_hpp

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

struct YoloBox
{
    double score;
    cv::Rect box;
    std::string name;
	cv::Scalar color;

    YoloBox(double _score, cv::Rect _box, std::string _name, cv::Scalar _color)
    : score(_score), box(_box), name(_name), color(_color)
    {}

    YoloBox(){}
};

void draw_label(cv::Mat& _img, cv::Rect& _box, std::string& _class_name, cv::Scalar& _class_color, double& _score);

class ObjectDetector_YOLO
{
public:
    ObjectDetector_YOLO();
    ~ObjectDetector_YOLO();

    void setup(std::string _cfg, std::string _names, std::string _weights, cv::Size _netSize = cv::Size(416,416));
    void set_nms_threshold(float _threshold);
	std::vector<YoloBox> detectObjects(cv::Mat& _inputMat, float _threshold);


    
private:
    std::vector<cv::String> get_outputsNames(cv::dnn::Net& _net);
    float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };
    cv::Scalar get_color(int _class, int _numClasses);
    void validateBoxes(std::vector<YoloBox>& _inputBoxes, cv::Size _inputSize);

    static bool compareBoxByScore(const YoloBox& _box1, const YoloBox& _box2);
    float IoU(YoloBox& _box1, YoloBox& _box2);
    std::vector<YoloBox> NMS(std::vector<YoloBox>& _boxes, float _threshold);


    cv::dnn::Net yolo_net;
    cv::Size netSize;
    std::ifstream classNamesFile;
    std::vector<std::string> classNames;
	std::vector<cv::Scalar> classColors;
    float nms_threshold;

};


#endif /* ObjectDetector_YOLO_hpp */
