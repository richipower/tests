//
//  ObjectDetector_YOLO.cpp
//  tests
//
//  Created by Ricardo Rodriguez on 14.11.18.
//  Copyright © 2018 Ricardo Rodriguez. All rights reserved.
//

#include "ObjectDetector_YOLO.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;


ObjectDetector_YOLO::ObjectDetector_YOLO()
{
}

ObjectDetector_YOLO::~ObjectDetector_YOLO()
{
}


void ObjectDetector_YOLO::setup(std::string _cfg, std::string _names, std::string _weights, cv::Size _netSize)
{
    this->yolo_net = readNet(_cfg, _weights);
    this->nms_threshold = 0.4;
    this->netSize = _netSize;

    this->classNamesFile.open(_names);
    std::string className = "";
    if(classNamesFile.is_open())
    {
        while (getline(classNamesFile, className))
            this->classNames.push_back(className);

        int numClasses = (int)classNames.size();
        for(int i=0; i<numClasses;i++)
            this->classColors.push_back(get_color(i, numClasses));
    }
    else
    {
        cout << "Class Names File could not be opened" << endl;
    }

    // - Close File
    classNamesFile.close();

}


void ObjectDetector_YOLO::set_nms_threshold(float _threshold)
{
    this->nms_threshold = _threshold;
}


std::vector<YoloBox> ObjectDetector_YOLO::detectObjects(cv::Mat& _inputMat, float _threshold)
{
    std::vector<YoloBox> detections;

    Mat inputBlob = blobFromImage(_inputMat, 1 / 255.F, netSize, Scalar(), true, false);

    yolo_net.setInput(inputBlob);
    vector<Mat> yolo_output_blobs;
    yolo_net.forward(yolo_output_blobs, get_outputsNames(yolo_net));

    // - Darknet Network produces output blob with a shape NxC where N is a number of
    //   detected objects and C is a number of classes + 4 where the first 4
    //   numbers are [center_x, center_y, width, height]
    for(int i=0; i<yolo_output_blobs.size(); i++)
    {
        int num_detections = yolo_output_blobs[i].size[0];
        int num_channels = yolo_output_blobs[i].size[1];
        float* yolo_data = (float*)yolo_output_blobs[i].data;

        for(int det=0; det<num_detections; det++)
        {
            Mat class_probabilities = yolo_output_blobs[i].row(det).colRange(5, yolo_output_blobs[i].cols);

            Point classIdPoint;
            double class_score;
            minMaxLoc(class_probabilities, NULL, &class_score, NULL, &classIdPoint);
            if(class_score < _threshold)
                continue;

            int centerX = (int)(yolo_data[num_channels*det + 0] * _inputMat.cols);
            int centerY = (int)(yolo_data[num_channels*det + 1] * _inputMat.rows);
            int width = (int)(yolo_data[num_channels*det + 2] * _inputMat.cols);
            int height = (int)(yolo_data[num_channels*det + 3] * _inputMat.rows);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            Rect box = Rect(left,top,width, height);
            detections.push_back(YoloBox(class_score, box, classNames[classIdPoint.x], classColors[classIdPoint.x]));
        }
    }

    detections = NMS(detections, nms_threshold);
    return detections;
}



void draw_label(Mat& _img, Rect& _box, string& _class_name, Scalar& _class_color, double& _score)
{
    int baseLine;
    double font_scale = 0.5;
    int thickness = 1;
    int score_text = (int)(_score * 100);
    std::string label = _class_name + ": " + std::to_string(score_text) + "%";

    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, font_scale, thickness, &baseLine);
    cv::Point startPoint = cv::Point(_box.x, _box.y + labelSize.height);
    cv::rectangle(_img, cv::Rect(_box.x, _box.y, labelSize.width, labelSize.height + baseLine), _class_color, cv::FILLED);
    cv::putText(_img, label, startPoint, cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(), thickness);
}



Scalar ObjectDetector_YOLO::get_color(int _class, int _numClasses)
{
    int offset = _class*123457 % _numClasses;

    float ratio = ((float)offset/_numClasses)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;

    float r = (1-ratio) * colors[i][2] + ratio*colors[j][2];
    float g = (1-ratio) * colors[i][1] + ratio*colors[j][1];
    float b = (1-ratio) * colors[i][0] + ratio*colors[j][0];

    return Scalar(b*255,g*255,r*255);
}


std::vector<cv::String> ObjectDetector_YOLO::get_outputsNames(cv::dnn::Net &_net)
{
    std::vector<String> names;
    if (names.empty())
    {
        std::vector<int> outLayers = _net.getUnconnectedOutLayers();
        std::vector<String> layersNames = _net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

std::vector<YoloBox> ObjectDetector_YOLO::NMS(std::vector<YoloBox>& _boxes, float _threshold)
{
    if (_boxes.size() == 0)
        return _boxes;

    std::sort(_boxes.begin(), _boxes.end(), compareBoxByScore);
    //bool flag[_faces.size()];
    //memset(flag, 0, _faces.size()); // - sets all the values in flag to zero
    vector<bool> flag(_boxes.size());
    std::fill(flag.begin(), flag.end(), 0);

    for (int i = 0; i < _boxes.size(); i++)
    {
        if (flag[i])
            continue;

        for (int j = i + 1; j < _boxes.size(); j++)
        {
            if (IoU(_boxes[i], _boxes[j]) > _threshold && _boxes[i].name == _boxes[j].name)
                flag[j] = 1;
        }
    }

    std::vector<YoloBox> boxes_nms;
    for (int i = 0; i < _boxes.size(); i++)
    {
        if (!flag[i])
            boxes_nms.push_back(_boxes[i]);
    }

    return boxes_nms;
}

bool ObjectDetector_YOLO::compareBoxByScore(const YoloBox &_box1, const YoloBox &_box2)
{
    return _box1.score > _box2.score;
}


float ObjectDetector_YOLO::IoU(YoloBox &_box1, YoloBox &_box2)
{
    float intersectionArea = (_box1.box & _box2.box).area();    // - Rectangle intersection
                                     //Rect b = box1 | box2;    // - Minimum Area Rectangle containing box1 and box2
    float unionArea = _box1.box.area() + _box2.box.area() - intersectionArea;
    float overlap = intersectionArea / unionArea;

    return overlap;
}














