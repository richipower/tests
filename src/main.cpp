
#include <opencv2/opencv.hpp>
#include <iostream>

#include "ObjectDetector_YOLO.hpp"


using namespace std;
using namespace cv;
using namespace cv::dnn;


int main(int argc, char *argv[])
{
    // --- YOLO initialization
    string models_folder = "../../../darknet_models/";
    string cfg_file = models_folder + "yolo_v3_tiny_prn.cfg";
    string weights_file = models_folder + "yolo_v3_tiny_prn.weights";
    string class_names_file = models_folder + "yolo_coco.names";

    ObjectDetector_YOLO yolo;
    yolo.setup(cfg_file, class_names_file, weights_file, Size(320,320));
    // --- END YOLO initialization


    Mat warmup_img = imread("/Users/richi/Downloads/dog-2.jpg");
    cout << "Warming up started..." << flush;
    for(int i = 0; i<10; i++)
    {
        auto yolo_boxes = yolo.detectObjects(warmup_img, 0.3);
        cout << i+1 << flush;
    }
    cout << " ---> DONE" << endl;


    string file_name = argv[1];
    VideoCapture cap(file_name);
    Mat frame;

    while(true)
    {
        bool ret = cap.read(frame);
        if(ret == false)
            break;

        resize(frame, frame, Size(), 0.5, 0.5);
        auto begin = std::chrono::system_clock::now();

        auto yolo_boxes = yolo.detectObjects(frame, 0.2);

        std::chrono::duration<double> elapsed_time = std::chrono::system_clock::now() - begin;
        cout << "Elapsed Time: " << elapsed_time.count() << " seconds." << endl;


        for(auto yoloBox : yolo_boxes)
        {
            rectangle(frame, yoloBox.box, Scalar(0,255,0), 3);
        }



        imshow("Output", frame);
        char key = waitKey(30);
        if(key == 'q')
            break;
    }

}

