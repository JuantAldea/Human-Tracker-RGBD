/**
 * @file objectDetection2.cpp
 * @author A. Huaman ( based in the classic facedetect.cpp in samples/c )
 * @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream - Using LBP here
 */
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <ctime>
using namespace std;
using namespace cv;


typedef Rect face_shape;
typedef Rect eye;
typedef vector<eye> eyes;
typedef pair<face_shape, eyes> face;
typedef vector<face> faces;

void print_chests(const faces &detected_faces, Mat &frame, float scale_width, float scale_height, uint32_t depth);
Point chest_position (const Point head_center, const uint32_t depth);
vector<Rect> chest_position_from_face_position(const faces &detected_faces, const uint32_t depth);

void print_faces(const faces &detected_faces, Mat &frame, float scale_width, float scale_height);
Mat calculate_image_mask(const Mat& image, uint16_t depth, uint16_t delta);
faces detect_faces(const Mat &frame);
vector<Rect> detect_upper_bodies(const Mat &frame);
void CannyThreshold(int, void* io);


String face_cascade_name = "cascades/lbpcascade_frontalface.xml";
String eyes_cascade_name = "cascades/haarcascade_eye_tree_eyeglasses.xml";
String upperbodycascade_name = "cascades/haarcascade_upperbody.xml";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier upperbody_cascade;

int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;

RNG rng(12345);

const float scale_width = 640/1280.;
const float scale_height = 480/1024.;

int main(void)
{
    if (!face_cascade.load(face_cascade_name)) {
        printf("--(!)Error loading\n");
        return -1;
    };

    if (!eyes_cascade.load(eyes_cascade_name)) {
        printf("--(!)Error loading\n");
        return -1;
    };

    if (!upperbody_cascade.load(upperbodycascade_name)) {
        printf("--(!)Error loading\n");
        return -1;
    };

    namedWindow("Color image FULL RES", CV_WINDOW_AUTOSIZE);
    namedWindow("Color image", CV_WINDOW_AUTOSIZE);
    //namedWindow("Grey canny", CV_WINDOW_AUTOSIZE);
    //namedWindow("Disparity map", CV_WINDOW_AUTOSIZE);
    namedWindow("Disparity map eq", CV_WINDOW_AUTOSIZE);
    //namedWindow("Disparity map canny", CV_WINDOW_AUTOSIZE);
    namedWindow("Disparity map canny eq", CV_WINDOW_AUTOSIZE);
    //namedWindow("Depth image", CV_WINDOW_AUTOSIZE);
    //namedWindow("Point cloud map", CV_WINDOW_AUTOSIZE);
    //namedWindow("Grey image", CV_WINDOW_AUTOSIZE);
    namedWindow("Valid depth pixels", CV_WINDOW_AUTOSIZE);
    namedWindow("color_image_masked_valid_depth", CV_WINDOW_AUTOSIZE);
    namedWindow("ROI1", CV_WINDOW_AUTOSIZE);
    namedWindow("ROI2", CV_WINDOW_AUTOSIZE);
    namedWindow("ROI3", CV_WINDOW_AUTOSIZE);

    namedWindow("DEPTH MASK", CV_WINDOW_AUTOSIZE);

    VideoCapture capture(CV_CAP_OPENNI);
    capture.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_SXGA_15HZ);
    cout << "REGISTRATION " << capture.get( CV_CAP_PROP_OPENNI_REGISTRATION ) << endl;
    capture.set(CV_CAP_PROP_OPENNI_REGISTRATION, 1);

    Mat color_frame_SXGA;
    Mat color_frame;
    Mat grey_frame;
    Mat depth_frame;
    Mat grey_canny;
    Mat disparity_map;
    Mat disparity_map_canny;
    Mat disparity_map_eq;
    Mat disparity_map_eq_canny;
    Mat valid_depth_pixels;
    Mat valid_depth_pixels_3_channels;
    Mat color_image_masked_valid_depth;

    pair<Mat*, Mat*> io(&grey_frame, &grey_canny);
    pair<Mat*, Mat*> io2(&disparity_map, &disparity_map_canny);
    pair<Mat*, Mat*> io3(&disparity_map_eq, &disparity_map_eq_canny);

    createTrackbar("Min Threshold:", "Disparity map canny", &lowThreshold, max_lowThreshold, CannyThreshold, &io2);
    createTrackbar("Min Threshold:", "Disparity map canny", &lowThreshold, max_lowThreshold, CannyThreshold, &io2);
    createTrackbar("Min Threshold:", "Disparity map canny eq", &lowThreshold, max_lowThreshold, CannyThreshold, &io3);

    if (!capture.isOpened()) {
        return -1;
    }

    while (true) {
        capture.grab();
        capture.retrieve(color_frame_SXGA, CV_CAP_OPENNI_BGR_IMAGE);
        capture.retrieve(grey_frame, CV_CAP_OPENNI_GRAY_IMAGE);
        capture.retrieve(disparity_map, CV_CAP_OPENNI_DISPARITY_MAP);
        capture.retrieve(depth_frame, CV_CAP_OPENNI_DEPTH_MAP);
        capture.retrieve(valid_depth_pixels, CV_CAP_OPENNI_VALID_DEPTH_MASK);
        //capture.retrieve(point_cloud_map, CV_CAP_OPENNI_POINT_CLOUD_MAP);
        
        resize(color_frame_SXGA, color_frame, disparity_map.size());
        color_image_masked_valid_depth.create(color_frame.size(), color_frame.type());

        Mat original = color_frame.clone();

        auto detected_faces = detect_faces(color_frame_SXGA);
        
        uint pixel_count = 0;
        uint32_t depth_sum = 0;
        uint32_t disparity_sum = 0;        
        
        for(auto face : detected_faces){
            Rect rectangle = face.first;
            Rect rectangle_scaled(rectangle.x * scale_width + rectangle.width * scale_width * 0.05, rectangle.y * scale_height, rectangle.width * scale_width * 0.90, rectangle.height * scale_height);
            Mat faceROI = color_frame(rectangle_scaled);
            Mat faceROI_disparity = disparity_map(rectangle_scaled);
            Mat faceROI_depth = depth_frame(rectangle_scaled);
            Mat faceROI_valid_depth = valid_depth_pixels(rectangle_scaled);

            imshow("ROI1", faceROI);
            imshow("ROI2", faceROI_disparity);

            
            int channels = faceROI_disparity.channels();
            int nRows = faceROI_disparity.rows;
            int nCols = faceROI_disparity.cols * channels;
            int total_pixels = nRows * nCols;
            for(int  i = 0; i < nRows; ++i){
                const uchar* p_disparity = faceROI_disparity.ptr<uchar>(i);
                const ushort* p_depth = faceROI_depth.ptr<ushort>(i);
                const uchar* p_valid_depth = faceROI_valid_depth.ptr<uchar>(i);
                for (int j = 0; j < nCols; ++j){
                    disparity_sum += (p_valid_depth[j] & 0x1) * p_disparity[j];
                    depth_sum += (p_valid_depth[j] & 0x1) * p_depth[j];
                    pixel_count += (p_valid_depth[j] & 0x1);
                }
            }
            cout << pixel_count << ' ' << total_pixels << ' ' << pixel_count/float(total_pixels) << endl;
            if (pixel_count > (total_pixels * 0.6)){
                cout << "average disparity " << disparity_sum / float(pixel_count) << endl;
                cout << "average distance " << depth_sum / float(pixel_count) << endl;
            }else{
                cout << "invalid data" << endl;
            }
        }

        {
            float depth_average = depth_sum / float(pixel_count);
            int nRows = depth_frame.rows;
            int nCols = depth_frame.cols * depth_frame.channels();
            
            Mat depth_mask(valid_depth_pixels.size(), valid_depth_pixels.type());
            float average_disparity_meters = depth_sum / float(pixel_count) / 1000.0;
            float depth_step_resolution_mm = (2.73 * average_disparity_meters * average_disparity_meters + 0.74 * average_disparity_meters - 0.58);
            cout << "resolution " << depth_step_resolution_mm << endl;
            for(int  i = 0; i < nRows; ++i){
                const ushort* p_depth = depth_frame.ptr<ushort>(i);
                const uchar* p_valid_depth = valid_depth_pixels.ptr<uchar>(i);
                uchar* p_depth_mask = depth_mask.ptr<uchar>(i);
                for (int j = 0; j < nCols; ++j){
                    p_depth_mask[j] = (p_valid_depth[j] & 0x1) * (std::abs(p_depth[j] - depth_average) < (100 * depth_step_resolution_mm * 0.5)) * 255;
                }
            }

            Mat color_frame_depth_masked;
            Mat depth_mask_3_channels;
            bitwise_not(depth_mask, depth_mask);
            Mat channels[] =  {depth_mask, depth_mask, depth_mask};
            merge(channels, 3, depth_mask_3_channels);
            bitwise_or(color_frame, depth_mask_3_channels, color_frame_depth_masked);
            imshow("DEPTH MASK", color_frame_depth_masked);
        }
        
        bitwise_not(valid_depth_pixels, valid_depth_pixels);
        Mat channels[] =  {valid_depth_pixels, valid_depth_pixels, valid_depth_pixels};
        merge(channels, 3, valid_depth_pixels_3_channels);
        bitwise_or(color_frame, valid_depth_pixels_3_channels, color_image_masked_valid_depth);
        bitwise_not(valid_depth_pixels, valid_depth_pixels);

        equalizeHist(disparity_map, disparity_map_eq);
        
        //CannyThreshold(0, &io);
        //CannyThreshold(0, &io2);
        //CannyThreshold(0, &io3);
    
        print_faces(detected_faces, color_frame_SXGA, 1, 1);
        print_faces(detected_faces, color_frame, scale_width, scale_height);
        print_faces(detected_faces, color_image_masked_valid_depth, scale_width, scale_height);
        print_faces(detected_faces, disparity_map_eq, scale_width, scale_height);
        print_faces(detected_faces, disparity_map_eq_canny, scale_width, scale_height);
        print_chests(detected_faces, color_frame_SXGA, 1, 1, depth_sum / float(pixel_count));
        auto chests = chest_position_from_face_position(detected_faces, depth_sum / float(pixel_count));
        if (chests.size()){
            int corner_x = std::max(0, chests[0].x);
            int corner_y = chests[0].y;
            if (corner_y < color_frame_SXGA.size().height){
                ellipse(color_frame_SXGA, Point(corner_x, corner_y), Size(5, 5), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
                int width = chests[0].width;
                if (chests[0].x < 0){
                    width = chests[0].width - chests[0].x;
                }else if(chests[0].x + chests[0].width >= color_frame_SXGA.size().width){
                    width = color_frame_SXGA.size().width - chests[0].x;
                }
                
                int height = chests[0].height;
                if (chests[0].y + chests[0].height >= color_frame_SXGA.size().height){
                    height = color_frame_SXGA.size().height - chests[0].y;
                }else{

                }
                cout << color_frame_SXGA.size().width << ' ' << color_frame_SXGA.size().height << endl;
                cout <<  chests[0].x << ' ' <<  chests[0].y << ' ' <<  chests[0].width << ' ' <<  chests[0].height << endl;
                cout << corner_x << ' ' << corner_y << ' ' << width << ' ' << height << endl;
                Mat roi_chest = color_frame_SXGA(Rect(corner_x, corner_y, width, height));
                imshow("ROI3", roi_chest);
            }
        }

        
        ///calculate_image_mask(depth_frame, 100, 10);

        int c = waitKey(10);
        if ((char)c == 'c') {
            break;
        } else if (char(c) == 's' || detected_faces.size() * 0) {
            vector<int> compression_params;
            compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(9);
            time_t now;
            char date_str[100];
            now = time(NULL);
            //strftime(date_str, 20, "./imgs/%H-%M-%S.png", gmtime(&now));
            //imwrite(date_str, disparity_map, compression_params);
            sprintf(date_str, "./imgs/%lu.png", now);
            //strftime(date_str, 20, "./imgs/%H-%M-%S.png", gmtime(&now));
            imwrite(date_str, color_frame_SXGA, compression_params);
            cout << "Frame saved to " << date_str << endl;
        }

        //mask color with valid depth
        

        //vector<Vec3f> circles;
        //blur(grey_frame, grey_frame, Size(3, 3));
        //HoughCircles(grey_frame, circles, CV_HOUGH_GRADIENT, 1, 1, 200, 100, 0, 0);
        /*
        auto bodies = detect_upper_bodies(color_frame);

        for (auto body : bodies) {
            rectangle(color_frame, body, Scalar(0, 0, 255));
        }
        for (size_t i = 0; i < circles.size(); i++) {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // circle center
            circle(color_frame, center, 3, Scalar(0, 255, 0), -1, 8, 0);
            // circle outline
            circle(color_frame, center, radius, Scalar(0, 0, 255), 3, 8, 0);
        }

        */
        {
            Mat depth_frame_3_channels(color_frame.size(), color_frame.type());
            Mat channels[] =  {depth_frame, depth_frame, depth_frame};
            merge(channels, 3, depth_frame_3_channels);
            addWeighted(color_frame, 0.99, depth_frame_3_channels, 0.01, 0.0, color_frame, color_frame.type());
        }
        imshow("Color image FULL RES", color_frame_SXGA);
        imshow("Color image", color_frame);
        //imshow("Disparity map", disparity_map);
        imshow("Disparity map eq", disparity_map_eq);
        //imshow("Disparity map canny", disparity_map_canny);
        //imshow("Disparity map canny eq", disparity_map_eq_canny);
        imshow("Valid depth pixels", valid_depth_pixels);
        imshow("color_image_masked_valid_depth", color_image_masked_valid_depth);

        //imshow("Grey canny", grey_canny);

        //imshow("Depth map", depth_frame);
        //imshow("Point cloud map", point_cloud_map);
        //imshow("Grey image", grey_frame);

    }
    return 0;
}


inline bool point_within_ellipse(const Point &point, const Point &center, const int radi_x, const int radi_y)
{
    return (((point.x - center.x) * (point.x - center.x)) / (radi_x * radi_x) + ((point.y - center.y) * (point.y - center.y)) / (radi_y * radi_y)) < 1;
}

Mat calculate_image_mask(const Mat& image, uint16_t depth, uint16_t delta)
{
    Mat mask;

    mask.create(image.size(), image.type());
    cout << "profundidad " <<  image.depth() << endl;
    int channels = image.channels();

    int nRows = image.rows;
    int nCols = image.cols * channels;

    if (image.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }
        
    cout << "(" + depth << delta << ")" << endl;

    /*
    uint16_t in_range_mask = 0xffff;
    uint16_t out_range_mask = 0x0;
    for (int i = 0; i < nRows; ++i) {
        uint *image_ptr = image.ptr<CV_16UC1>(i);
        uint *mask_ptr  = mask.ptr<CV_16UC1>(i);
        for (int j = 0; j < nCols; ++j) {
            mask_ptr[j] = std::abs(image_ptr[j] - depth) <= delta ? in_range_mask : out_range_mask;
        }
    }
    */

    return mask;
}

vector<Rect> chest_position_from_face_position(const faces &detected_faces, const uint32_t depth)
{
    vector<Rect> chests;
    for (auto detected_face : detected_faces) {
        const face_shape &f = detected_face.first;
        const Point center((f.x + f.width / 2 + f.width * 0.50 * 0.05), (f.y + f.height / 2));
        const Point chest_center = chest_position(center, depth);
        const Point roi_corner(chest_center.x - 1.5 * f.width / 2, chest_center.y - 1.75 * f.height / 2);
        const Size  roi_size(f.width*1.5*0.8, f.height*3.5/2.0);
        chests.push_back(Rect(roi_corner, roi_size));
    }
    return chests;
}

Point chest_position (const Point head_center, const uint32_t depth)
{
    const float f = 583;
    const float delta_y = 800;
    Point chest(head_center.x, (head_center.y * depth + f * delta_y) / depth);
    cout << "FACE " << head_center.x << ' ' << head_center.y << " CHEST " << chest.x << ' ' << chest.y << endl;
    return chest;
}

void print_faces(const faces &detected_faces, Mat &frame, float scale_width, float scale_height)
{
    for (auto detected_face : detected_faces) {
        face_shape &f = detected_face.first;
        eyes &e = detected_face.second;
        Point center((f.x + f.width / 2 + f.width * 0.50 * 0.05) * scale_width, (f.y + f.height / 2) * scale_height);
        ellipse(frame, center, Size((f.width / 2) * scale_width * 0.8, (f.height / 2) * scale_height), 0, 0, 360, Scalar(255, 0, 0), 2, 8, 0);
        ellipse(frame, center, Size(5, 5), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
        for (size_t j = 0; j < e.size(); j++) {
            Point eye_center((f.x + e[j].x + e[j].width / 2) * scale_width, (f.y + e[j].y + e[j].height / 2) * scale_height);
            int radius = cvRound((e[j].width + e[j].height) * 0.25 * scale_width);
            circle(frame, eye_center, radius, Scalar(255, 0, 255), 3, 8, 0);
        }
    }
}

void print_chests(const faces &detected_faces, Mat &frame, float scale_width, float scale_height, uint32_t depth)
{
    for (auto detected_face : detected_faces) {
        face_shape &f = detected_face.first;
        Point center((f.x + f.width / 2 + f.width * 0.50 * 0.05) * scale_width, (f.y + f.height / 2) * scale_height);
        ellipse(frame, center, Size((f.width / 2) * scale_width * 0.8, (f.height / 2) * scale_height), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
        ellipse(frame, center, Size(5, 5), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
        Point chest_center = chest_position(center, depth);
        ellipse(frame, chest_center, Size((1.5 * f.width / 2) * scale_width * 0.8, (1.75 * f.height / 2) * scale_height), 0, 0, 360, Scalar(255, 0, 0), 2, 8, 0);
        ellipse(frame, chest_center, Size(5, 5), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
    }
}


faces detect_faces(const Mat &frame)
{
    Mat frame_gray;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    faces detected_faces;

    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0, Size(80, 80));
    for (size_t i = 0; i < faces.size(); i++) {
        Mat faceROI = frame_gray(faces[i]);
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
        if (eyes.size() == 2) {
            detected_faces.push_back(face(faces[i], eyes));
        }
    }

    return detected_faces;
}

vector<Rect> detect_upper_bodies(const Mat &frame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    std::vector<Rect> upperbodies;
    upperbody_cascade.detectMultiScale(frame, upperbodies, 1.1, 2, 0, Size(110, 110));
    //for (size_t i = 0; i < upperbodies.size(); i++) {
    //Point center(upperbodies[i].x + upperbodies[i].width / 2, upperbodies[i].y + upperbodies[i].height / 2);
    //ellipse(frame, center, Size(upperbodies[i].width / 2, upperbodies[i].height / 2), 0, 0, 360, Scalar(500, 0, 0), 2, 8, 0);
    //}
    return upperbodies;

}

void CannyThreshold(int, void *io)
{
    Mat input(*((std::pair<Mat*, Mat*>*)io)->first);
    Mat *output = ((std::pair<Mat*, Mat*>*)io)->second;
    blur(input, *output, Size(5, 5));
    Canny(*output, *output, lowThreshold, lowThreshold * ratio, kernel_size);
}
