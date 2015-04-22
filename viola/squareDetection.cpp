#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
Point chest_position(const Point head_center, const uint32_t depth);
vector<Rect> chest_positions_from_face_positions(const faces &detected_faces, const uint32_t depth);
Rect clamp_rect_to_image_size(const Mat &img, const Rect &rect);
void print_faces(const faces &detected_faces, Mat &frame, float scale_width, float scale_height);
faces detect_faces(const Mat &frame);
vector<Rect> detect_upper_bodies(const Mat &frame);
void CannyThreshold(int, void* io);

Mat create_ellipse_mask(const Rect &rectangle, const int ndims);
Mat create_ellipse_mask(const Point &center, const int radi_x, const int radi_y, const int channels);

inline bool point_within_ellipse(const Point &point, const Point &center, const int radi_x, const int radi_y);
void show_histogram(const Mat &hsv, const Mat &mask, vector<Mat> &histograms);

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

const float scale_width = 640 / 1280.;
const float scale_height = 480 / 1024.;

struct HSI {
    float I;
    float S;
    float H;
    HSI(int R, int G, int B):
        I((R + G + B) / 3.0),
        S(I > 0 ? (1 - std::min(R, std::min(G, B)) / I) : 0) {
        float tmp = acos((R - 0.5 * G - 0.5 * B) / sqrt(R * R + G * G + B * B - R * G - R * B - G * B));
        if (G >= B) {
            H = tmp;
        } else {
            H = 2 * M_PI - tmp;
        }
    }
};

void show_histogram(const Mat &hsv, const Mat &mask, vector<Mat> &histograms)
{
    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 32, sbins = 32;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };

    const float* ranges[] = { hranges, sranges };
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};


    int scale = 10;
    histograms.push_back(Mat::zeros(sbins * scale, hbins * scale, CV_32F));
    
    //calcHist converts the output matrix to CV_32F
    Mat histogram;
    calcHist(&hsv, 1, channels, mask, histogram, 2, histSize, ranges, true, false);
    Mat histImg = Mat::zeros(sbins * scale, hbins * scale, CV_8UC1);
    Mat histImgNorm = Mat::zeros(sbins * scale, hbins * scale, CV_32F);
    Mat histImgNormImg = Mat::zeros(sbins * scale, hbins * scale, CV_8UC3);
    
    double sum = 0;
    for (int h = 0; h < hbins; h++) {
        for (int s = 0; s < sbins; s++) {
            sum += histogram.at<float>(h, s);
        }
    }

    double sumNorm = 0;
    double maxValNorm = 0;
    for (int h = 0; h < hbins; h++) {
        for (int s = 0; s < sbins; s++) {
            histograms.back().at<float>(h, s) = histogram.at<float>(h, s) / sum;
            sumNorm += histograms.back().at<float>(h, s);
        }
    }

    

    double maxVal = 0;
    minMaxLoc(histogram, 0, &maxVal, 0, 0);
    minMaxLoc(histograms.back(), 0, &maxValNorm, 0, 0);

    //cout << "MAX1 " << maxVal << " MAX2 " << maxValNorm << endl;
    //cout << "sum " << sum << " sumNorm " << sumNorm << endl;
    
    for (int h = 0; h < hbins; h++) {
        for (int s = 0; s < sbins; s++) {
            const float binVal = histogram.at<float>(h, s);
            const int intensity = cvRound(255 * (binVal / maxVal));
            rectangle(histImg, Point(h * scale, s * scale),
                      Point((h + 1) * scale - 1, (s + 1) * scale - 1),
                      Scalar::all(intensity), CV_FILLED);
            /*
            if(binVal != 0.0){
                cout << 255*((binVal/sum)/maxValNorm) << endl;
            }
            */
        }
    }
    cout << "-----------------------" << endl;
    for (int h = 0; h < hbins; h++) {
        for (int s = 0; s < sbins; s++) {
            const float binVal = histograms.back().at<float>(h, s);
            const int intensity = cvRound(255 * (binVal / maxValNorm));
            rectangle(histImgNormImg, Point(h * scale, s * scale),
                      Point((h + 1) * scale - 1, (s + 1) * scale - 1),
                      Scalar::all(intensity), CV_FILLED);
            /*
            if(binVal != 0.0){
                cout << 255* (binVal / maxValNorm) << endl;
            }
            */
        }
    }

    if(histograms.size() > 1){
        cout << "first " << compareHist(histograms.back(), histograms.front(), CV_COMP_BHATTACHARYYA) << endl;
        cout << "self " <<compareHist(histograms.back(), histograms.back(), CV_COMP_BHATTACHARYYA) << endl;
        cout << "current " << compareHist(histograms.back(), histograms[histograms.size() - 2], CV_COMP_BHATTACHARYYA) << endl;
    }

    imshow("H-S Histogram", histImg);
    imshow("H-S Histogram-norm", histImgNormImg);
}

vector<Rect> init ()
{
    return vector<Rect>({Rect(85, 330, 100, 50)});
}

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

    namedWindow("Color image", CV_WINDOW_AUTOSIZE);
    namedWindow("H-S Histogram", CV_WINDOW_AUTOSIZE);
    namedWindow("H-S Histogram-norm", CV_WINDOW_AUTOSIZE);
    namedWindow("ROI", CV_WINDOW_AUTOSIZE);

    VideoCapture capture("output.mpg");

    Mat color_frame;
    Mat frame_color_hsv;

    if (!capture.isOpened()) {
        return -1;
    }

    vector<MatND> histograms;
    while (true) {
        capture.grab();
        capture >> color_frame;
        if(!color_frame.empty()){
            Mat frame_color_hsv;
            cvtColor(color_frame, frame_color_hsv, COLOR_BGR2HSV);

            Mat original = color_frame.clone();
            auto detected = init();
            
            Mat detectedROI = frame_color_hsv(detected.front()).clone();
            auto mask = create_ellipse_mask(detected.front(), 1);

            imshow("ROI", detectedROI);
            show_histogram(detectedROI, mask, histograms);

            rectangle(color_frame, detected.front(), Scalar(0,0,255), 3, 8, 0);
            imshow("Color image", color_frame);
            int c = waitKey(10);
            while((char)c != 's'){
                c = waitKey(10);
            }
            if ((char)c == 'c') {
                break;
            } else if (char(c) == 's') {

            }
        } else {
            capture.set(CV_CAP_PROP_POS_FRAMES, 0);
        }

    }
    return 0;
}

inline bool point_within_ellipse(const Point &point, const Point &center, const int radi_x, const int radi_y)
{
    return (((point.x - center.x) * (point.x - center.x)) / float((radi_x * radi_x)) + ((point.y - center.y) * (point.y - center.y)) / float((radi_y * radi_y))) <= 1;
}

Mat create_ellipse_mask(const Rect &rectangle, const int ndims)
{
    return create_ellipse_mask(Point(rectangle.width / 2, rectangle.height / 2), rectangle.width / 2, rectangle.height / 2, ndims);
}

Mat create_ellipse_mask(const Point &center, const int radi_x, const int radi_y, const int ndims)
{
    Mat mask;
    mask.create(radi_y * 2, radi_x * 2, CV_8UC1);
    int channels = mask.channels();
    int nRows = mask.rows;
    int nCols = mask.cols * channels;
    for (int i = 0; i < nRows; i++) {
        uchar* mask_row = mask.ptr<uchar>(i);
        for (int j = 0; j < nCols; j++) {
            mask_row[j] = point_within_ellipse(Point(j, i), center, radi_x, radi_y) ? 0xff : 0x0;
        }
    }

    vector<Mat> mask_channels;
    for (int i = 0; i < ndims; i++) {
        mask_channels.push_back(mask);
    }

    Mat mask_ndims;
    merge(mask_channels, mask_ndims);
    return mask_ndims;
}

Rect clamp_rect_to_image_size(const Mat &img, const Rect &rect)
{
    Rect clamped;
    Size img_size = img.size();
    clamped.x = std::max(0, rect.x);
    clamped.y = std::max(0, rect.y);
    clamped.x = std::min(clamped.x, img_size.width - 1);
    clamped.y = std::min(clamped.y, img_size.height - 1);
    if (rect.x < 0) {
        clamped.width = rect.width - rect.x - 1;
    } else if ((rect.x + rect.width) >= img_size.width) {
        clamped.width = img_size.width - rect.x - 1;
    } else {
        clamped.width = rect.width;
    }

    if (rect.y < 0) {
        clamped.height = rect.height - rect.y - 1;
    } else if ((rect.y + rect.height) >= img_size.height) {
        clamped.height = img_size.height - rect.y - 1;
    } else {
        clamped.height = rect.height;
    }

    return clamped;
}

vector<Rect> chest_positions_from_face_positions(const faces &detected_faces, const uint32_t depth)
{
    vector<Rect> chests;
    for (auto detected_face : detected_faces) {
        const face_shape &f = detected_face.first;
        const Point chest_vertex = chest_position(Point(f.x, f.y), depth);
        chests.push_back(Rect(chest_vertex.x, chest_vertex.y, f.width, f.height * 2.0));
    }
    return chests;
}

Point chest_position(const Point head_center, const uint32_t depth)
{
    const float f = 583;
    //const float f = 502;
    const float delta_y = 650;
    Point chest(head_center.x, (head_center.y * depth + f * delta_y) / depth);
    //cout << "FACE " << head_center.x << ' ' << head_center.y << " CHEST " << chest.x << ' ' << chest.y << endl;
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

void print_chests(const faces &detected_faces, Mat &frame, const float scale_width, const float scale_height, const uint32_t depth)
{
    auto chests = chest_positions_from_face_positions(detected_faces, depth);
    for (auto detected_chest : chests) {
        rectangle(frame, Rect(detected_chest.x * scale_width, detected_chest.y * scale_height, detected_chest.width * scale_width, detected_chest.height * scale_height),
                  Scalar(255, 0, 0), 2, 8, 0);
    }

    /*
    for (auto detected_face : detected_faces) {
        face_shape &f = detected_face.first;
        //face roi
        rectangle(frame, Rect(f.x, f.y, f.width, f.height), Scalar(255, 0, 0), 2, 8, 0);
        const Point center((f.x + f.width / 2 + f.width * 0.50 * 0.05) * scale_width, (f.y + f.height / 2) * scale_height);

        //ellipse(frame, center, Size((f.width / 2) * scale_width * 0.8, (f.height / 2) * scale_height), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
        //face center
        ellipse(frame, center, Size(5, 5), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);

        const Point chest_vertex = chest_position(Point(f.x, f.y), depth);
        //line head vertex - chest vertex
        line(frame, Point(f.x, f.y), chest_vertex, Scalar(0, 255, 0), 2, 8, 0);
        //chest center
        ellipse(frame, chest_vertex, Size(5, 5), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
        //chest roi
        rectangle(frame, Rect(chest_vertex.x, chest_vertex.y, f.width, f.height*2.0), Scalar(255, 0, 0), 2, 8, 0);
        //chest diagonals
        line(frame, Point(chest_vertex.x, chest_vertex.y), Point(chest_vertex.x + f.width, chest_vertex.y + f.height * 2.0), Scalar(0, 255, 0), 2, 8, 0);
        line(frame, Point(chest_vertex.x + f.width, chest_vertex.y), Point(chest_vertex.x, chest_vertex.y + f.height * 2.0), Scalar(0, 255, 0), 2, 8, 0);
        //chest ellipse
        const Point chest_center = Point(chest_vertex.x + f.width/2.0, chest_vertex.y + f.height);
        ellipse(frame, chest_center, Size(f.width/2.0, f.height), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
    }*/
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
