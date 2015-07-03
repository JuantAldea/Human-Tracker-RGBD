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
    int hbins = 31;
    int sbins = 32;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, it's scaled down by a half
    // so that it fits in a byte.
    float hranges[] = {0, 180};
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = {0, 256};
    const float* ranges[] = {hranges, sranges};
    int channels[] = {0, 1};

    int scale = 10;
    histograms.push_back(Mat::zeros(sbins * scale, hbins * scale, CV_32F));
    
    //calcHist converts the output matrix to CV_32F
    Mat histogram;
    calcHist(&hsv, 1, channels, mask, histogram, 2, histSize, ranges, true, false);

    Mat histogram_v;
    Mat histV;

    {
        vector<Mat> hsv_planes;
        split(hsv, hsv_planes);
        int histSize[] = {32,};
        float range[] = { 0, 256 } ;
        const float* histRange = {range};
        
        calcHist(&hsv_planes[2], 1, 0, mask, histogram_v, 1, histSize, &histRange, true, false);
        cout << "sizes " << histogram_v.rows << ' ' << histogram_v.cols << endl;
        histogram_v = histogram_v.t();

        cout << "sizes " << histogram.rows << ' ' << histogram.cols << endl;
        histogram.push_back(histogram_v);
        cout << "sizes " << histogram.rows << ' ' << histogram.cols << endl;
        
        double sum = 0;
        cout << "orig \n";
        for (int h = 0; h < histogram_v.rows; h++) {
            for (int s = 0; s < histogram_v.cols; s++) {
                sum += histogram_v.at<float>(h, s);
            }
        }

        cout << "sum " << sum << endl;
        cout << "normalized \n";
        for (int h = 0; h < histogram_v.rows; h++) {
            for (int s = 0; s < histogram_v.cols; s++) {
                histogram_v.at<float>(h, s) /= sum;
            }
        }
        cout << endl;

        histV = Mat::zeros(histogram_v.rows * scale, histogram_v.cols * scale, CV_8UC1);
        double maxV;
        minMaxLoc(histogram_v, 0, &maxV, 0, 0);
        for (int h = 0; h < histogram_v.rows; h++) {
            for (int s = 0; s < histogram_v.cols; s++) {
                const float binVal = histogram_v.at<float>(h, s);
                const int intensity = cvRound(255 * (binVal / 1));
                rectangle(histV, Point(s * scale, h * scale),
                          Point((s + 1) * scale - 1, (h + 1) * scale - 1),
                          Scalar::all(intensity), CV_FILLED);
            }
        }
        cout << endl;
    }

    sbins = histogram.rows;
    hbins = histogram.cols;
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
    
    for (int h = 0; h < hbins; h++) {
        for (int s = 0; s < sbins; s++) {
            const float binVal = histogram.at<float>(h, s);
            const int intensity = cvRound(255 * (binVal / maxVal));
            rectangle(histImg, Point(h * scale, s * scale),
                      Point((h + 1) * scale - 1, (s + 1) * scale - 1),
                      Scalar::all(intensity), CV_FILLED);
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
        }
    }

    if(histograms.size() > 1){
        cout << "first " << compareHist(histograms.back(), histograms.front(), CV_COMP_BHATTACHARYYA) << endl;
        cout << "self " <<compareHist(histograms.back(), histograms.back(), CV_COMP_BHATTACHARYYA) << endl;
        cout << "current " << compareHist(histograms.back(), histograms[histograms.size() - 2], CV_COMP_BHATTACHARYYA) << endl;
    }

    imshow("H-S Histogram", histV);
    imshow("H-S Histogram-norm", histImgNormImg);
}

Mat histogram_to_image(const Mat &histogram, const int scale)
{
    Mat histImg = Mat::zeros(histogram.rows * scale, histogram.cols * scale, CV_8UC1);
    double maxVal = 0;
    minMaxLoc(histogram, 0, &maxVal, 0, 0);
    for (int row = 0; row < histogram.rows; row++) {
        for (int col = 0; col < histogram.cols; col++) {
            const float binVal = histogram.at<float>(row, col);
            const int intensity = cvRound(255 * (binVal / maxVal));
            rectangle(histImg, Point(row * scale, col * scale),
                      Point((row + 1) * scale - 1, (col + 1) * scale - 1),
                      Scalar::all(intensity), CV_FILLED);
        }
    }
    return histImg;
}

Mat compute_color_model(const Mat &hsv, const Mat &mask)
{
    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    Mat histogram;
    const int hbins = 31;
    const int sbins = 32;
    
    {
        // hue varies from 0 to 179, it's scaled down by a half
        const int histSize[] = {hbins, sbins};
        // so that it fits in a byte.
        const float hranges[] = {0, 180};
        // saturation varies from 0 (black-gray-white) to
        // 255 (pure spectrum color)
        const float sranges[] = {0, 256};
        const float* ranges[] = {hranges, sranges};
        const int channels[] = {0, 1};
        calcHist(&hsv, 1, channels, mask, histogram, 2, histSize, ranges, true, false);
    }

    Mat histogram_v;
    {
        const int channels[] = {2};
        const int histSize[] = {sbins};
        const float range[] = {0, 256} ;
        const float* histRange = {range};
        calcHist(&hsv, 1, channels, mask, histogram_v, 1, histSize, &histRange, true, false);
    }
    
    histogram_v = histogram_v.t();
    histogram.push_back(histogram_v);
    
    double sum = 0;
    for (int h = 0; h < histogram.rows; h++) {
        for (int s = 0; s < histogram.cols; s++) {
            sum += histogram.at<float>(h, s);
        }
    }

    for (int h = 0; h < histogram.rows; h++) {
        for (int s = 0; s < histogram.cols; s++) {
            histogram.at<float>(h, s) /= sum;
        }
    }

    return histogram;
}

Mat init (Mat frame)
{
    vector<Rect> detected({Rect(85, 330, 100, 50)});
    Mat frame_hsv;
    cvtColor(frame, frame_hsv, COLOR_BGR2HSV);
    Mat detectedROI = frame_hsv(detected.front()).clone();
    Mat detectedBGRROI = frame(detected.front()).clone();
    Mat mask = create_ellipse_mask(detected.front(), 1);
    Mat model = compute_color_model(detectedROI, mask);
    imshow("ROI", detectedBGRROI);
    rectangle(frame, detected.front(), Scalar(0,0,255), 3, 8, 0);
    imshow("Color image", frame);
    return model;
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
    namedWindow("Color model", CV_WINDOW_AUTOSIZE);
    namedWindow("H-S Histogram-norm", CV_WINDOW_AUTOSIZE);
    namedWindow("ROI", CV_WINDOW_AUTOSIZE);

    VideoCapture capture("output.mpg");

    Mat color_frame;
    Mat frame_color_hsv;

    if (!capture.isOpened()) {
        return -1;
    }

    vector<Mat> histograms;
    bool target_found = false;
    Mat model;
    while (true) {
        capture.grab();
        capture >> color_frame;
        
        if(!color_frame.empty()){
            Mat copy = color_frame.clone();
        
            if (!target_found){
                target_found = true;
                model = init(copy);
            }

            Mat object = init(color_frame);
            double comparison = compareHist(model, object, CV_COMP_BHATTACHARYYA);


            cout << "comparison " << comparison << endl;
            
            int c = waitKey(10);            
            while((char)c != 's'){
                c = waitKey(10);
            }

            if ((char)c == 'c') {
                break;
            }

        } else {
            //rewind
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
