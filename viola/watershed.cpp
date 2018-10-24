// Usage: ./app input.jpg
#include "opencv2/opencv.hpp"
#include <string>

using namespace cv;
using namespace std;

class WatershedSegmenter{
private:
    cv::Mat markers;
public:
    void setMarkers(cv::Mat& markerImage)
    {
        markerImage.convertTo(markers, CV_32S);
    }

    cv::Mat process(cv::Mat &image)
    {
        cv::watershed(image, markers);
        markers.convertTo(markers,CV_8U);
        return markers;
    }
};


int main(int argc, char* argv[])
{
    double max, min;
    int x = atoi(argv[1]);
    int y = atoi(argv[2]);
    cv::Mat depth3 = cv::imread(argv[3]);
    cv::Mat depth4 = cv::imread(argv[4]);

    cv::Mat image = cv::Mat(depth3.rows, depth3.cols, CV_32FC1);

    size_t n_bytes = depth3.rows * depth3.cols;

    uchar *image_ptr = image.data;
    uchar *depth3_ptr = depth3.data;
    uchar *depth4_ptr = depth4.data;

    #pragma omp parallel for
    for (size_t i = 0; i < n_bytes; i++){
      *(image_ptr + 4 * i + 0) = *(depth3_ptr + 3 * i + 0);
      *(image_ptr + 4 * i + 1) = *(depth3_ptr + 3 * i + 1);
      *(image_ptr + 4 * i + 2) = *(depth3_ptr + 3 * i + 2);
      *(image_ptr + 4 * i + 3) = *(depth4_ptr + 3 * i + 0);
    }

    cv::Mat image2 = image.clone();
    cv::cvtColor(image2, image,CV_GRAY2BGR, 3);
    //image2.convertTo(image, CV_GRAY2BGR);

    cv::Mat blank(image.size(),CV_8U,cv::Scalar(0xFF));
    cv::Mat dest;

    // Create markers image
    cv::Mat markers(image.size(),CV_8U,cv::Scalar(-1));
    //Rect(topleftcornerX, topleftcornerY, width, height);
    //top rectangle
    markers(Rect(0,0,image.cols, 5)) = Scalar::all(1);
    //bottom rectangle
    markers(Rect(0,image.rows-5,image.cols, 5)) = Scalar::all(1);
    //left rectangle
    markers(Rect(0,0,5,image.rows)) = Scalar::all(1);
    //right rectangle
    markers(Rect(image.cols-5,0,5,image.rows)) = Scalar::all(1);
    //centre rectangle
    int centreW = image.cols/4;
    int centreH = image.rows/4;
    markers(Rect((image.cols/2)-(centreW/2),(image.rows/2)-(centreH/2), centreW, centreH)) = Scalar::all(1);
    markers.convertTo(markers,CV_BGR2GRAY);
    imshow("markers", markers);

    //Create watershed segmentation object
    WatershedSegmenter segmenter;
    segmenter.setMarkers(markers);
    cv::Mat wshedMask = segmenter.process(image);
    cv::Mat mask;
    convertScaleAbs(wshedMask, mask, 1, 0);
    double thresh = threshold(mask, mask, 1, 255, THRESH_BINARY);
    bitwise_and(image, image, dest, mask);

    cv::minMaxLoc(mask, &min, &max);
    mask -= min;
    mask /= (max - min);
    mask *= 255;
    mask.convertTo(mask, CV_8UC1);


    cv::minMaxLoc(dest, &min, &max);
    dest -= min;
    dest /= (max - min);
    dest *= 255;
    dest.convertTo(dest,CV_8U);
    imshow("final_result", mask);
    imshow("final_reasdsult", dest);
    imshow("final_reasdsult123", image);

    /*
    cv::Mat mask = cv::Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
    cv::Rect bounding_box;
    cv::floodFill(image, mask, cv::Point(x, y), cv::Scalar(128), &bounding_box, cv::Scalar(20.0f), cv::Scalar(20.0f), cv::FLOODFILL_MASK_ONLY);
    image(bounding_box) = Scalar::all(128);


    cv::minMaxLoc(image, &min, &max);
    image -= min;
    image /= (max - min);
    image *=255;
    image.convertTo(image, CV_8UC1);


    cv::minMaxLoc(mask, &min, &max);
    mask -= min;
    mask /= (max - min);
    mask *= 255;
    mask.convertTo(mask, CV_8UC1);
    imshow("final_result", mask);
    imshow("final_reasdsult", image);
    */

    cv::waitKey(0);

    return 0;
}
