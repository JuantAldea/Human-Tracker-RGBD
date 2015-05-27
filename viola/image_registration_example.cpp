#include "ImageRegistration.h"
#include <cstdlib>

  void dispDepth(const cv::Mat &in, cv::Mat &out, const float maxValue)
  {
    cv::Mat tmp = cv::Mat(in.rows, in.cols, CV_8U);
    const uint32_t maxInt = 255;

    #pragma omp parallel for
    for(int r = 0; r < in.rows; ++r)
    {
      const uint16_t *itI = in.ptr<uint16_t>(r);
      uint8_t *itO = tmp.ptr<uint8_t>(r);

      for(int c = 0; c < in.cols; ++c, ++itI, ++itO)
      {
        *itO = (uint8_t)std::min((*itI * maxInt / maxValue), 255.0f);
      }
    }

    cv::applyColorMap(tmp, out, cv::COLORMAP_JET);
  }

  void combine(const cv::Mat &inC, const cv::Mat &inD, cv::Mat &out)
  {
    out = cv::Mat(inC.rows, inC.cols, CV_8UC3);

    #pragma omp parallel for
    for(int r = 0; r < inC.rows; ++r)
    {
      const cv::Vec3b
      *itC = inC.ptr<cv::Vec3b>(r),
       *itD = inD.ptr<cv::Vec3b>(r);
      cv::Vec3b *itO = out.ptr<cv::Vec3b>(r);

      for(int c = 0; c < inC.cols; ++c, ++itC, ++itD, ++itO)
      {
        itO->val[0] = (itC->val[0] + itD->val[0]) >> 1;
        itO->val[1] = (itC->val[1] + itD->val[1]) >> 1;
        itO->val[2] = (itC->val[2] + itD->val[2]) >> 1;
      }
    }
  }

int main(int argc, char **argv)
{
    char *calib_dir = getenv("HOME");
  std::string calib_path = std::string(calib_dir) + "/kinect2_calib/";
  std::string sensor = "0";
  ImageRegistration reg;
  reg.init(calib_path, sensor);
  cv::Mat color = cv::imread(argv[1]);
  cv::Mat depth = cv::imread(argv[2], CV_16UC1);
  
  cv::Mat registered_depth;
  reg.register_images(color, depth, registered_depth);
  std::cout << "reg.depth_high_res " << registered_depth.rows << ' ' << registered_depth.cols << std::endl;
  
  cv::Mat depthDisp;
  dispDepth(registered_depth, depthDisp, 12000.0f);
  cv::Mat combined;
  cv::flip(color, color, 1);
  combine(color, depthDisp, combined);
  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display window", combined);
  
  char s;
  do{
    s = cv::waitKey(0);
    
  }while(s != 's');

  return 0;
}