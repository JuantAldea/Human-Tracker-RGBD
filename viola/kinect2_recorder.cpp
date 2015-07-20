#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <unistd.h>


#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/threading.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;

bool RECORDING = false;
bool RUNNING = true;

FileStorage fs;
string fic_name;
int num_fic = 1;
chrono::time_point<chrono::high_resolution_clock> t0, t1, dt;

libfreenect2::Freenect2Device *dev;
libfreenect2::Freenect2 freenect2;
libfreenect2::SyncMultiFrameListener *listener;
libfreenect2::FrameMap frames_kinect2;
libfreenect2::PacketPipeline *pipeline;
size_t rgb_height, rgb_width, depth_height, depth_width;
size_t n_pixels_depth;

void stop_recording()
{
    if (RECORDING) {
        RECORDING = false;
        cout << "RECORDING STOPPED" << endl;
    }
}

void start_recording()
{
    if (!RECORDING) {
        cout << "RECORDING STARTED" << endl;
        RECORDING = true;
    }
}

void exit_program()
{
    stop_recording();
    RUNNING = false;
}

void keyboard_handler()
{
    namedWindow("keyboard_handler", WINDOW_AUTOSIZE);

    while(RUNNING){
        const int key = cv::waitKey(1);
        if (key == (int)'a') {
            start_recording();
        } else if (key == (int)'b') {
            stop_recording();
        } else if (key == (int)'c') {
            exit_program();
        }
    }
}


int handle_OpenCV_error( int status, const char* func_name, const char* err_msg, const char* file_name, int line, void* userdata )
{
    std::cerr << func_name << ": " << err_msg << std::endl;
    throw;
    return 0;
}

void write_images_threads(const int frame_count, uchar * __restrict  rgb_data, uchar * __restrict  depth_data)
{
    const cv::Mat color_mat = cv::Mat(rgb_height, rgb_width, CV_8UC3, rgb_data);
    const cv::Mat depth_mat = cv::Mat(depth_height, depth_width, CV_32FC1, depth_data);

    const cv::Mat depth_mat_1_3 = cv::Mat(depth_height, depth_width, CV_8UC3);
    const cv::Mat depth_mat_4 = cv::Mat(depth_height, depth_width, CV_8UC3);

    uchar *depth3_ptr = depth_mat_1_3.data;
    uchar *depth4_ptr = depth_mat_4.data;
    uchar *depth_ptr = depth_mat.data;
    
    for (size_t i = 0; i < n_pixels_depth; i++){
      *(depth3_ptr + 0) = *(depth_ptr + 0);
      *(depth3_ptr + 1) = *(depth_ptr + 1);
      *(depth3_ptr + 2) = *(depth_ptr + 2);
      *(depth4_ptr + 0) = *(depth_ptr + 3);
      
      //*(depth4_ptr + 1) = *(depth_ptr + 3);
      //*(depth4_ptr + 2) = *(depth_ptr + 3);

      depth3_ptr += 3;
      depth4_ptr += 3;
      depth_ptr += 4;
    }


    std::ostringstream out_depth_1_3;
    std::ostringstream out_depth_4;
    std::ostringstream out_color;
    out_depth_1_3 << "/media/juant/VIDEOS" << "/video/depth_1_3_" << frame_count << ".png";
    out_depth_4 << "/media/juant/VIDEOS" << "/video/depth_4_" << frame_count << ".png";
    out_color   << "/media/juant/VIDEOS" << "/video/color_" << frame_count << ".png";
    cv::imwrite(out_color.str(), color_mat);
    cv::imwrite(out_depth_1_3.str(), depth_mat_1_3);
    cv::imwrite(out_depth_4.str(), depth_mat_4);
    delete rgb_data;
    delete depth_data;
}

int main(int argc, char **argv)
{
    cv::redirectError(handle_OpenCV_error);

    if (freenect2.enumerateDevices() == 0) {
        std::cout << "no device connected!" << std::endl;
        return -1;
    }

    //kinect2 initialization
    const std::string serial = freenect2.getDefaultDeviceSerialNumber();

    pipeline = new libfreenect2::OpenCLPacketPipeline();

    const float minDepth = 0;
    const float maxDepth = 12.0;
    const bool bilateral_filter = true;
    const bool edge_aware_filter = true;

    libfreenect2::DepthPacketProcessor::Config config;
    config.EnableBilateralFilter = bilateral_filter;
    config.EnableEdgeAwareFilter = edge_aware_filter;
    config.MinDepth = minDepth;
    config.MaxDepth = maxDepth;

    pipeline->getDepthPacketProcessor()->setConfiguration(config);
    dev = freenect2.openDevice(serial, pipeline);

    if (dev == nullptr) {
        std::cout << "no device connected or failure opening the default one!" << std::endl;
        exit(-1);
    }

    //listener = new libfreenect2::SyncMultiFrameListener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
    listener = new libfreenect2::SyncMultiFrameListener(libfreenect2::Frame::Color | libfreenect2::Frame::Depth);
    dev->setColorFrameListener(listener);
    dev->setIrAndDepthFrameListener(listener);
    dev->start();
        
    int FRAMERATE = 20;
    
    if (argc > 1){
       FRAMERATE = atoi(argv[1]);
    }
    
    std::chrono::milliseconds T_FPS(uint64_t(std::round((1 / float(FRAMERATE)) * 1000)));
    
    std::thread keyboard_handler_thread(keyboard_handler);

    listener->waitForNewFrame(frames_kinect2);
    libfreenect2::Frame *rgb = frames_kinect2[libfreenect2::Frame::Color];
    libfreenect2::Frame *depth = frames_kinect2[libfreenect2::Frame::Depth];
    
    n_pixels_depth = depth->height * depth->width;
    
    rgb_height = rgb->height;
    rgb_width = rgb->width;
    depth_height = depth->height;
    depth_width = depth->width;
    
    //VideoWriter video_D1_3("capture_D1_3.avi", CV_FOURCC('F','F','V','1'), 24, cvSize(depth->width, depth->height));
    //VideoWriter video_D4("capture_D4.avi", CV_FOURCC('F','F','V','1'), 24, cvSize(depth->width, depth->height));
    //VideoWriter video_RGB("capture_RGB.avi", CV_FOURCC('M','J','P','G'), 24, cvSize(rgb->width, rgb->height));
    
    //VideoWriter video_D1_3("capture_D1_3.avi", CV_FOURCC('L','A','G','S'), 24, cvSize(depth->width, depth->height));
    //VideoWriter video_D4("capture_D4.avi",  CV_FOURCC('L','A','G','S'), 24, cvSize(depth->width, depth->height));
    //VideoWriter video_RGB("capture_RGB.avi", CV_FOURCC('L','A','G','S'), 24, cvSize(rgb->width, rgb->height));

    char current_dir[100];
    getcwd(current_dir, 100);

    listener->waitForNewFrame(frames_kinect2);
    t0 = chrono::high_resolution_clock::now();
    size_t framecount = 0;

    time_t start, end;
    int counter = 0;
    double sec;
    double fps;
    time_t start_start, end_end;;
    time(&start_start);

    while (RUNNING) {        
        
        if (counter == 0){
            time(&start);
        }
        
        t1 = chrono::high_resolution_clock::now();
        std::chrono::milliseconds dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);    
        
        if (dt < T_FPS){
            continue;
        }
        
        listener->waitForNewFrame(frames_kinect2);

        //const auto d_actual = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();
        //const auto t_fps_count = std::chrono::duration_cast<std::chrono::milliseconds>(T_FPS).count();
        //std::cout << "DT " << d_actual << ' ' << t_fps_count << std::endl;
        
        t0 = t1;


        rgb = frames_kinect2[libfreenect2::Frame::Color];
        depth = frames_kinect2[libfreenect2::Frame::Depth];
        
        if (RECORDING) {
            uchar *rgb_data = new uchar[rgb->height * rgb->width * 3];
            uchar *depth_data = new uchar[depth->height * depth->width * 4];
            
            memcpy(rgb_data, rgb->data, rgb->height * rgb->width * 3);
            memcpy(depth_data, depth->data, depth->height * depth->width * 4);

            std::thread *recorder =  new std::thread(write_images_threads, framecount, rgb_data, depth_data);
            recorder->detach();
            framecount++;
        }

        counter++;
        if (counter > 30){
            time(&end);
            sec = difftime(end, start);
            fps = counter/sec;
            printf("%.2f fps\n", fps);
        }

        if (counter == (INT_MAX - 1000)){
            counter = 0;
        }


        listener->release(frames_kinect2);
    }
    time(&end_end);
    cout << "TOTAL TIME " << difftime(end_end, start_start) << std::endl;

    listener->waitForNewFrame(frames_kinect2);
    listener->release(frames_kinect2);
    dev->stop();
    dev->close();
    
    exit_program();
    keyboard_handler_thread.join();
    return 0;

}
