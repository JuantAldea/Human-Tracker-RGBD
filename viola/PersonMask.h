#include <algorithm>

cv::Mat person_mask(const float x, const float y, const float z, cv::Mat rgb_frame, cv::Mat depth_frame, cv::Mat background_depth, cv::Mat cameraMatrix, cv::Mat lookupX, cv::Mat lookupY, cv::Mat &display, cv::Mat &display_mask)
{

    std::cout <<"SIZE " << depth_frame.rows << ' ' << depth_frame.cols << std::endl;
    using namespace Eigen;

    const float fx = cameraMatrix.at<double>(0, 0);
    const float fy = cameraMatrix.at<double>(1, 1);
    const float cx = cameraMatrix.at<double>(0, 2);
    const float cy = cameraMatrix.at<double>(1, 2);

    float Z_TOLERANCE = 750.0;
    Vector3f translation_upper_left(-1.0, -1, 0.0);
    Vector3f translation_lower_right(1.0, 0.5, 0.0);

    Eigen::Vector3f point_3D = point_3D_reprojection(x, y, z, lookupX, lookupY);
    std::cout << "person_mask 3d " << ' ' << point_3D[0] << ' ' << point_3D[1] << ' ' << point_3D[2] << std::endl;
    point_3D[1] = 0;

    const Vector3f point_3D_upper_left = point_3D + translation_upper_left;
    Vector2i upper_left = point_3D_projection(point_3D_upper_left, fx, fy, cx, cy);

    const Vector3f point_3D_lower_right = point_3D + translation_lower_right;
    Vector2i lower_right = point_3D_projection(point_3D_lower_right, fx, fy, cx, cy);

    lower_right[0] = std::max(0, std::min(rgb_frame.cols - 1, lower_right[0]));
    lower_right[1] = std::max(0, std::min(rgb_frame.rows - 1, lower_right[1]));

    upper_left[0] = std::max(0, std::min(rgb_frame.cols - 1, upper_left[0]));
    upper_left[1] = std::max(0, std::min(rgb_frame.rows - 1, upper_left[1]));

    //Vector2i upper_left = translate_2D_vector_in_3D_space(x, y, z, translation_upper_left, cameraMatrix, lookupX, lookupY);
    //Vector2i upper_right = translate_2D_vector_in_3D_space(x, y, z, translation_lower_right, cameraMatrix, lookupX, lookupY);

    std::cout << "person_mask LEFT " << upper_left[0] << ' ' << upper_left[1] << std::endl;
    std::cout << "person_mask RIGTH " << lower_right[0] << ' ' << lower_right[1] << std::endl;

    cv::Rect person_rect(upper_left[0], upper_left[1], lower_right[0] - upper_left[0], lower_right[1] - upper_left[1]);
    std::cout << "person_mask RECT " << person_rect << std::endl;
    person_rect = clamp_rect_to_frame(person_rect, depth_frame);
    std::cout << "person_mask CLAMP " << person_rect << std::endl;

    //cv::Mat person_rgb_roi = rgb_frame(person_rect);
    cv::Mat person_depth_roi = depth_frame(person_rect).clone();
    cv::Mat person_mask;
    cv::inRange(person_depth_roi, std::max(z - Z_TOLERANCE, 600.0f), z + Z_TOLERANCE, person_mask);

    //std::cout << person_depth_roi.type() << ' ' << person_mask.type() << std::endl;
    //cv::Mat ones = cv::Mat::ones(background_depth.size(), background_depth.type());
    //cv::Mat person_mask2;
    //bitwise_and(background_depth, ones, person_mask2, person_mask);
    display_mask = person_mask.clone();
    //person_mask = person_depth_roi;

    //cv::Mat flood_mask = cv::Mat::zeros(person_mask.rows + 2, person_mask.cols + 2, CV_8UC1);

    float mean_x = 0;
    float mean_y = 0;
    int n = 0;
    cout << "TYPE MASK " << type2str(person_mask.type()) << std::endl;
    //int mask_row_step = depth_frame.rows - person_mask.rows;
    for(int y = 0; y < person_mask.rows; y++){
        //uint8_t *mask_col = person_mask.at<uint8_t>(y, x);
        for(int x = 0; x < person_mask.cols; x++){
            const uint8_t pixel_in_mask = person_mask.at<uint8_t>(y, x);
            std::cout << "ASD" << int(pixel_in_mask) << std::endl;
            mean_x += x * (pixel_in_mask != 0);
            mean_y += y * (pixel_in_mask != 0);
            n += (pixel_in_mask != 0);
        }
    }

    int mean_pixel_x = 0.5 + (mean_x / n);
    int mean_pixel_y = 0.5 + (mean_y / n);

    cout << "TYPE MASK " << mean_x << ' ' << mean_y << ' ' << n << std::endl;


    const Eigen::Vector3f center_3D = point_3D_reprojection(mean_pixel_x, mean_pixel_y, z, lookupX, lookupY);

    const Vector3f point_3D_left = center_3D + Vector3f(-0.12, -1, 0.0);
    Vector2i left = point_3D_projection(point_3D_left, fx, fy, cx, cy);
    Vector2i right = Vector2i(mean_pixel_x + (mean_pixel_x - left[0]), mean_pixel_y + (mean_pixel_y - left[1]));
    //const Vector3f point_3D_right = point_3D + (0.12., 0.0, 0.0);
    //cv::line(display, cv::Point(left[0], left[1]), cv::Point(right[0], right[1]), cv::Scalar(255, 0, 0), 3);
    cv::Point offset;
    cv::Size size;
    person_depth_roi.locateROI(size, offset);

    //cv::line(display, offset + cv::Point(left[0], left[1]), offset + cv::Point(right[0], right[1]), cv::Scalar(0), 3);
    //cv::ellipse(display, offset + cv::Point(mean_pixel_x, mean_pixel_y), cv::Size(10, 10), 0, 0, 360, cv::Scalar(255, 0, 0), -3, 8, 0);

    cv::rectangle(display_mask, cv::Point(left[0], left[1]), cv::Point(right[0], right[1]), cv::Scalar(128), 3);
    //cv::ellipse(display_mask, cv::Point(mean_pixel_x, mean_pixel_y), cv::Size(10, 10), 0, 0, 360, cv::Scalar(128), -3, 8, 0);
    left[0] = std::max(0, left[0]);
    right[0] = std::min(person_mask.cols - 1, right[0]);

    left[1] = std::max(0, left[1]);
    right[1] = std::min(person_mask.rows - 1, right[1]);

    mean_x = 0;
    mean_y = 0;
    n = 0;

    for(int y = left[1]; y < right[1]; y++){
        //uint8_t *mask_col = person_mask.at<uint8_t>(y, x);
        for(int x = left[0]; x < right[0]; x++){
            const uint8_t pixel_in_mask = person_mask.at<uint8_t>(y, x);
            mean_x += x * (pixel_in_mask != 0);
            mean_y += y * (pixel_in_mask != 0);
            n += (pixel_in_mask != 0);
        }
    }

    mean_pixel_x = 0.5 + (mean_x / n);
    mean_pixel_y = 0.5 + (mean_y / n);

    //cv::ellipse(display, offset + cv::Point(mean_pixel_x, mean_pixel_y), cv::Size(10, 10), 0, 0, 360, cv::Scalar(255, 255, 0), -3, 8, 0);
    //cv::line(display, offset + cv::Point(mean_pixel_x, 0), offset + cv::Point(mean_pixel_x, person_mask.rows-1), cv::Scalar(0), 5);
    //cv::line(person_mask, cv::Point(mean_pixel_x, 0), cv::Point(mean_pixel_x, person_mask.rows-1), cv::Scalar(128), 5);
    cv::ellipse(display_mask, cv::Point(mean_pixel_x, mean_pixel_y), cv::Size(10, 10), 0, 0, 360, cv::Scalar(128), -3, 100, 0);

    std::vector<int> y_pixels(right[1] - left[1]);

    for(int y = left[1]; y < right[1]; y++){
        const uint8_t pixel_in_mask = person_mask.at<uint8_t>(y, mean_pixel_x);
        y_pixels[y - left[1]] = y * (pixel_in_mask != 0);
        //std::cout << "COSO PIX " <<  bool(pixel_in_mask) << ' ';
    }

    //std::cout << std::endl;
    std::sort(y_pixels.begin(), y_pixels.end(), std::less<int>());
    int last_zero = 0;
    for (last_zero = 0; last_zero < y_pixels.size(); last_zero++){
        if (y_pixels[last_zero] != 0){
            break;
        }
    }
    cout << "COSO LAST " << last_zero << ' '<< y_pixels.size() << ' ' << (y_pixels.size() - last_zero) * 0.25<< std::endl;

    //y_pixels.erase(y_pixels.begin(), y_pixels.begin() + last_zero);
    float sum = std::accumulate(y_pixels.begin() + last_zero, y_pixels.begin() + last_zero + int((y_pixels.size() - last_zero) * 0.25), 0);

    const int average_y = 0.5 + sum / int((y_pixels.size() - last_zero) * 0.25);
    cout <<"COSO " << average_y << std::endl;


    //cv::line(display, offset + cv::Point(left[0], average_y), offset + cv::Point(right[0], average_y), cv::Scalar(0), 5);
    cv::line(display_mask, cv::Point(left[0], average_y), cv::Point(right[0], average_y), cv::Scalar(128), 5);
    cv::ellipse(display_mask, cv::Point(mean_pixel_x, average_y), cv::Size(10, 10), 0, 0, 360, cv::Scalar(128), -3, 50, 0);


    //cv::Rect bounding_box;
    //cv::floodFill(person_mask, flood_mask, cv::Point(x, y), cv::Scalar(128), &bounding_box, cv::Scalar(20.0f), cv::Scalar(20.0f), cv::FLOODFILL_MASK_ONLY);
    //cv::Mat zero_depth = cv::Mat::zeros(person_depth_roi.rows, person_depth_roi.cols, person_depth_roi.type());
    //cv::bitwise_not(person_mask, person_mask);
    //bitwise_and(person_depth_roi, zero_depth, person_depth_roi, in_range_mask);
    return person_mask;
}
