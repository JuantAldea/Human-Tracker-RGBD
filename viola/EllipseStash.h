#pragma once
#include <tuple>
#include <map>

using namespace std;

#include <mrpt/otherlibs/do_opencv_includes.h>
#include "EllipseFunctions.h"
#include "ImageRegistration.h"
#include "ModelParameters.h"
/*
enum class BodyPart
{
    HEAD = 0,
    TORSO = 1,
    CD = 2
};
*/

/*
#define STARTENUM() constexpr const int enumStart = __LINE__;
#define ENUMITEM(Name) \
struct Name {\
    static constexpr const int id = __LINE__ - enumStart - 1;\
    static constexpr const char* name = #Name;\
};

namespace BodyPart2 {
STARTENUM()
ENUMITEM(HEAD2)
ENUMITEM(TORSO2)
ENUMITEM(CD2)
}
*/

#define BodyPartMacro(FOO) \
FOO(HEAD) \
FOO(TORSO) \
FOO(CD)

#define DO_DESCRIPTION(e)  #e,
#define DO_ENUM(e)  e,

char* BodyPart_description[] = {
    BodyPartMacro(DO_DESCRIPTION)
};

enum class BodyPart {
    BodyPartMacro(DO_ENUM)
};


class EllipseStash
{
public:
    using EllipseData = std::tuple<cv::Mat, cv::Mat, cv::Mat, int>;
    using EllipseDepthMap = std::map<int, EllipseData>;

    inline EllipseData &get_ellipse(const BodyPart part, const float depth)
    {
        const int depth_rounded = cvRound(depth);
        EllipseDepthMap &part_map = body_part_ellipses[part];
        EllipseDepthMap::iterator it = part_map.find(depth_rounded);
        if (it == part_map.end()){
            EllipseData &d = part_map[depth_rounded];
            d = build_ellipse(part, depth);

            //printf("%s MODELO NUEVO %d\n", BodyPart_description[(int)part], depth_rounded);
            return d;
        }

        //printf("%s MODELO REUSADO %d\n", BodyPart_description[(int)part], depth_rounded);
        return it->second;
    };

    inline const cv::Mat &get_ellipse_mask_1D(const BodyPart part, const float depth)
    {
        const EllipseData &e = get_ellipse(part, depth);
        return get<0>(e);
    };

    inline const cv::Mat &get_ellipse_mask_3D(const BodyPart part, const float depth)
    {
        const EllipseData &e = get_ellipse(part, depth);
        return get<1>(e);
    };

    inline const cv::Mat &get_ellipse_mask_weights(const BodyPart part, const float depth)
    {
        const EllipseData &e = get_ellipse(part, depth);
        return get<2>(e);
    };

    inline int get_ellipse_pixels(const BodyPart part, const float depth)
    {
        const EllipseData &e = get_ellipse(part, depth);
        return int(get<3>(e));
    };

    inline cv::Rect get_ellipse_rect(const BodyPart part, const float depth)
    {
        const EllipseData &e = get_ellipse(part, depth);
        const cv::Mat &m = get<0>(e);
        return cv::Rect(0, 0, m.cols, m.rows);
    };

    inline cv::Size get_ellipse_size(const BodyPart part, const float depth)
    {
        EllipseData &e = get_ellipse(part, depth);
        cv::Mat &m = get<0>(e);
        return cv::Size(m.cols, m.rows);
    };


    inline EllipseStash(const ImageRegistration &r)
    {
        reg = r;
    }

protected:

    inline EllipseData build_ellipse(const BodyPart part, const int depth)
    {
        const float model_semiaxis_x = (part == BodyPart::TORSO) ? PERSON_TORSO_X_AXIS_METTERS : PERSON_HEAD_X_SEMIAXIS_METTERS;
        const float model_semiaxis_y = (part == BodyPart::TORSO) ? PERSON_TORSO_Y_AXIS_METTERS : PERSON_HEAD_Y_SEMIAXIS_METTERS;

        const float cx = reg.cameraMatrixColor.at<double>(0, 2);
        const float cy = reg.cameraMatrixColor.at<double>(1, 2);

        Eigen::Vector2i top_corner, bottom_corner;
        std::tie(top_corner, bottom_corner) = project_model(Eigen::Vector2f(cx, cy), depth,
                                          Eigen::Vector2f(model_semiaxis_x, model_semiaxis_y), reg.cameraMatrixColor,
                                          reg.lookupX, reg.lookupY);
        int n_pixels;
        cv::Rect region = cv::Rect(top_corner[0], top_corner[1], bottom_corner[0] - top_corner[0],
                                bottom_corner[1] - top_corner[1]);
        cv::Mat e3d = fast_create_ellipse_mask(region, 3, n_pixels);
        cv::Mat e1d = cv::Mat(e3d.rows, e3d.cols, CV_8UC1);
        int from_to[] = {0,0};
        mixChannels(&e3d, 1, &e1d, 1, from_to, 1);
        cv::Mat ew1d = create_ellipse_weight_mask(e1d);
        return make_tuple(e1d, e3d, ew1d, n_pixels);
    };

    std::map<BodyPart, EllipseDepthMap> body_part_ellipses;
    ImageRegistration reg;
};

#include "BoostSerializers.h"

class EllipseStashLoader : public EllipseStash
{
public:
    using EllipseData = std::tuple<cv::Mat, cv::Mat, cv::Mat, int>;
    using EllipseDepthMap = std::map<int, EllipseData>;

    EllipseStashLoader(const ImageRegistration &r) :
        EllipseStash(r)
    {
        ;
    }

    EllipseStashLoader(const ImageRegistration &r, const std::vector<BodyPart> &parts, const std::vector<std::string> &filenames) :
        EllipseStash(r)
    {
        assert(parts.size() == filenames.size());

        for (size_t i = 0; i < parts.size(); i++) {
            EllipseDepthMap &part_map = body_part_ellipses[parts[i]];
            std::ifstream ifs(filenames[i]);
            if (ifs.is_open()) {
                std::cout << "Reading file " << filenames[i] << " for boddy part " << BodyPart_description[(int)parts[i]] << std::endl;
                boost::archive::binary_iarchive ar(ifs);
                ar & part_map;
                ifs.close();
            } else {
                std::cout << "Generating ellipses for body part: " << BodyPart_description[(int)parts[i]] << std::endl;
                for (int z = 0; z < 5000; z++){
                    this->get_ellipse(parts[i], z);
                }
            }
        }
    };
};
