#ifndef CObservationImageWithModel_H
#define CObservationImageWithModel_H

#include <mrpt/obs/CObservationImage.h>
#include <mrpt/utils/CSerializable.h>
#include <mrpt/utils/CStream.h>
#include <mrpt/utils/CImage.h>
#include <mrpt/obs/CObservation.h>
#include <mrpt/otherlibs/do_opencv_includes.h>

namespace mrpt
{
namespace obs
{
    DEFINE_SERIALIZABLE_PRE_CUSTOM_BASE_LINKAGE( CObservationImageWithModel , CObservationImage,OBS_IMPEXP )

    /** Declares a class derived from "CObservation" that encapsules an image from a camera, whose relative pose to robot is also stored.
         The next figure illustrate the coordinates reference systems involved in this class:<br>
         <center>
         <img src="CObservationImage_figRefSystem.png">
         </center>
     *
     * \sa CObservation, CObservationStereoImages
     * \ingroup mrpt_obs_grp
     */
    class OBS_IMPEXP CObservationImageWithModel : public CObservationImage
    {
        // This must be added to any CSerializable derived class:
        DEFINE_SERIALIZABLE( CObservationImageWithModel )
     public:

        CObservationImageWithModel(void *iplImage = NULL);
        cv::Mat model;

    }; // End of class def.
    DEFINE_SERIALIZABLE_POST_CUSTOM_BASE_LINKAGE( CObservationImageWithModel , CObservationImage,OBS_IMPEXP )


    } // End of namespace
} // End of namespace



#include <mrpt/utils/CSerializable.h>
#include <mrpt/utils/CStartUpClassesRegister.h>


using namespace mrpt::obs;
using namespace mrpt::utils;

void registerclass_CObservationImageWithModel();

CStartUpClassesRegister  register_CObservationImageWithModel(&registerclass_CObservationImageWithModel);

void registerclass_CObservationImageWithModel()
{
    registerClass(CLASS_ID(CObservationImageWithModel));
}
const volatile int dumm = register_CObservationImageWithModel.do_nothing();

#endif
