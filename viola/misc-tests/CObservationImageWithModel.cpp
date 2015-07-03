
#include <mrpt/config.h>

#include "CObservationImageWithModel.h"

using namespace mrpt::obs;
using namespace mrpt::utils;

// This must be added to any CSerializable class implementation file.
IMPLEMENTS_SERIALIZABLE(CObservationImageWithModel, CObservationImage,mrpt::obs)

CObservationImageWithModel::CObservationImageWithModel(void *iplImage) :
    CObservationImage(iplImage)
{
    ;
}

void CObservationImageWithModel::readFromStream(mrpt::utils::CStream &in, int version)
{
	CObservationImage::readFromStream(in, version);
	model = cv::Mat::zeros(100, 100, CV_8UC1);
}

void CObservationImageWithModel::writeToStream(mrpt::utils::CStream &out, int *version) const
{
	CObservationImage::writeToStream(out, version);
}