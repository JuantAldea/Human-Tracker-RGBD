#pragma once


constexpr float SQRT_2PI = sqrt(2.f * M_PI);

double TRANSITION_MODEL_STD_XY   = 0;
double TRANSITION_MODEL_STD_VXY  = 0;
double NUM_PARTICLES             = 0;

const float LIKEHOOD_FOUND  = 0.7;
const float LIKEHOOD_UPDATE = 0.9;
const float SIGMA_COLOR = 0.3;
const float ELLIPSE_FITTING_ANGLE_STEP = 4;
const float PERCENTAGE = 0.8;

using DEPTH_DATA_TYPE = uint16_t;
