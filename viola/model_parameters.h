#pragma once

constexpr float SQRT_2PI = sqrt(2.f * M_PI);

double TRANSITION_MODEL_STD_XY   = 0;
double TRANSITION_MODEL_STD_VXY  = 0;
double NUM_PARTICLES             = 0;

constexpr float LIKEHOOD_FOUND  = 0.5;
constexpr float LIKEHOOD_UPDATE = 0.9;

constexpr float ELLIPSE_FITTING_ANGLE_STEP = 4;

constexpr float MODEL_AXIS_X_METTERS = 0.12;
constexpr float MODEL_AXIS_Y_METTERS = 0.12;

constexpr float MODEL_SEMIAXIS_X_METTERS = MODEL_AXIS_X_METTERS * 0.5;
constexpr float MODEL_SEMIAXIS_Y_METTERS = MODEL_AXIS_Y_METTERS * 0.5;

using DEPTH_TYPE = uint16_t;
