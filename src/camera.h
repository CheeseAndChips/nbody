#ifndef _CAMERA_H
#define _CAMERA_H
#include "particle_set.h"

struct camera_settings_t {
    vec2d_t center;
    scalar_t zoom;
    camera_settings_t() : center(vec2d_t(0, 0)), zoom(1.0) { }
    camera_settings_t(vec2d_t center, scalar_t zoom) : center(center), zoom(zoom) { }
};

#endif