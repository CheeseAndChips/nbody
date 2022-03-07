#ifndef _PARTICLE_WRAPPER_H
#define _PARTICLE_WRAPPER_H
#include "particle_set.h"
#include <fstream>

class particle_wrapper
{
private:
    particle_set_t pset;

    void construct_from_file(std::istream& file);

public:
    particle_wrapper(int32_t n);
    particle_wrapper(std::istream& file);
    particle_wrapper(const std::string& filename);

    void do_simulation_timestep(scalar_t deltaT, scalar_t bigG, scalar_t distanceAdded);
    void dump_to_file(std::ostream& file);
    void dump_to_file(const std::string& filename);
    void set_particle_values(int32_t i, const vec2d_t& pos, const vec2d_t& vel, scalar_t mass);
    vec2d_t get_particle_position(int32_t i);
};

#endif