#include "particle_wrapper.h"
#include "file_util.h"

particle_wrapper::particle_wrapper(int32_t n) : pset(particle_set_t(n)) { }

particle_wrapper::particle_wrapper(const particle_set_t& pset) : pset(pset) { }

void particle_wrapper::set_particle_values(int32_t i, const vec2d_t& pos, const vec2d_t& vel, scalar_t mass)
{
    pset.positions[i] = pos;
    pset.velocities[i] = vel;
    pset.mass[i] = mass;
}

vec2d_t particle_wrapper::get_particle_position(int32_t i)
{
    return pset.positions[i];
}