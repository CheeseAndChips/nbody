#ifndef _PARTICLE_WRAPPER_CPU_H
#define _PARTICLE_WRAPPER_CPU_H
#include "particle_set.h"
#include "particle_wrapper.h"
#include "simulation_util.h"
#include <fstream>

class particle_wrapper_cpu : public particle_wrapper
{
public:
    particle_wrapper_cpu(int32_t n) : particle_wrapper(n) { }
    particle_wrapper_cpu(std::istream& file) : particle_wrapper(file) { }
    particle_wrapper_cpu(const std::string& filename) : particle_wrapper(filename) { }
    particle_wrapper_cpu(const particle_set_t& pset) : particle_wrapper(pset) { }

    void do_timestep(simulation_settings_t& settings);
};

#endif