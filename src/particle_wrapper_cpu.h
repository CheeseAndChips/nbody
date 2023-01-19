#ifndef _PARTICLE_WRAPPER_CPU_H
#define _PARTICLE_WRAPPER_CPU_H
#include "particle_set.h"
#include "particle_wrapper.h"
#include "simulation_util.h"
#include <fstream>

class particle_wrapper_cpu : public particle_wrapper
{
public:
    particle_wrapper_cpu(int32_t n, int32_t threadcnt) : particle_wrapper(n), threadcnt(threadcnt) { }
    particle_wrapper_cpu(std::istream& file, int32_t threadcnt) : particle_wrapper(file), threadcnt(threadcnt) { }
    particle_wrapper_cpu(const std::string& filename, int32_t threadcnt) : particle_wrapper(filename), threadcnt(threadcnt) { }
    particle_wrapper_cpu(const particle_set_t& pset, int32_t threadcnt) : particle_wrapper(pset), threadcnt(threadcnt) { }

    void do_timestep(simulation_settings_t& settings);
    void init_context(simulation_settings_t &settings) {
        particle_wrapper::init_context(settings);    
    }

    void exit_context() {
        do_timestep(*this->context_settings);
        particle_wrapper::exit_context();
    }
private:
    int threadcnt;
};

#endif