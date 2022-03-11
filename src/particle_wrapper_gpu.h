#ifndef _PARTICLE_WRAPPER_GPU_H
#define _PARTICLE_WRAPPER_GPU_H
#include "particle_wrapper.h"
#include "simulation_util.h"
#include <fstream>
#include <string>

class particle_wrapper_gpu : public particle_wrapper {
private:
    vec2d_t* d_positions;
    vec2d_t* d_velocities;
    scalar_t* d_mass;

    void setup_cuda_memory();
    void host_to_device();
    void device_to_host();

    bool device_outdated = false;
    bool host_outdated = false;
public:
    particle_wrapper_gpu(int n) : particle_wrapper(n) { setup_cuda_memory(); }
    particle_wrapper_gpu(std::istream& file) : particle_wrapper(file) { setup_cuda_memory(); }
    particle_wrapper_gpu(const std::string& filename) : particle_wrapper(filename) { setup_cuda_memory(); }
    particle_wrapper_gpu(const particle_set_t& pset) : particle_wrapper(pset) { setup_cuda_memory(); }

    particle_wrapper_gpu(const particle_wrapper_gpu& other) = delete;
    particle_wrapper_gpu(particle_wrapper_gpu&& other) = delete;
    particle_wrapper_gpu& operator=(const particle_wrapper_gpu& other) = delete;
    particle_wrapper_gpu& operator=(particle_wrapper_gpu&& other) = delete;

    ~particle_wrapper_gpu();

    void set_particle_values(int32_t i, const vec2d_t& pos, const vec2d_t& vel, scalar_t mass);
    vec2d_t get_particle_position(int32_t i);

    void do_timestep(simulation_settings_t& settings);
};

__global__ void simulation_timestep_kernel(particle_set_t* pset);

#endif