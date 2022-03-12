#include "particle_wrapper_gpu.h"
#include <iostream>

#ifdef __CUDACC__

void particle_wrapper_gpu::setup_cuda_memory() {
    int n = pset.n;
    cudaMalloc(&d_positions, n * sizeof(vec2d_t));
    cudaMalloc(&d_velocities, n * sizeof(vec2d_t));
    cudaMalloc(&d_mass, n * sizeof(scalar_t));
    device_outdated = true;
}

void particle_wrapper_gpu::host_to_device()
{
    int n = pset.n;
    cudaMemcpy(d_positions, pset.positions, n * sizeof(vec2d_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, pset.velocities, n * sizeof(vec2d_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, pset.mass, n * sizeof(scalar_t), cudaMemcpyHostToDevice);
    device_outdated = false;
}

void particle_wrapper_gpu::device_to_host()
{
    int n = pset.n;
    cudaMemcpy(pset.positions, d_positions, n * sizeof(vec2d_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(pset.velocities, d_velocities, n * sizeof(vec2d_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(pset.mass, d_mass, n * sizeof(scalar_t), cudaMemcpyDeviceToHost);
    host_outdated = false;
}

particle_wrapper_gpu::~particle_wrapper_gpu()
{
    cudaFree(d_positions);
    cudaFree(d_velocities);
    cudaFree(d_mass);
}

void particle_wrapper_gpu::do_timestep(simulation_settings_t& settings)
{
    if(this->device_outdated) host_to_device();

    int blockcnt = (pset.n + THREAD_COUNT - 1) / THREAD_COUNT;
    simulation_gpu<<<blockcnt, THREAD_COUNT>>>(pset.n, d_positions, d_velocities, d_mass, settings);
    posupdate_gpu<<<blockcnt, THREAD_COUNT>>>(pset.n, d_positions, d_velocities, settings);
    this->host_outdated = true;
    cudaDeviceSynchronize();
    device_to_host();
}

void particle_wrapper_gpu::set_particle_values(int32_t i, const vec2d_t& pos, const vec2d_t& vel, scalar_t mass)
{
    this->device_outdated = true;
    particle_wrapper::set_particle_values(i, pos, vel, mass);
}

vec2d_t particle_wrapper_gpu::get_particle_position(int32_t i)
{
    if(this->host_outdated) device_to_host();
    return particle_wrapper::get_particle_position(i);
}

#endif