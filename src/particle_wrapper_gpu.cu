#ifdef USING_CUDA

#include "particle_wrapper_gpu.h"
#include <iostream>

void particle_wrapper_gpu::setup_cuda_memory() {
    int n = pset.n;
    padded_n = n + (n % THREAD_COUNT != 0 ? THREAD_COUNT : 0); 

    cudaMalloc(&d_positions, padded_n * sizeof(vec2d_t));
    cudaMalloc(&d_velocities, padded_n * sizeof(vec2d_t));
    cudaMalloc(&d_mass, padded_n * sizeof(scalar_t));

    cudaMemset(d_mass, 0, padded_n * sizeof(scalar_t)); // ensure that padded particles have mass of 0
}

void particle_wrapper_gpu::host_to_device()
{
    auto n = pset.n;
    cudaMemcpy(d_positions, pset.positions, n * sizeof(vec2d_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, pset.velocities, n * sizeof(vec2d_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, pset.mass, n * sizeof(scalar_t), cudaMemcpyHostToDevice);
    device_outdated = false;
}

void particle_wrapper_gpu::device_to_host()
{
    auto n = pset.n;
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
    wait_for_lock();
    this->ongoing_calculation = true;    
    if(this->device_outdated) host_to_device();

    int blockcnt = padded_n / THREAD_COUNT;
    simulation_gpu<<<blockcnt, THREAD_COUNT>>>(padded_n, d_positions, d_velocities, d_mass, settings);
    posupdate_gpu<<<blockcnt, THREAD_COUNT>>>(padded_n, d_positions, d_velocities, settings);
    this->host_outdated = true;
}

void particle_wrapper_gpu::set_particle_values(int32_t i, const vec2d_t& pos, const vec2d_t& vel, scalar_t mass)
{
    if(ongoing_calculation) {
        PyErr_SetString(PyExc_BlockingIOError, "Trying to change particle values with ongoing calculations");
        return;
    }

    if(this->host_outdated) device_to_host();
    particle_wrapper::set_particle_values(i, pos, vel, mass);
    this->device_outdated = true;
}

vec2d_t particle_wrapper_gpu::get_particle_position(int32_t i)
{
    if(!this->ongoing_calculation && this->host_outdated) device_to_host();
    return particle_wrapper::get_particle_position(i);
}

void particle_wrapper_gpu::wait_for_lock()
{
    if(ongoing_calculation) {
        cudaDeviceSynchronize();
        ongoing_calculation = false;
        device_to_host();
    }
}

#endif