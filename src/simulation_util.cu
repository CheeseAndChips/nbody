#include "simulation_util.h"
#include <stdio.h>

__device__ __host__ scalar_t distance_sqr(const vec2d_t& a, const vec2d_t& b) {
    scalar_t A = a.x - b.x;
    scalar_t B = a.y - b.y;
    return A*A + B*B;
}

__device__ __host__ void simulation_timestep(int32_t j, int32_t n, const vec2d_t* positions, vec2d_t* velocities, const scalar_t* mass, const simulation_settings_t& settings) {
    scalar_t coeff = settings.bigG * settings.deltaT;
    scalar_t jx = positions[j].x;
    scalar_t jy = positions[j].y;
    for(int32_t i = 0; i < n; i++) {
        scalar_t x = (positions[i].x - jx);
        scalar_t y = (positions[i].y - jy);

        scalar_t distance = x*x + y*y + settings.distanceAdded;
        scalar_t invDistance = rsqrtf(distance);
        scalar_t invDistanceCube = invDistance * invDistance * invDistance;

        scalar_t s = coeff * mass[i] * invDistanceCube;
        velocities[j].x += x * s;
        velocities[j].y += y * s;
    }
}

__device__ __host__ void update_positions(int32_t j, vec2d_t* positions, vec2d_t* velocities, scalar_t deltaT)
{
    positions[j].x += velocities[j].x * deltaT;
    positions[j].y += velocities[j].y * deltaT;
}

void simulation_cpu(int32_t n, int32_t threadnum, int32_t threadcnt, vec2d_t* positions, vec2d_t* velocities, scalar_t* mass, simulation_settings_t& settings)
{
    for(int i = threadnum; i < n; i += threadcnt){
        simulation_timestep(i, n, positions, velocities, mass, settings);
    }
}

void posupdate_cpu(int32_t n, int32_t threadnum, int32_t threadcnt, vec2d_t* positions, vec2d_t* velocities, simulation_settings_t& settings)
{
    for(int i = threadnum; i < n; i += threadcnt){
        update_positions(i, positions, velocities, settings.deltaT);
    }
}

__global__ void simulation_gpu(int32_t n, vec2d_t* positions, vec2d_t* velocities, scalar_t* mass, simulation_settings_t settings)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < n;
        i += blockDim.x * gridDim.x){
            simulation_timestep(i, n, positions, velocities, mass, settings);
        }
}

__global__ void posupdate_gpu(int32_t n, vec2d_t* positions, vec2d_t* velocities, simulation_settings_t settings)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < n;
        i += blockDim.x * gridDim.x){
            update_positions(i, positions, velocities, settings.deltaT);
        }
}