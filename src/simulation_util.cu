#include "simulation_util.h"
#include <stdio.h>

__device__ __host__ scalar_t distance_sqr(const vec2d_t& a, const vec2d_t& b) {
    scalar_t A = a.x - b.x;
    scalar_t B = a.y - b.y;
    return A*A + B*B;
}

__device__ __host__ void simulation_timestep(int32_t n, const vec2d_t& ppos, vec2d_t& pvel, const vec2d_t* positions, const scalar_t* mass, const simulation_settings_t& settings) {
    scalar_t coeff = settings.bigG * settings.deltaT;
    scalar_t jx = ppos.x;
    scalar_t jy = ppos.y;
    for(int32_t i = 0; i < n; i++) {
        scalar_t x = (positions[i].x - jx);
        scalar_t y = (positions[i].y - jy);

        scalar_t distance = x*x + y*y + settings.distanceAdded;
        scalar_t invDistance = rsqrtf(distance);
        scalar_t invDistanceCube = invDistance * invDistance * invDistance;

        scalar_t s = coeff * mass[i] * invDistanceCube;
        pvel.x += x * s;
        pvel.y += y * s;
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
        simulation_timestep(n, positions[i], velocities[i], positions, mass, settings);
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
    __shared__ vec2d_t poscache[THREAD_COUNT];
    __shared__ scalar_t masscache[THREAD_COUNT];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for(int j = 0; j < n / THREAD_COUNT; j++){
        int k = j * THREAD_COUNT + threadIdx.x;

        poscache[threadIdx.x] = positions[k];
        masscache[threadIdx.x] = mass[k];
        __syncthreads();
        simulation_timestep(THREAD_COUNT, positions[i], velocities[i], poscache, masscache, settings);
    }
}

__global__ void posupdate_gpu(int32_t n, vec2d_t* positions, vec2d_t* velocities, simulation_settings_t settings)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        update_positions(i, positions, velocities, settings.deltaT);
    }
}