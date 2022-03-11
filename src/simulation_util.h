#ifndef _SIMULATION_UTIL_H
#define _SIMULATION_UTIL_H
#include "particle_set.h"
#include "math.h"

#ifndef __CUDACC__
#define __device__
#define __global__
#define __host__
#endif

struct simulation_settings_t{
    scalar_t deltaT;
    scalar_t bigG;
    scalar_t distanceAdded;

    simulation_settings_t(scalar_t deltaT, scalar_t bigG, scalar_t distanceAdded) : deltaT(deltaT), bigG(bigG), distanceAdded(distanceAdded) { }
};

void simulation_cpu(int32_t n, int32_t threadnum, int32_t threadcnt, vec2d_t* positions, vec2d_t* velocities, scalar_t* mass, simulation_settings_t& settings);
void posupdate_cpu(int32_t n, int32_t threadnum, int32_t threadcnt, vec2d_t* positions, vec2d_t* velocities, simulation_settings_t& settings);
#ifdef __CUDACC__
__global__ void simulation_gpu(int32_t n, vec2d_t* positions, vec2d_t* velocities, scalar_t* mass, simulation_settings_t settings);
__global__ void posupdate_gpu(int32_t n, vec2d_t* positions, vec2d_t* velocities, simulation_settings_t settings);
#endif


// __device__ __host__ scalar_t distance_sqr(const vec2d_t& a, const vec2d_t& b);
// __device__ __host__ void simulation_timestep(int32_t j, int32_t n, vec2d_t* positions, vec2d_t* velocities, scalar_t* mass, simulation_settings_t* settings);
// __device__ __host__ void update_positions(int32_t j, vec2d_t* positions, vec2d_t* velocities, scalar_t deltaT);


#endif