#ifndef _SIMULATION_UTIL_H
#define _SIMULATION_UTIL_H
#include "particle_set.h"
#include "math.h"
#include <stdio.h>
#include <iostream>

#ifdef USING_CUDA
#define THREAD_COUNT 256

#include <cuda.h>
#include <cuda_runtime_api.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) exit(code);
   }
}
#endif

#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __host__
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
#ifdef USING_CUDA
__global__ void simulation_gpu(int32_t n, vec2d_t* positions, vec2d_t* velocities, scalar_t* mass, simulation_settings_t settings);
__global__ void posupdate_gpu(int32_t n, vec2d_t* positions, vec2d_t* velocities, simulation_settings_t settings);
#endif

#endif