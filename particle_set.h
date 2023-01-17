#ifndef _PARTICLE_SET_H
#define _PARTICLE_SET_H

#include <utility>
#include <cstdint>
#include <fstream>

typedef float scalar_t;

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

struct vec2d_t
{
    scalar_t x, y;
    vec2d_t() = default;
    __host__ __device__ vec2d_t(scalar_t x, scalar_t y) : x(x), y(y) { }
};

class particle_set_t
{
private:
    void allocate_arrays();
    void construct_from_file(std::istream& file);
public:
    int32_t n = 0;
    vec2d_t* positions = nullptr;
    vec2d_t* velocities = nullptr;
    scalar_t* mass = nullptr;

    particle_set_t() = default;

    particle_set_t(int32_t n) : n(n) { allocate_arrays(); }

    particle_set_t(std::istream& file);
    particle_set_t(const std::string& filename);

    void dump_to_file(std::ostream& file);
    void dump_to_file(const std::string& filename);

    ~particle_set_t()
    {
        if(positions != nullptr) delete[] positions;
        if(velocities != nullptr) delete[] velocities;
        if(mass != nullptr) delete[] mass;
    }

    particle_set_t(const particle_set_t& other) : particle_set_t(other.n)
    {
        for(int32_t i = 0; i < n; i++){
            positions[i] = other.positions[i];
            velocities[i] = other.velocities[i];
            mass[i] = other.mass[i];
        }
    }

    particle_set_t(particle_set_t&& other) noexcept
    {
        n = std::exchange(other.n, 0);
        positions = std::exchange(other.positions, nullptr);
        velocities = std::exchange(other.velocities, nullptr);
        mass = std::exchange(other.mass, nullptr);
    }

    particle_set_t& operator=(const particle_set_t& other)
    {
        return *this = particle_set_t(other);
    }
    
    particle_set_t& operator=(particle_set_t&& other) noexcept
    {
        std::swap(this->n, other.n);
        std::swap(this->positions, other.positions);
        std::swap(this->velocities, other.velocities);
        std::swap(this->mass, other.mass);
        return *this;
    }
};

#endif