#ifndef _PARTICLE_H
#define _PARTICLE_H

#include <utility>
#include <cstdint>

typedef float scalar_t;

struct vec2d_t
{
    scalar_t x, y;
    vec2d_t() : x(0), y(0) { }
    vec2d_t(scalar_t x, scalar_t y) : x(x), y(y) { }
};

class particle_set_t
{
public:
    int32_t n;
    vec2d_t* positions;
    vec2d_t* velocities;
    scalar_t* mass;

    particle_set_t() : n(0), positions(nullptr), velocities(nullptr), mass(nullptr) { }

    particle_set_t(int32_t n) : n(n)
    {
        positions = new vec2d_t[n];
        velocities = new vec2d_t[n];
        mass = new scalar_t[n];
    }

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

    void set_particle_pos(const vec2d_t& pos, int32_t i);
    void simulation_timestep(int32_t particleindex, scalar_t deltaT, scalar_t gravitationalConstant, scalar_t distanceAdded);
    void update_particle_position(int32_t particleindex, scalar_t deltaT);
    void threaded_timestep(int32_t threadnum, int32_t threadcnt, scalar_t deltaT, scalar_t gravitationalConstant, scalar_t distanceAdded);
    void threaded_position_update(int32_t threadnum, int32_t threadcnt, scalar_t deltaT);
};

#endif