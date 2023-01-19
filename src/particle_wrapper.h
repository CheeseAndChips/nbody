#ifndef _PARTICLE_WRAPPER_H
#define _PARTICLE_WRAPPER_H
#include "camera.h"
#include "particle_set.h"
#include <fstream>
#include <memory>
#include <iostream>
#include "simulation_util.h"
#include "simulation_context.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

class particle_wrapper
{
protected:
    particle_set_t pset;
    simulation_settings_t *context_settings;
    bool context_initialized = false;

public:
    particle_wrapper(int32_t n);
    particle_wrapper(const particle_set_t& pset);

    void dump_to_file(std::ostream& file) { pset.dump_to_file(file); }
    void dump_to_file(const std::string& filename) { pset.dump_to_file(filename); }

    int get_count() const { return pset.n; }

    virtual void set_particle_values(int32_t i, const vec2d_t& pos, const vec2d_t& vel, scalar_t mass);
    virtual vec2d_t get_particle_position(int32_t i);

    virtual void do_timestep(simulation_settings_t& settings) = 0;

    virtual void init_context(simulation_settings_t &settings) {
        this->context_settings = new simulation_settings_t(settings);
        this->context_initialized = true;
    }

    virtual void exit_context() {
        if(!this->context_initialized) {
            std::cerr << "Exiting non-initialized context\n";
            exit(1);
        }

        delete this->context_settings;
        this->context_settings = nullptr;
        this->context_initialized = false;
    }

    void write_to_array(boost::python::numpy::ndarray &arr, const camera_settings_t &camera);
    simulation_context_t get_context(const simulation_settings_t &settings) {
        return simulation_context_t(this, settings);
    }
};

#endif