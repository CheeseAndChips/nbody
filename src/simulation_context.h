#ifndef _SIMULATION_CONTEXT_H
#define _SIMULATION_CONTEXT_H
#include <boost/python.hpp>
#include "simulation_util.h"

class particle_wrapper;

class simulation_context_t {
private:
    particle_wrapper *wrapper;
    simulation_settings_t settings;

public:
    simulation_context_t(particle_wrapper *wrapper, const simulation_settings_t& settings) : wrapper(wrapper), settings(settings) { }
    boost::python::object enter_context();
    void exit_context(boost::python::object a, boost::python::object b, boost::python::object c);
};

#endif