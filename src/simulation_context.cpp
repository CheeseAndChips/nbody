#include "simulation_context.h"
#include "particle_wrapper.h"
#include <iostream>

boost::python::object simulation_context_t::enter_context() {
    this->wrapper->do_timestep(this->settings);
    return boost::python::object(this);
}

void simulation_context_t::exit_context(boost::python::object a, boost::python::object b, boost::python::object c) {
    this->wrapper->wait_for_lock();
}