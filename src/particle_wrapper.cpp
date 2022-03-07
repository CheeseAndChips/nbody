#include "particle_wrapper.h"
#include "file_util.h"

void particle_wrapper::construct_from_file(std::istream& file)
{
    int32_t n;
    readFromFile(file, n);
    pset = particle_set_t(n);
}

particle_wrapper::particle_wrapper(int32_t n) : pset(particle_set_t(n)) { }

particle_wrapper::particle_wrapper(std::istream& file) { construct_from_file(file); }

particle_wrapper::particle_wrapper(const std::string& filename)
{
    std::ifstream file(filename);
    construct_from_file(file);
}


void particle_wrapper::do_simulation_timestep(scalar_t deltaT, scalar_t bigG, scalar_t distanceAdded)
{
    pset.simulation_timestep(deltaT, bigG, distanceAdded);
}

void particle_wrapper::dump_to_file(std::ostream& file)
{
    writeToFile(file, pset.n);
    for(int i = 0; i < pset.n; i++) writeToFile(file, pset.positions[i]);
    for(int i = 0; i < pset.n; i++) writeToFile(file, pset.velocities[i]);
    for(int i = 0; i < pset.n; i++) writeToFile(file, pset.mass[i]);
}

void particle_wrapper::dump_to_file(const std::string& filename)
{
    std::ofstream outfile(filename, std::fstream::out | std::fstream::binary);
    dump_to_file(outfile);
}
void particle_wrapper::set_particle_values(int32_t i, const vec2d_t& pos, const vec2d_t& vel, scalar_t mass)
{
    pset.positions[i] = pos;
    pset.velocities[i] = vel;
    pset.mass[i] = mass;
}

vec2d_t particle_wrapper::get_particle_position(int32_t i)
{
    return pset.positions[i];
}