#include "particle_set.h"
#include <math.h>
#include <iostream>
#include "file_util.h"

void particle_set_t::allocate_arrays()
{
    if(positions != nullptr) delete[] positions;
    if(velocities != nullptr) delete[] velocities;
    if(mass != nullptr) delete[] mass;

    positions = new vec2d_t[n];
    velocities = new vec2d_t[n];
    mass = new scalar_t[n];
}

void particle_set_t::construct_from_file(std::istream& file)
{
    readFromFile(file, n);
    allocate_arrays();
    for(int i = 0; i < n; i++) readFromFile(file, positions[i]);
    for(int i = 0; i < n; i++) readFromFile(file, velocities[i]);
    for(int i = 0; i < n; i++) readFromFile(file, mass[i]);
}

void particle_set_t::dump_to_file(std::ostream& file)
{
    writeToFile(file, n);
    for(int i = 0; i < n; i++) writeToFile(file, positions[i]);
    for(int i = 0; i < n; i++) writeToFile(file, velocities[i]);
    for(int i = 0; i < n; i++) writeToFile(file, mass[i]);
}

void particle_set_t::dump_to_file(const std::string& filename)
{
    std::ofstream outfile(filename, std::fstream::out | std::fstream::binary);
    dump_to_file(outfile);
}

particle_set_t::particle_set_t(std::istream& file) { construct_from_file(file); }

particle_set_t::particle_set_t(const std::string& filename)
{
    std::ifstream file(filename);
    construct_from_file(file);
}