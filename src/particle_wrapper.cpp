#include "particle_wrapper.h"
#include "file_util.h"

particle_wrapper::particle_wrapper(int32_t n) : pset(particle_set_t(n)) { }

particle_wrapper::particle_wrapper(const particle_set_t& pset) : pset(pset) { }

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

void particle_wrapper::write_to_array(boost::python::numpy::ndarray &arr, const camera_settings_t &camera) {
    auto shape = arr.get_shape();
    auto height = shape[0];
    auto width = shape[1];

    if(arr.get_nd() != 2) {
        std::cerr << "Bad array shape" << std::endl;
        exit(1);
    }

    char *data = arr.get_data();
    memset(data, 0, height*width);

    for(int i = 0; i < get_count(); i++){
        auto pos = get_particle_position(i);
        int x = (pos.x - camera.center.x) * camera.zoom + width / 2;
        int y = (pos.y - camera.center.y) * camera.zoom + height / 2;

        if(x < 0 || y < 0) continue;
        if(x >= width || y >= height) continue;
        arr[y][x] = 255;
    }
}