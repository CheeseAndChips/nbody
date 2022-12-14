#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "simulation_util.h"
#include "particle_wrapper_cpu.h"
#include "encoder_util.h"
#include "particle_wrapper.h"
#ifdef USING_CUDA
#include "particle_wrapper_gpu.h"
#endif

namespace p = boost::python;
namespace np = boost::python::numpy;

BOOST_PYTHON_MODULE(nbody)
{
    Py_Initialize();
    np::initialize();
    using namespace boost::python;
    class_<simulation_settings_t>("SimulationSettings", init<scalar_t, scalar_t, scalar_t>())
        .def_readwrite("deltaT", &simulation_settings_t::deltaT)
        .def_readwrite("bigG", &simulation_settings_t::bigG)
        .def_readwrite("distanceAdded", &simulation_settings_t::distanceAdded)
    ;

    class_<codec_settings_t>("CodecSettings", init<const std::string&, int64_t>())
        .def(init<const std::string&, const std::string&, int>())
    ;

    class_<vec2d_t>("Vec2D", init<>())
        .def(init<scalar_t, scalar_t>())
        .def_readwrite("x", &vec2d_t::x)
        .def_readwrite("y", &vec2d_t::y)
    ;

    class_<video_encoder, boost::noncopyable>("VideoEncoder", init<const std::string&, int, int, int, const codec_settings_t&>())
        .def("update_pixels", &video_encoder::update_pixels)
        .def("write_frame", &video_encoder::write_frame)
        .def("write_wrapper", &video_encoder::write_from_wrapper)
        .def("write_array", &video_encoder::write_array)
        .def("generate_pixels", &video_encoder::generate_pixels)
    ;

    class_<camera_settings_t>("CameraSettings", init<vec2d_t, scalar_t>())
        .def_readwrite("center", &camera_settings_t::center)
        .def_readwrite("zoom", &camera_settings_t::zoom)
    ;

    class_<particle_wrapper, boost::noncopyable>("ParticleWrapper", boost::python::no_init);

    class_<particle_wrapper_cpu, bases<particle_wrapper>>("ParticleWrapperCPU", init<int32_t, int32_t>())
        .def(init<const std::string&, int32_t>())
        .def("do_timestep", &particle_wrapper_cpu::do_timestep)
        .def("set_particle", &particle_wrapper::set_particle_values)
    ;
}