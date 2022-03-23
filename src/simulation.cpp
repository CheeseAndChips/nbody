#include <vector>
#include <fstream>
#include "particle_wrapper.h"
#include "particle_wrapper_cpu.h"
#include "particle_wrapper_gpu.h"
#include "encoder_util.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <chrono>
#include "global_settings.h"

scalar_t randomflt()
{
    return (scalar_t)rand() / RAND_MAX;
}

int nbody_settings::thread_count;

const int video_width = 1920;
const int video_height = 1080;
const int N = 1024*1024;
const int fps = 30;
const int runtime = 6;
const double bigG = 1e-6/3;
const double softening = 1e-2;
const scalar_t coord_visible = 1.2f; // [-coord_visible; coord_visible] will be visible smaller axis

scalar_t scalevalue = (std::min(video_height, video_width) / 2) / coord_visible;

#define TIMETAKEN(t1, t2) (std::chrono::duration<double>(t2 - t1).count())

void perform(int totalcnt, video_encoder* enc, particle_wrapper* pset, simulation_settings_t& settings)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<uint8_t>> data(video_width, std::vector<uint8_t>(video_height));

    // pset->dump_to_file("/home/joris/nbody-big/init.dump");

    for(int T = 0; T < totalcnt; T++)
    {
        auto frame_start = std::chrono::high_resolution_clock::now();
        pset->do_timestep(settings);

        auto cuda_end = std::chrono::high_resolution_clock::now();
        for(auto& row : data){
            for(auto& cell : row) cell = 0;
        }

        for(int i = 0; i < pset->get_count(); i++){
            vec2d_t pos = pset->get_particle_position(i);
            int x = pos.x * scalevalue + video_width / 2;
            int y = pos.y * scalevalue + video_height / 2;

            if(x >= 0 && y >= 0 && x < video_width && y < video_height){
                data.at(x).at(y) = 255;
            }
        }
        auto framecreation_end = std::chrono::high_resolution_clock::now();

        if(enc != nullptr)
        {
            enc->update_pixels(data);
            enc->write_frame();
        }

        auto frame_end = std::chrono::high_resolution_clock::now();
        std::cout << "Frame #" << T << " " << TIMETAKEN(frame_start, cuda_end) << " frame creation " << TIMETAKEN(cuda_end, framecreation_end);
        std::cout << " encoding " << TIMETAKEN(framecreation_end, frame_end) << std::endl;

        // if((T + 1) % fps == 0) pset->dump_to_file("/home/joris/nbody-big/frame" + std::to_string(T+1) + ".dump"); 
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time taken: " << diff.count() << std::endl;
}

int main()
{
    nbody_settings::thread_count = 8;

    // video_encoder enc1("/media/RAMDISK/output1.264", video_width, video_height, fps, codec_settings_t("libx264", "slow", 0));
    // video_encoder enc2("/media/RAMDISK/output1.264", video_width, video_height, fps, codec_settings_t("libx264", "slow", 0));
    // video_encoder enc("/home/joris/nbody-big/1mil.264", video_width, video_height, fps, codec_settings_t("libx264", "slow", 0));
    video_encoder enc("/media/RAMDISK/test.264", video_width, video_height, fps, codec_settings_t("libx264", "slow", 0));

    // particle_set_t pset(N);
    particle_set_t pset("/home/joris/tomas.dump");

    simulation_settings_t settings(1.0/fps, bigG, softening);

    // std::cout << pset.n << std::endl;
    // std::cout << pset.mass[0] << std::endl;
    // return 0;
    /*for(int i = 0; i < N; i++)
    {
        scalar_t x = randomflt();
        scalar_t y = randomflt();

        // scale [0, 1] to [-1, 1]
        x = 2*x - 1;
        y = 2*y - 1;

        scalar_t mass = randomflt();
        mass = exp(2*mass - 1) - exp(-1);

        pset.positions[i] = vec2d_t(x, y);
        pset.velocities[i] = vec2d_t(0, 0);
        pset.mass[i] = mass;
    }*/

    // particle_wrapper_cpu psetcpu(pset);
    particle_wrapper_gpu psetgpu(pset);

    //perform(fps*runtime, enc1, &psetcpu, settings);
    // perform(fps*runtime, &enc2, &psetgpu, settings);
    perform(fps * runtime, &enc, &psetgpu, settings);
}