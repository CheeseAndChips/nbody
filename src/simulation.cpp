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
const int N = 4;
const int fps = 60;
const int runtime = 10;
const double bigG = 1e-1;
const double softening = 1e-2;
const scalar_t coord_visible = 3.0f; // [-coord_visible; coord_visible] will be visible smaller axis

scalar_t scalevalue = (std::min(video_height, video_width) / 2) / coord_visible;


void perform(int totalcnt, video_encoder& enc, particle_wrapper* pset, simulation_settings_t& settings)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<uint8_t>> data(video_width, std::vector<uint8_t>(video_height));

    for(int T = 0; T < totalcnt; T++)
    {
        auto frame_start = std::chrono::high_resolution_clock::now();
        pset->do_timestep(settings);

        for(auto& row : data){
            for(auto& cell : row) cell = 0;
        }

        for(int i = 0; i < N; i++){
            vec2d_t pos = pset->get_particle_position(i);
            int x = pos.x * scalevalue + video_width / 2;
            int y = pos.y * scalevalue + video_height / 2;

            if(x >= 0 && y >= 0 && x < video_width && y < video_height){
                data.at(x).at(y) = 255;
            }
        }

        enc.update_pixels(data);
        enc.write_frame();

        auto frame_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> timetaken = frame_end - frame_start;
        std::cout << "Time taken for entire frame: " << timetaken.count() << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time taken: " << diff.count() << std::endl;
}

int main()
{
    nbody_settings::thread_count = 1;

    video_encoder enc1("/media/RAMDISK/output1.264", video_width, video_height, fps, codec_settings_t("libx264", "slow", 0));
    video_encoder enc2("/media/RAMDISK/output2.264", video_width, video_height, fps, codec_settings_t("libx264", "slow", 0));

    particle_set_t pset(N);

    simulation_settings_t settings(1.0/fps, bigG, softening);

    for(int i = 0; i < N; i++)
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
    }

    particle_wrapper_cpu psetcpu(pset);
    particle_wrapper_gpu psetgpu(pset);

    perform(fps*runtime, enc1, &psetcpu, settings);
    perform(fps*runtime, enc2, &psetgpu, settings);
}