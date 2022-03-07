#include <vector>
#include <fstream>
#include "particle_wrapper.h"
#include "encoder_util.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>

scalar_t randomflt()
{
    return (scalar_t)rand() / RAND_MAX;
}

int main()
{
    const int video_width = 1920;
    const int video_height = 1080;

    const int N = 20000;
    particle_wrapper pset(N);
    int fps = 60;
    int time = 30;

    video_encoder enc("/media/RAMDISK/output.264", video_width, video_height, fps, codec_settings_t("libx264", "slow", 0));

    std::vector<std::vector<uint8_t>> data(video_width, std::vector<uint8_t>(video_height));

    scalar_t coord_visible = 3.0f; // [-coord_visible; coord_visible] will be visible smaller axis
    scalar_t scalevalue = (std::min(video_height, video_width) / 2) / coord_visible;

    for(int i = 0; i < N; i++)
    {
        scalar_t x = randomflt();
        scalar_t y = randomflt();

        // scale [0, 1] to [-1, 1]
        x = 2*x - 1;
        y = 2*y - 1;

        scalar_t mass = randomflt();
        mass = exp(2*mass - 1) - exp(-1);

        pset.set_particle_values(i, vec2d_t(x, y), vec2d_t(0, 0), mass);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for(int T = 0; T < fps*time; T++)
    {
        std::cout << "Starting timestep " << T << std::endl;
        pset.do_simulation_timestep(1.0/fps, 1e-4 / 2, 1e-3);

        std::cout << "Creating image" << std::endl;
        for(auto& row : data){
            for(auto& cell : row) cell = 0;
        }

        for(int i = 0; i < N; i++){
            vec2d_t pos = pset.get_particle_position(i);
            int x = pos.x * scalevalue + video_width / 2;
            int y = pos.y * scalevalue + video_height / 2;

            if(x >= 0 && y >= 0 && x < video_width && y < video_height){
                data.at(x).at(y) = 255;
            }
        }

        std::cout << "Updating pixel data" << std::endl;
        enc.update_pixels(data);
        std::cout << "Writing out frame data" << std::endl;
        enc.write_frame();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time taken: " << diff.count() << std::endl;
}