#include "particle_wrapper_cpu.h"
#include "file_util.h"
#include <vector>
#include <thread>
#include "simulation_util.h"
#include <math.h>

void particle_wrapper_cpu::do_timestep(simulation_settings_t& settings)
{
    if(threadcnt == 1){
        simulation_cpu(pset.n, 0, 1, pset.positions, pset.velocities, pset.mass, settings);
        posupdate_cpu(pset.n, 0, 1, pset.positions, pset.velocities, settings);
    }else{
        std::vector<std::thread> threads;
        for(int i = 0; i < threadcnt; i++)
        {
            threads.push_back(std::thread(simulation_cpu, pset.n, i, threadcnt, pset.positions, pset.velocities, pset.mass, std::ref(settings)));
        }

        for(std::thread& t : threads) t.join();
        threads.clear();

        for(int i = 0; i < threadcnt; i++)
        {
            threads.push_back(std::thread(posupdate_cpu, pset.n, i, threadcnt, pset.positions, pset.velocities, std::ref(settings)));
        }

        for(std::thread& t : threads) t.join();
        threads.clear();
    }
}