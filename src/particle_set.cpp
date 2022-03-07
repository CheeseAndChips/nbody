#include "particle_set.h"
#include <math.h>

scalar_t distance_sqr(const vec2d_t& a, const vec2d_t& b) {
    scalar_t A = a.x - b.x;
    scalar_t B = a.y - b.y;
    return A*A + B*B;
}

void particle_set_t::set_particle_pos(const vec2d_t& pos, int32_t i) {
    this->positions[i] = pos;
}

void particle_set_t::simulation_timestep(scalar_t deltaT, scalar_t gravitationalConstant, scalar_t distanceAdded) {
    for(int32_t i = 0; i < n; i++) {
        for(int32_t j = i + 1; j < n; j++) {
            scalar_t distance = sqrt(distance_sqr(positions[i], positions[j])) + distanceAdded;
        scalar_t force = gravitationalConstant * mass[i] * mass[j] / (distance*distance);

            vec2d_t direction(
                positions[j].x - positions[i].x,
                positions[j].y - positions[i].y
            );

            scalar_t magnitude = sqrt(direction.x*direction.x + direction.y*direction.y);

            direction.x /= magnitude;
            direction.y /= magnitude;

            velocities[i].x += direction.x * force / mass[i] * deltaT;
            velocities[i].y += direction.y * force / mass[i] * deltaT;
            velocities[j].x -= direction.x * force / mass[j] * deltaT;
            velocities[j].y -= direction.y * force / mass[j] * deltaT;
        }
    }

    for(int32_t i = 0; i < n; i++) {
        positions[i].x += velocities[i].x * deltaT;
        positions[i].y += velocities[i].y * deltaT;
    }
}