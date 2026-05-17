// photonic_sim.h — simulator engine declarations
#ifndef PHOTONIC_SIM_H
#define PHOTONIC_SIM_H

#include <stddef.h>
#include "../core/complex.h"
#include "../core/photonic.h"

// Photonic Simulator State
typedef struct {
    int             num_layers;
    int             layer_dim;
    PhotonicLayer  *layers;
    double          total_energy;
    long long       total_ops;
    Complex       **layer_outputs;
    double          shot_noise;
} SimState;

SimState sim_init(int num_layers, int layer_dim);
void sim_free(SimState *state);
int sim_forward(SimState *state, const Complex *input, Complex *output);
void sim_reset_counters(SimState *state);

#endif // PHOTONIC_SIM_H
