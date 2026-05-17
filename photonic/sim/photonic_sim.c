// photonic_sim.c — main photonic simulator engine
//
// Simulates photonic layer forward pass using unitary matrix transformations
// and optical interference patterns.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../core/complex.h"
#include "../core/matrix.h"
#include "../core/photonic.h"
#include "../core/activation.h"
#include "../core/memory.h"

// ─── Photonic Simulator State ───────────────────────────────────────

typedef struct {
    int             num_layers;
    int             layer_dim;      // dimension of each photonic layer
    PhotonicLayer  *layers;         // array of layers
    double          total_energy;   // accumulated energy consumption
    long long       total_ops;      // total operation count
    Complex       **layer_outputs;  // intermediate outputs for training/debug
    double          shot_noise;     // detector shot noise std dev
} SimState;

// Initialize simulator state
SimState sim_init(int num_layers, int layer_dim) {
    SimState s;
    s.num_layers   = num_layers;
    s.layer_dim    = layer_dim;
    s.total_energy = 0.0;
    s.total_ops    = 0;
    s.layers       = NULL;
    s.layer_outputs= NULL;
    s.shot_noise   = 0.0; // Default: no shot noise
    
    if (num_layers > 0 && layer_dim > 0) {
        s.layers = (PhotonicLayer *)pho_alloc(num_layers * sizeof(PhotonicLayer), "SimState.layers");
        s.layer_outputs = (Complex **)pho_alloc(num_layers * sizeof(Complex *), "sim.layer_outputs");
        
        if (s.layers && s.layer_outputs) {
            for (int i = 0; i < num_layers; i++) {
                s.layers[i] = photonic_layer_init(layer_dim);
                s.layer_outputs[i] = (Complex *)pho_alloc(layer_dim * sizeof(Complex), "sim.layer_out_i");
            }
        } else {
            s.num_layers = 0;
        }
    }
    
    return s;
}

// Free simulator resources
void sim_free(SimState *state) {
    if (state) {
        if (state->layers) {
            for (int i = 0; i < state->num_layers; i++) {
                photonic_layer_free(&state->layers[i]);
            }
            pho_free(state->layers, "SimState.layers");
            state->layers = NULL;
        }
        if (state->layer_outputs) {
            for (int i = 0; i < state->num_layers; i++) {
                if (state->layer_outputs[i]) {
                    pho_free(state->layer_outputs[i], "sim.layer_out_i");
                }
            }
            pho_free(state->layer_outputs, "sim.layer_outputs");
            state->layer_outputs = NULL;
        }
    }
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double rand_normal_sim(double mean, double std) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    if (u1 < 1e-15) u1 = 1e-15;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + z * std;
}

// Run forward pass through all layers
int sim_forward(SimState *state, const Complex *input, Complex *output) {
    if (!state || !state->layers || !input || !output) return -1;
    
    int dim = state->layer_dim;
    
    // We need ping-pong buffers for intermediate states between layers
    Complex *buf1 = (Complex *)pho_alloc(dim * sizeof(Complex), "sim.buf1");
    if (!buf1) return -1;
    
    Complex *buf2 = (Complex *)pho_alloc(dim * sizeof(Complex), "sim.buf2");
    if (!buf2) {
        pho_free(buf1, "sim.buf1");
        return -1;
    }
    
    // Copy initial input to buf1
    for (int i = 0; i < dim; i++) {
        buf1[i] = input[i];
    }
    
    // Forward through all layers sequentially
    for (int l = 0; l < state->num_layers; l++) {
        Complex *in_buf  = (l % 2 == 0) ? buf1 : buf2;
        Complex *out_buf = (l % 2 == 0) ? buf2 : buf1;
        
        photonic_layer_forward(&state->layers[l], in_buf, out_buf);
        
        // Save intermediate states
        if (state->layer_outputs && state->layer_outputs[l]) {
            for (int i = 0; i < dim; i++) {
                state->layer_outputs[l][i] = out_buf[i];
            }
        }
        
        // Naive energy/ops tracking per layer
        // A matrix-vector multiplication involves dim^2 complex MAC operations
        state->total_ops += (long long)dim * dim;
        // Assuming ~0.01 pJ per interference MAC on chip
        state->total_energy += (dim * dim) * 0.01; 
    }
    // Determine which buffer holds the final result and apply photodetector noise
    Complex *final_buf = (state->num_layers % 2 == 0) ? buf1 : buf2;
    for (int i = 0; i < dim; i++) {
        Complex z = final_buf[i];
        if (state->shot_noise > 0.0) {
            double intensity = complex_norm_sq(z);
            double shot_std = state->shot_noise * sqrt(intensity);
            double thermal_std = state->shot_noise * 0.5; // background thermal noise
            
            double xi = rand_normal_sim(0.0, shot_std);
            double zeta = rand_normal_sim(0.0, thermal_std);
            
            double noisy_intensity = intensity + xi + zeta;
            if (noisy_intensity < 0.0) noisy_intensity = 0.0;
            
            if (intensity > 1e-15) {
                double scale = sqrt(noisy_intensity / intensity);
                z = complex_new(z.real * scale, z.imag * scale);
            } else {
                z = complex_new(sqrt(noisy_intensity), 0.0);
            }
        }
        output[i] = z;
    }
    
    pho_free(buf1, "sim.buf1");
    pho_free(buf2, "sim.buf2");
    
    return 0;
}

// Reset energy/ops counters
void sim_reset_counters(SimState *state) {
    if (!state) return;
    state->total_energy = 0.0;
    state->total_ops    = 0;
}
