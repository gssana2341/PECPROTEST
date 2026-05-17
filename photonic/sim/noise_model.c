// noise_model.c — realistic hardware noise simulation
//
// จำลอง noise ที่เกิดจาก photonic hardware จริง เช่น
// - Phase noise (thermal fluctuation)
// - Shot noise (photon counting)
// - Crosstalk between waveguides

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../core/complex.h"

// ─── Noise Configuration ────────────────────────────────────────────

typedef struct {
    double phase_noise_std;    // standard deviation of phase noise (radians)
    double shot_noise_level;   // intensity-dependent noise factor
    double crosstalk_coeff;    // crosstalk coupling coefficient
    unsigned int seed;         // RNG seed for reproducibility
} NoiseConfig;

// Default noise configuration (realistic values)
NoiseConfig noise_default_config(void) {
    NoiseConfig cfg = {
        .phase_noise_std  = 0.01,   // ~0.6 degrees
        .shot_noise_level = 0.001,
        .crosstalk_coeff  = 0.005,
        .seed             = 42
    };
    return cfg;
}

// Apply noise to a complex signal vector
// TODO: implement after core math is complete
int noise_apply(const NoiseConfig *cfg, Complex *signal, size_t length) {
    (void)cfg;
    (void)signal;
    (void)length;
    // Placeholder
    return 0;
}
