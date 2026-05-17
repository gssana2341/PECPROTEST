// compression.h — Model compression, DAC quantization, and unitarity checking

#ifndef PHOTONIC_COMPRESSION_H
#define PHOTONIC_COMPRESSION_H

#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Prunes waveguide phase shifters based on DAC resolution constraints (Strategy B - Soft Threshold).
 * If a phase angle is below the minimum LSB voltage step phase threshold, it is driven to exactly 0.0 rad.
 */
void compress_prune_phases_dac_aware(double *phases, int count, int dac_bits, double v_pi);

/**
 * Simulates a finite B-bit digital-to-analog converter (DAC) resolution on micro-heater control voltage.
 */
double compress_quantize_voltage(double voltage, int dac_bits, double max_voltage);

/**
 * Reconstructs a unitary matrix W from Clements MZI phase shifts (thetas, phis) and diagonal remaining phases.
 */
int clements_reconstruct(
    int N,
    const double *thetas,
    const double *phis,
    const Complex *diagonal,
    Matrix *out_W
);

/**
 * Computes the unitarity error of a matrix W: ||W^H * W - I||_F (Frobenius norm).
 * Used to verify Stiefel/unitarity constraints are maintained after pruning.
 */
double matrix_unitarity_error(const Matrix *W);

#ifdef __cplusplus
}
#endif

#endif // PHOTONIC_COMPRESSION_H
