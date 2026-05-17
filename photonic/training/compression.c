// compression.c — Model compression and quantization engine implementation

#include "compression.h"
#include "complex.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void compress_prune_phases_dac_aware(double *phases, int count, int dac_bits, double v_pi) {
    if (!phases || count <= 0 || dac_bits <= 0) return;

    // Minimum LSB voltage step
    double levels = (double)((1 << dac_bits) - 1);
    double v_lsb = v_pi / levels;

    // Corresponding minimum physical phase LSB threshold
    // V = V_pi * sqrt(theta / Pi) => theta = Pi * (V / V_pi)^2
    double theta_min_step = M_PI * (v_lsb / v_pi) * (v_lsb / v_pi);

    for (int i = 0; i < count; i++) {
        double angle = phases[i];
        
        // Wrap/clamp to standard [0, 2*Pi] range
        while (angle < 0.0) angle += 2.0 * M_PI;
        while (angle >= 2.0 * M_PI) angle -= 2.0 * M_PI;

        // Check if the phase is near zero OR near 2*Pi
        double dist_to_zero = angle;
        double dist_to_2pi = 2.0 * M_PI - angle;

        if (dist_to_zero < theta_min_step) {
            phases[i] = 0.0;
        } else if (dist_to_2pi < theta_min_step) {
            phases[i] = 2.0 * M_PI;
        } else {
            phases[i] = angle;
        }
    }
}

double compress_quantize_voltage(double voltage, int dac_bits, double max_voltage) {
    if (dac_bits <= 0) return voltage;

    double levels = (double)((1 << dac_bits) - 1);
    double step = max_voltage / levels;

    double quantized = round(voltage / step) * step;

    if (quantized > max_voltage) quantized = max_voltage;
    if (quantized < 0.0) quantized = 0.0;

    return quantized;
}

int clements_reconstruct(
    int N,
    const double *thetas,
    const double *phis,
    const Complex *diagonal,
    Matrix *out_W
) {
    if (N <= 1 || !thetas || !phis || !diagonal || !out_W) return -1;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            out_W->data[i * N + j] = (i == j) ? diagonal[i] : complex_new(0.0, 0.0);
        }
    }

    int n_mzis = N * (N - 1) / 2;

    // Precompute the forward Givens rotation port pairs (p, q) to guarantee 100% exact reverse ordering!
    int *p_arr = (int *)malloc(n_mzis * sizeof(int));
    int *q_arr = (int *)malloc(n_mzis * sizeof(int));
    if (!p_arr || !q_arr) {
        free(p_arr);
        free(q_arr);
        return -1;
    }

    int idx = 0;
    for (int col = 0; col < N - 1; col++) {
        for (int row = N - 1; row > col; row--) {
            p_arr[idx] = row - 1;
            q_arr[idx] = row;
            idx++;
        }
    }

    // Traverse the precomputed rotation list in exact reverse order
    for (int mzi_idx = n_mzis - 1; mzi_idx >= 0; mzi_idx--) {
        int p = p_arr[mzi_idx];
        int q = q_arr[mzi_idx];

        double theta = thetas[mzi_idx];
        double phi = phis[mzi_idx];

        Complex c = complex_new(cos(theta), 0.0);
        Complex s = complex_new(sin(theta) * cos(phi), sin(theta) * sin(phi));
        Complex s_conj = complex_conj(s);

        // Apply R^H multiplication in reverse order
        for (int k = 0; k < N; k++) {
            Complex up_k = out_W->data[p * N + k];
            Complex uq_k = out_W->data[q * N + k];

            out_W->data[p * N + k] = complex_sub(complex_mul(c, up_k), complex_mul(s, uq_k));
            out_W->data[q * N + k] = complex_add(complex_mul(s_conj, up_k), complex_mul(c, uq_k));
        }
    }

    free(p_arr);
    free(q_arr);
    return 0;
}

double matrix_unitarity_error(const Matrix *W) {
    if (!W || W->rows != W->cols || W->rows <= 0) return 0.0;

    int N = W->rows;
    double sum_sq = 0.0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Complex val = complex_new(0.0, 0.0);

            // Compute element (i, j) of W^H * W
            for (int k = 0; k < N; k++) {
                // W^H_ik = conj(W_ki)
                Complex w_ki = W->data[k * N + i];
                Complex w_kj = W->data[k * N + j];
                val = complex_add(val, complex_mul(complex_conj(w_ki), w_kj));
            }

            // Subtract Identity matrix elements
            if (i == j) {
                val.real -= 1.0;
            }

            sum_sq += val.real * val.real + val.imag * val.imag;
        }
    }

    return sqrt(sum_sq);
}
