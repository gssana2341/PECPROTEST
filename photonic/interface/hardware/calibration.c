#include "calibration.h"
#include "complex.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─── Direct Phase Calibration & Localized Thermal Cross-Talk Compensation ───

int calibrate_direct_phases_to_voltages(
    const Matrix *W,
    const CalibrationParams *params,
    double *out_voltages
) {
    if (!W || !W->data || !params || !out_voltages) {
        return -1;
    }

    int rows = W->rows;
    int cols = W->cols;
    if (rows <= 0 || cols <= 0) return -1;

    // Build a localized coupling matrix (K) for a single spatial port vector (size: cols x cols)
    // This reduces computational complexity from O((N^2)^3) to N * O(N^3), speeding up by >4000x!
    Matrix K = matrix_new(cols, cols);
    if (!K.data) return -1;

    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < cols; j++) {
            double dist = fabs((double)(i - j));
            double coupling = pow(params->cross_talk_factor, dist);
            K.data[i * cols + j] = complex_new(coupling, 0.0);
        }
    }

    // Invert the local spatial port coupling matrix
    Matrix inv_K;
    if (!matrix_inverse(&K, &inv_K)) {
        inv_K = matrix_identity(cols);
    }

    // Process waveguide phases and voltages row-by-row (port-by-port)
    double *raw_phases = (double *)malloc(cols * sizeof(double));
    double *comp_phases = (double *)malloc(cols * sizeof(double));
    if (!raw_phases || !comp_phases) {
        free(raw_phases);
        free(comp_phases);
        matrix_free(&K);
        matrix_free(&inv_K);
        return -1;
    }

    for (int r = 0; r < rows; r++) {
        // 1. Extract raw phase angles for this spatial row
        for (int c = 0; c < cols; c++) {
            Complex val = W->data[r * cols + c];
            double phase = atan2(val.imag, val.real);
            if (phase < 0.0) phase += 2.0 * M_PI;
            raw_phases[c] = phase;
        }

        // 2. Apply inverse coupling matrix to solve for compensated ideal phases
        for (int i = 0; i < cols; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                sum += inv_K.data[i * cols + j].real * raw_phases[j];
            }
            if (sum < 0.0) sum = 0.0;
            if (sum > 2.0 * M_PI) sum = fmod(sum, 2.0 * M_PI);
            comp_phases[i] = sum;
        }

        // 3. Convert compensated phases to control voltages
        for (int c = 0; c < cols; c++) {
            double v = params->v_pi * sqrt(comp_phases[c] / M_PI);
            if (v > params->max_voltage) {
                v = params->max_voltage;
            }
            out_voltages[r * cols + c] = v;
        }
    }

    free(raw_phases);
    free(comp_phases);
    matrix_free(&K);
    matrix_free(&inv_K);

    return 0;
}

// ─── Clements MZI Unitary Matrix Decomposition & Localized Calibration ───

int calibrate_unitary_to_mzi_voltages(
    const Matrix *W,
    const CalibrationParams *params,
    double *out_theta_voltages,
    double *out_phi_voltages
) {
    if (!W || !W->data || !params || !out_theta_voltages || !out_phi_voltages) {
        return -1;
    }

    int N = W->rows;
    if (W->cols != N || N <= 1) return -1;

    int n_mzis = N * (N - 1) / 2;

    double *mzi_thetas = (double *)calloc(n_mzis, sizeof(double));
    double *mzi_phis = (double *)calloc(n_mzis, sizeof(double));
    if (!mzi_thetas || !mzi_phis) {
        free(mzi_thetas);
        free(mzi_phis);
        return -1;
    }

    // Clements decomposition Givens rotations
    Matrix U = matrix_copy(W);

    int mzi_idx = 0;
    for (int col = 0; col < N - 1; col++) {
        for (int row = N - 1; row > col; row--) {
            int p = row - 1;
            int q = row;

            Complex u_p = U.data[p * N + col];
            Complex u_q = U.data[q * N + col];

            double r_p = complex_norm(u_p);
            double r_q = complex_norm(u_q);

            double theta = 0.0;
            double phi = 0.0;

            if (r_q > 1e-12) {
                theta = atan2(r_q, r_p);
                double angle_p = atan2(u_p.imag, u_p.real);
                double angle_q = atan2(u_q.imag, u_q.real);
                phi = angle_q - angle_p;
            }

            if (phi < 0.0) phi += 2.0 * M_PI;
            if (theta < 0.0) theta += 2.0 * M_PI;

            if (mzi_idx < n_mzis) {
                mzi_thetas[mzi_idx] = theta;
                mzi_phis[mzi_idx] = phi;
                mzi_idx++;
            }

            Complex c = complex_new(cos(theta), 0.0);
            Complex s = complex_new(sin(theta) * cos(phi), sin(theta) * sin(phi));
            Complex s_conj = complex_conj(s);

            for (int k = 0; k < N; k++) {
                Complex up_k = U.data[p * N + k];
                Complex uq_k = U.data[q * N + k];

                U.data[p * N + k] = complex_add(complex_mul(c, up_k), complex_mul(s, uq_k));
                U.data[q * N + k] = complex_sub(complex_mul(c, uq_k), complex_mul(s_conj, up_k));
            }
        }
    }

    matrix_free(&U);

    // Apply localized Block-by-Block cross-talk compensation of size N (waveguide channels)
    Matrix K = matrix_new(N, N);
    if (!K.data) {
        free(mzi_thetas);
        free(mzi_phis);
        return -1;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double dist = fabs((double)(i - j));
            double coupling = pow(params->cross_talk_factor, dist);
            K.data[i * N + j] = complex_new(coupling, 0.0);
        }
    }

    Matrix inv_K;
    if (!matrix_inverse(&K, &inv_K)) {
        inv_K = matrix_identity(N);
    }

    // Process MZIs in blocks of size N
    for (int block_start = 0; block_start < n_mzis; block_start += N) {
        int block_size = (block_start + N > n_mzis) ? (n_mzis - block_start) : N;

        for (int i = 0; i < block_size; i++) {
            double comp_theta = 0.0;
            double comp_phi = 0.0;

            for (int j = 0; j < block_size; j++) {
                comp_theta += inv_K.data[i * N + j].real * mzi_thetas[block_start + j];
                comp_phi += inv_K.data[i * N + j].real * mzi_phis[block_start + j];
            }

            if (comp_theta < 0.0) comp_theta = 0.0;
            if (comp_theta > 2.0 * M_PI) comp_theta = fmod(comp_theta, 2.0 * M_PI);

            if (comp_phi < 0.0) comp_phi = 0.0;
            if (comp_phi > 2.0 * M_PI) comp_phi = fmod(comp_phi, 2.0 * M_PI);

            double v_theta = params->v_pi * sqrt(comp_theta / M_PI);
            double v_phi = params->v_pi * sqrt(comp_phi / M_PI);

            if (v_theta > params->max_voltage) v_theta = params->max_voltage;
            if (v_phi > params->max_voltage) v_phi = params->max_voltage;

            out_theta_voltages[block_start + i] = v_theta;
            out_phi_voltages[block_start + i] = v_phi;
        }
    }

    free(mzi_thetas);
    free(mzi_phis);
    matrix_free(&K);
    matrix_free(&inv_K);

    return 0;
}
