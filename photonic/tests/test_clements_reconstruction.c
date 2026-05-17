#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "matrix.h"
#include "complex.h"
#include "compression.h"

// Generate a truly random Haar-distributed unitary matrix via Gram-Schmidt orthogonalization
Matrix generate_random_unitary(int N) {
    Matrix W = matrix_new(N, N);
    // 1. Fill W with random Gaussian complex numbers using Box-Muller
    for (int i = 0; i < N * N; i++) {
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        if (u1 < 1e-15) u1 = 1e-15;
        double r = sqrt(-2.0 * log(u1));
        W.data[i] = complex_new(r * cos(2.0 * M_PI * u2), r * sin(2.0 * M_PI * u2));
    }
    // 2. Gram-Schmidt orthogonalization on columns (so that W^H * W = I)
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < j; k++) {
            Complex proj = complex_new(0.0, 0.0);
            for (int r = 0; r < N; r++) {
                Complex w_rk = W.data[r * N + k];
                Complex w_rj = W.data[r * N + j];
                proj = complex_add(proj, complex_mul(complex_conj(w_rk), w_rj));
            }
            for (int r = 0; r < N; r++) {
                W.data[r * N + j] = complex_sub(W.data[r * N + j], complex_mul(proj, W.data[r * N + k]));
            }
        }
        // Normalize column j
        double norm = 0.0;
        for (int r = 0; r < N; r++) {
            norm += complex_norm_sq(W.data[r * N + j]);
        }
        norm = sqrt(norm);
        for (int r = 0; r < N; r++) {
            if (norm > 1e-12) {
                W.data[r * N + j] = complex_new(W.data[r * N + j].real / norm, W.data[r * N + j].imag / norm);
            } else {
                W.data[r * N + j] = complex_new(1.0, 0.0);
            }
        }
    }
    return W;
}

int main() {
    srand(time(NULL));
    int N = 64; // Test 64x64 matrix to match physical layers
    printf("=== Clements Decomposition & Reconstruction Diagnostic Test ===\n");
    printf("Matrix Dimension N = %d\n", N);

    Matrix W = generate_random_unitary(N);
    int n_mzis = N * (N - 1) / 2;

    double *thetas = (double *)calloc(n_mzis, sizeof(double));
    double *phis = (double *)calloc(n_mzis, sizeof(double));
    Complex *diagonal = (Complex *)calloc(N, sizeof(Complex));

    // 1. Decompose W
    Matrix U = matrix_copy(&W);
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
                phi = angle_p - angle_q;
            }

            if (phi < 0.0) phi += 2.0 * M_PI;
            if (theta < 0.0) theta += 2.0 * M_PI;

            thetas[mzi_idx] = theta;
            phis[mzi_idx] = phi;
            mzi_idx++;

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

    for (int i = 0; i < N; i++) {
        double norm = complex_norm(U.data[i * N + i]);
        if (norm > 1e-12) {
            diagonal[i] = complex_new(U.data[i * N + i].real / norm, U.data[i * N + i].imag / norm);
        } else {
            diagonal[i] = complex_new(1.0, 0.0);
        }
    }
    matrix_free(&U);

    // 2. Reconstruct W_rec
    Matrix W_rec = matrix_new(N, N);
    clements_reconstruct(N, thetas, phis, diagonal, &W_rec);

    // 3. Measure error element-wise
    double diff_sq = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Complex diff = complex_sub(W.data[i * N + j], W_rec.data[i * N + j]);
            diff_sq += complex_norm_sq(diff);
        }
    }
    double frob_error = sqrt(diff_sq);

    printf("Frobenius Reconstruction Error ||W - W_reconstructed||_F: %.15f\n", frob_error);
    if (frob_error < 1e-10) {
        printf("✅ SUCCESS! Clements Reconstruction matches original matrix perfectly!\n");
    } else {
        printf("❌ FAILURE! Reconstruction has high error!\n");
    }

    // Free memory
    matrix_free(&W);
    matrix_free(&W_rec);
    free(thetas);
    free(phis);
    free(diagonal);

    return 0;
}
