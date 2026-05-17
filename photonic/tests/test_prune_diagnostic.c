// test_prune_diagnostic.c — Clements MZI phase baseline diagnostics for pruning potential

#include "pholang.h"
#include "photonic_sim.h"
#include "matrix.h"
#include "complex.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Private definition of PhoNetwork struct for direct state inspection
struct PhoNetwork {
    SimState sim;
    double   learning_rate;
    int      epochs;
    int      batch_size;
    int      early_stop;
    double   phase_noise;
    double   shot_noise;
};

// Decomposes weight matrix into raw Clements MZI phase angles
int extract_clements_phases(const Matrix *W, double *out_thetas, double *out_phis) {
    int N = W->rows;
    if (W->cols != N || N <= 1) return -1;

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

            // Map phases to symmetric [-Pi, Pi] or wrap them to [0, Pi] to check zero proximity
            // Standard MZI phase shifts represent physical micro-heater settings
            out_thetas[mzi_idx] = theta;
            out_phis[mzi_idx] = phi;
            mzi_idx++;
        }
    }

    matrix_free(&U);
    return mzi_idx;
}

void print_layer_diagnostics(int layer_idx, const double *thetas, const double *phis, int count) {
    double thresholds[] = {0.01, 0.05, 0.10, 0.20, 0.50};
    int num_thresholds = sizeof(thresholds) / sizeof(thresholds[0]);

    printf("\n=== Layer %d Clements Phase Pruning Diagnostics (%d MZIs / %d total phase shifters) ===\n", 
           layer_idx, count, count * 2);
    printf("-------------------------------------------------------------------------\n");
    printf("%-15s | %-12s | %-12s | %-12s | %-12s\n", 
           "Threshold (rad)", "Near Zero Th", "Near Zero Ph", "Total Near 0", "Heater Savings");
    printf("-------------------------------------------------------------------------\n");

    for (int t = 0; t < num_thresholds; t++) {
        double eps = thresholds[t];
        int near_zero_th = 0;
        int near_zero_ph = 0;

        for (int i = 0; i < count; i++) {
            // Theta and Phi represent physical phases. If phase is near zero OR near 2*Pi,
            // it means the micro-heater doesn't need to heat the waveguide (voltage -> 0V).
            double diff_th = fmod(thetas[i], 2.0 * M_PI);
            if (diff_th > M_PI) diff_th = 2.0 * M_PI - diff_th;
            if (fabs(diff_th) < eps) {
                near_zero_th++;
            }

            double diff_ph = fmod(phis[i], 2.0 * M_PI);
            if (diff_ph > M_PI) diff_ph = 2.0 * M_PI - diff_ph;
            if (fabs(diff_ph) < eps) {
                near_zero_ph++;
            }
        }

        int total_near = near_zero_th + near_zero_ph;
        double pct = 100.0 * (double)total_near / (double)(count * 2);

        printf("< %-13.2f | %-12d | %-12d | %-12d | %-11.2f%%\n", 
               eps, near_zero_th, near_zero_ph, total_near, pct);
    }
    printf("-------------------------------------------------------------------------\n");
}

int main(void) {
    printf("=== Starting Clements MZI Phase Pruning Diagnostic ===\n");
    printf("Loading PhoLang model...\n");

    const char *pho_path = "photonic/lang/mnist.pho";
    PhoNetwork *net = pho_network_load(pho_path);
    if (!net) {
        fprintf(stderr, "Failed to load network.\n");
        return 1;
    }

    // Step 1: Train the model for 2 quick epochs to converge weights to a realistic manifold
    printf("Training dynamic model for 2 epochs on MNIST to generate realistic trained weights...\n");
    PhoTrainConfig cfg = { .epochs = 2, .lr = 0.01, .batch_size = 16 };
    PhoResult res = pho_network_train(net, "data/mnist_scaled.csv", cfg);
    printf("Training completed. Final accuracy: %.2f%%, Loss: %.6f\n", 
           res.final_accuracy, res.final_loss);

    // Step 2: Extract and decompose weight matrices of both Layer 0 and Layer 1
    int N = net->sim.layer_dim; // 64
    int n_mzis = N * (N - 1) / 2; // 2016

    double *l0_thetas = (double *)calloc(n_mzis, sizeof(double));
    double *l0_phis = (double *)calloc(n_mzis, sizeof(double));
    double *l1_thetas = (double *)calloc(n_mzis, sizeof(double));
    double *l1_phis = (double *)calloc(n_mzis, sizeof(double));

    if (!l0_thetas || !l0_phis || !l1_thetas || !l1_phis) {
        fprintf(stderr, "Out of memory.\n");
        return 1;
    }

    printf("\nDecomposing Layer 0 unitary weight matrix (%d x %d)...\n", N, N);
    extract_clements_phases(&net->sim.layers[0].weights, l0_thetas, l0_phis);

    printf("Decomposing Layer 1 unitary weight matrix (%d x %d)...\n", N, N);
    extract_clements_phases(&net->sim.layers[1].weights, l1_thetas, l1_phis);

    // Step 3: Print baseline diagnostics for both layers
    print_layer_diagnostics(0, l0_thetas, l0_phis, n_mzis);
    print_layer_diagnostics(1, l1_thetas, l1_phis, n_mzis);

    // Clean up
    free(l0_thetas);
    free(l0_phis);
    free(l1_thetas);
    free(l1_phis);
    pho_network_free(net);

    printf("\n🎉 CLEMENTS MZI PRUNING DIAGNOSTIC COMPLETED SUCCESSFULLY! 🎉\n");
    return 0;
}
