#include "../core/photonic.h"
#include "../core/activation.h"
#include "../core/loss.h"
#include "../core/memory.h"
#include "../sim/photonic_sim.h"
#include "../training/gradient_analytic.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Helper to compute matrix norm of difference
double compute_matrix_diff_norm(const Matrix *a, const Matrix *b) {
    double sum = 0.0;
    size_t dim = a->rows;
    for (size_t i = 0; i < dim * dim; i++) {
        Complex diff = complex_sub(a->data[i], b->data[i]);
        sum += complex_norm_sq(diff);
    }
    return sqrt(sum);
}

// XOR Data Setup (4 samples)
static const double xor_inputs[4][4] = {
    {0.0, 0.0, 0.0, 1.0}, // [0,0]
    {0.0, 1.0, 0.0, 0.0}, // [0,1]
    {1.0, 0.0, 0.0, 0.0}, // [1,0]
    {0.7071, 0.7071, 0.0, 0.0} // [1,1] normalized
};
static const double xor_targets[4][4] = {
    {1.0, 0.0, 0.0, 0.0}, // Class 0 (0)
    {0.0, 1.0, 0.0, 0.0}, // Class 1 (1)
    {0.0, 1.0, 0.0, 0.0}, // Class 1 (1)
    {1.0, 0.0, 0.0, 0.0}  // Class 0 (0)
};

// Batch loss calculator (Finite Difference helper)
double compute_xor_batch_loss(SimState *sim, double h_param) {
    (void)h_param;
    double total_loss = 0.0;
    int dim = sim->layer_dim;
    Complex *net_input = (Complex*)malloc(dim * sizeof(Complex));
    Complex *net_output = (Complex*)malloc(dim * sizeof(Complex));
    double *probs = (double*)malloc(dim * sizeof(double));

    for (int i = 0; i < 4; i++) {
        for (int k = 0; k < dim; k++) {
            net_input[k] = complex_new(xor_inputs[i][k], 0.0);
        }
        sim_forward(sim, net_input, net_output);
        double loss = loss_cross_entropy_softmax_optical(
            net_output, 
            sim->layers[1].detector_gain, 
            xor_targets[i], 
            4, 
            probs
        );
        total_loss += loss;
    }

    free(net_input);
    free(net_output);
    free(probs);
    return total_loss / 4.0;
}

// Numerical gradient calculation for a layer on XOR dataset
Matrix compute_xor_numerical_gradient(SimState *sim, int layer_idx, double h) {
    PhotonicLayer *layer = &sim->layers[layer_idx];
    size_t dim = layer->weights.rows;
    Matrix grad = matrix_new(dim, dim);

    for (size_t i = 0; i < dim * dim; i++) {
        Complex orig_w = layer->weights.data[i];

        // --- Real ---
        layer->weights.data[i] = complex_new(orig_w.real + h, orig_w.imag);
        double loss_plus_real = compute_xor_batch_loss(sim, h);

        layer->weights.data[i] = complex_new(orig_w.real - h, orig_w.imag);
        double loss_minus_real = compute_xor_batch_loss(sim, h);

        double dL_dreal = (loss_plus_real - loss_minus_real) / (2.0 * h);

        // --- Imag ---
        layer->weights.data[i] = complex_new(orig_w.real, orig_w.imag + h);
        double loss_plus_imag = compute_xor_batch_loss(sim, h);

        layer->weights.data[i] = complex_new(orig_w.real, orig_w.imag - h);
        double loss_minus_imag = compute_xor_batch_loss(sim, h);

        double dL_dimag = (loss_plus_imag - loss_minus_imag) / (2.0 * h);

        // Restore
        layer->weights.data[i] = orig_w;
        grad.data[i] = complex_new(dL_dreal, dL_dimag);
    }
    return grad;
}

int main() {
    printf("=== Photonic Analytic Gradient Verification ===\n\n");
    srand(1337);

    int num_layers = 2;
    int dim = 4;
    double h = 1e-5;

    // 1. Initialize Network
    SimState sim = sim_init(num_layers, dim);
    
    // Perturb initial weights slightly to break symmetry and make matrices random unitary
    for (int l = 0; l < num_layers; l++) {
        sim.layers[l].kerr_gamma = 0.5;      // Increase non-linearity to gamma=0.5
        sim.layers[l].detector_gain = 15.0;  // Increase TIA detector gain to 15.0
        
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                double noise_r = ((double)rand() / RAND_MAX - 0.5) * 0.1;
                double noise_i = ((double)rand() / RAND_MAX - 0.5) * 0.1;
                if (i == j) {
                    sim.layers[l].weights.data[i * dim + j] = complex_new(1.0 + noise_r, noise_i);
                } else {
                    sim.layers[l].weights.data[i * dim + j] = complex_new(noise_r, noise_i);
                }
            }
        }
    }

    // ────────────────────────────────────────────────────────────────
    // TEST 1: Analytic vs Finite Difference Gradient
    // ────────────────────────────────────────────────────────────────
    printf("[Test 1] Calculating gradients for XOR sample 0...\n");
    
    // Get sample 0
    Complex net_input[4];
    for (int k = 0; k < dim; k++) {
        net_input[k] = complex_new(xor_inputs[0][k], 0.0);
    }

    // Forward pass
    Complex intermediate[4];
    // Layer 0 forward: in -> intermediate
    photonic_layer_forward(&sim.layers[0], net_input, intermediate);
    // Layer 1 forward: intermediate -> final
    Complex final_output[4];
    photonic_layer_forward(&sim.layers[1], intermediate, final_output);

    // Compute Loss Gradient: dL/dfinal
    Complex loss_grad[4];
    loss_cross_entropy_softmax_optical_grad(
        final_output,
        sim.layers[1].detector_gain,
        xor_targets[0],
        4,
        loss_grad
    );

    // Backprop Layer 1: output_grad = loss_grad
    Matrix grad_l1_analytic = matrix_new(dim, dim);
    Complex input_grad_l1[4];
    photonic_layer_backward(
        &sim.layers[1].weights,
        intermediate,
        loss_grad,
        sim.layers[1].kerr_gamma,
        &grad_l1_analytic,
        input_grad_l1,
        dim
    );

    // Backprop Layer 0: output_grad = input_grad_l1
    Matrix grad_l0_analytic = matrix_new(dim, dim);
    Complex input_grad_l0[4];
    photonic_layer_backward(
        &sim.layers[0].weights,
        net_input,
        input_grad_l1,
        sim.layers[0].kerr_gamma,
        &grad_l0_analytic,
        input_grad_l0,
        dim
    );

    // Compute numerical gradients for comparison on batch size = 1 (just sample 0)
    // To do this exactly, we temporarily compute numerical gradients on sample 0
    // Modify compute_xor_batch_loss temporarily inside a local numerical grad
    Matrix grad_l0_numerical = matrix_new(dim, dim);
    Matrix grad_l1_numerical = matrix_new(dim, dim);
    
    for (size_t i = 0; i < (size_t)dim * dim; i++) {
        Complex orig_w = sim.layers[0].weights.data[i];
        
        // +h
        sim.layers[0].weights.data[i] = complex_new(orig_w.real + h, orig_w.imag);
        Complex out_temp[4];
        photonic_layer_forward(&sim.layers[0], net_input, intermediate);
        photonic_layer_forward(&sim.layers[1], intermediate, out_temp);
        double loss_p = loss_cross_entropy_softmax_optical(out_temp, sim.layers[1].detector_gain, xor_targets[0], 4, NULL);

        // -h
        sim.layers[0].weights.data[i] = complex_new(orig_w.real - h, orig_w.imag);
        photonic_layer_forward(&sim.layers[0], net_input, intermediate);
        photonic_layer_forward(&sim.layers[1], intermediate, out_temp);
        double loss_m = loss_cross_entropy_softmax_optical(out_temp, sim.layers[1].detector_gain, xor_targets[0], 4, NULL);

        sim.layers[0].weights.data[i] = orig_w; // restore
        double dL_dr = (loss_p - loss_m) / (2.0 * h);

        // Imag
        sim.layers[0].weights.data[i] = complex_new(orig_w.real, orig_w.imag + h);
        photonic_layer_forward(&sim.layers[0], net_input, intermediate);
        photonic_layer_forward(&sim.layers[1], intermediate, out_temp);
        loss_p = loss_cross_entropy_softmax_optical(out_temp, sim.layers[1].detector_gain, xor_targets[0], 4, NULL);

        sim.layers[0].weights.data[i] = complex_new(orig_w.real, orig_w.imag - h);
        photonic_layer_forward(&sim.layers[0], net_input, intermediate);
        photonic_layer_forward(&sim.layers[1], intermediate, out_temp);
        loss_m = loss_cross_entropy_softmax_optical(out_temp, sim.layers[1].detector_gain, xor_targets[0], 4, NULL);

        sim.layers[0].weights.data[i] = orig_w; // restore
        double dL_di = (loss_p - loss_m) / (2.0 * h);

        grad_l0_numerical.data[i] = complex_new(dL_dr, dL_di);
    }
    
    // Do the same numerical gradient for Layer 1 on sample 0
    for (size_t i = 0; i < (size_t)dim * dim; i++) {
        Complex orig_w = sim.layers[1].weights.data[i];
        
        // +h
        sim.layers[1].weights.data[i] = complex_new(orig_w.real + h, orig_w.imag);
        Complex out_temp[4];
        photonic_layer_forward(&sim.layers[0], net_input, intermediate);
        photonic_layer_forward(&sim.layers[1], intermediate, out_temp);
        double loss_p = loss_cross_entropy_softmax_optical(out_temp, sim.layers[1].detector_gain, xor_targets[0], 4, NULL);

        // -h
        sim.layers[1].weights.data[i] = complex_new(orig_w.real - h, orig_w.imag);
        photonic_layer_forward(&sim.layers[0], net_input, intermediate);
        photonic_layer_forward(&sim.layers[1], intermediate, out_temp);
        double loss_m = loss_cross_entropy_softmax_optical(out_temp, sim.layers[1].detector_gain, xor_targets[0], 4, NULL);

        sim.layers[1].weights.data[i] = orig_w; // restore
        double dL_dr = (loss_p - loss_m) / (2.0 * h);

        // Imag
        sim.layers[1].weights.data[i] = complex_new(orig_w.real, orig_w.imag + h);
        photonic_layer_forward(&sim.layers[0], net_input, intermediate);
        photonic_layer_forward(&sim.layers[1], intermediate, out_temp);
        loss_p = loss_cross_entropy_softmax_optical(out_temp, sim.layers[1].detector_gain, xor_targets[0], 4, NULL);

        sim.layers[1].weights.data[i] = complex_new(orig_w.real, orig_w.imag - h);
        photonic_layer_forward(&sim.layers[0], net_input, intermediate);
        photonic_layer_forward(&sim.layers[1], intermediate, out_temp);
        loss_m = loss_cross_entropy_softmax_optical(out_temp, sim.layers[1].detector_gain, xor_targets[0], 4, NULL);

        sim.layers[1].weights.data[i] = orig_w; // restore
        double dL_di = (loss_p - loss_m) / (2.0 * h);

        grad_l1_numerical.data[i] = complex_new(dL_dr, dL_di);
    }

    double diff_l0 = compute_matrix_diff_norm(&grad_l0_analytic, &grad_l0_numerical);
    double diff_l1 = compute_matrix_diff_norm(&grad_l1_analytic, &grad_l1_numerical);

    printf("  Layer 0 Analytic vs Finite Difference Diff Norm: %g\n", diff_l0);
    printf("  Layer 1 Analytic vs Finite Difference Diff Norm: %g\n", diff_l1);

    if (diff_l0 < 1e-4 && diff_l1 < 1e-4) {
        printf("  [PASSED] Test 1: Analytic vs Finite Difference match within 1e-4!\n\n");
    } else {
        printf("  [FAILED] Test 1: Gradient difference is too large!\n\n");
    }

    matrix_free(&grad_l0_numerical);
    matrix_free(&grad_l1_numerical);

    // ────────────────────────────────────────────────────────────────
    // TEST 2: Unitarity Preserved after update
    // ────────────────────────────────────────────────────────────────
    printf("[Test 2] Checking Cayley update unitarity...\n");
    
    // Project G onto tangent space
    Matrix G_skew_l0 = matrix_new(dim, dim);
    Matrix G_skew_l1 = matrix_new(dim, dim);
    
    compute_skew_hermitian(&grad_l0_analytic, &sim.layers[0].weights, &G_skew_l0);
    compute_skew_hermitian(&grad_l1_analytic, &sim.layers[1].weights, &G_skew_l1);
    
    // Re-orthogonalize current layers to ensure they start as unitary
    // Using simple identity for starting weights to prove unitarity conservation
    Matrix starting_W0 = matrix_identity(dim);
    Matrix starting_W1 = matrix_identity(dim);
    
    cayley_update(&starting_W0, &G_skew_l0, 1.0);
    cayley_update(&starting_W1, &G_skew_l1, 1.0);

    int unitary0 = matrix_is_unitary(&starting_W0, 1e-6);
    int unitary1 = matrix_is_unitary(&starting_W1, 1e-6);

    printf("  Layer 0 is Unitary post-update: %s\n", unitary0 ? "YES" : "NO");
    printf("  Layer 1 is Unitary post-update: %s\n", unitary1 ? "YES" : "NO");

    if (unitary0 && unitary1) {
        printf("  [PASSED] Test 2: Unitarity preserved successfully!\n\n");
    } else {
        printf("  [FAILED] Test 2: Matrix lost unitarity!\n\n");
    }

    matrix_free(&grad_l0_analytic);
    matrix_free(&grad_l1_analytic);
    matrix_free(&G_skew_l0);
    matrix_free(&G_skew_l1);
    matrix_free(&starting_W0);
    matrix_free(&starting_W1);

    // ────────────────────────────────────────────────────────────────
    // TEST 3: Training Convergence on XOR
    // ────────────────────────────────────────────────────────────────
    printf("[Test 3] Training XOR using Analytic gradients for 1000 epochs...\n");
    
    // Re-perturb weights to non-identity for learning
    for (int l = 0; l < num_layers; l++) {
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                double noise_r = ((double)rand() / RAND_MAX - 0.5) * 0.05;
                double noise_i = ((double)rand() / RAND_MAX - 0.5) * 0.05;
                if (i == j) {
                    sim.layers[l].weights.data[i * dim + j] = complex_new(1.0 + noise_r, noise_i);
                } else {
                    sim.layers[l].weights.data[i * dim + j] = complex_new(noise_r, noise_i);
                }
            }
        }
    }

    double lr = 0.05; // Tuned learning rate
    double initial_loss = compute_xor_batch_loss(&sim, 0.0);
    printf("  Initial Loss: %.6f\n", initial_loss);

    for (int epoch = 1; epoch <= 1000; epoch++) {
        // Accumulate gradients over the 4 batch samples
        Matrix grad_l0_total = matrix_new(dim, dim);
        Matrix grad_l1_total = matrix_new(dim, dim);
        
        for (int s_idx = 0; s_idx < 4; s_idx++) {
            Complex net_in[4];
            for (int k = 0; k < dim; k++) {
                net_in[k] = complex_new(xor_inputs[s_idx][k], 0.0);
            }

            Complex inter[4];
            photonic_layer_forward(&sim.layers[0], net_in, inter);
            Complex final_out[4];
            photonic_layer_forward(&sim.layers[1], inter, final_out);

            Complex l_grad[4];
            loss_cross_entropy_softmax_optical_grad(
                final_out,
                sim.layers[1].detector_gain,
                xor_targets[s_idx],
                4,
                l_grad
            );

            Matrix g_l1 = matrix_new(dim, dim);
            Complex in_g_l1[4];
            photonic_layer_backward(
                &sim.layers[1].weights,
                inter,
                l_grad,
                sim.layers[1].kerr_gamma,
                &g_l1,
                in_g_l1,
                dim
            );

            Matrix g_l0 = matrix_new(dim, dim);
            Complex in_g_l0[4];
            photonic_layer_backward(
                &sim.layers[0].weights,
                net_in,
                in_g_l1,
                sim.layers[0].kerr_gamma,
                &g_l0,
                in_g_l0,
                dim
            );

            // Add to total
            for (int idx = 0; idx < dim * dim; idx++) {
                grad_l0_total.data[idx] = complex_add(grad_l0_total.data[idx], g_l0.data[idx]);
                grad_l1_total.data[idx] = complex_add(grad_l1_total.data[idx], g_l1.data[idx]);
            }

            matrix_free(&g_l0);
            matrix_free(&g_l1);
        }

        // Average gradients
        for (int idx = 0; idx < dim * dim; idx++) {
            grad_l0_total.data[idx] = complex_new(grad_l0_total.data[idx].real / 4.0, grad_l0_total.data[idx].imag / 4.0);
            grad_l1_total.data[idx] = complex_new(grad_l1_total.data[idx].real / 4.0, grad_l1_total.data[idx].imag / 4.0);
        }

        // Tangent projections
        Matrix G_skew_l0 = matrix_new(dim, dim);
        Matrix G_skew_l1 = matrix_new(dim, dim);
        compute_skew_hermitian(&grad_l0_total, &sim.layers[0].weights, &G_skew_l0);
        compute_skew_hermitian(&grad_l1_total, &sim.layers[1].weights, &G_skew_l1);

        // Update
        cayley_update(&sim.layers[0].weights, &G_skew_l0, lr);
        cayley_update(&sim.layers[1].weights, &G_skew_l1, lr);

        matrix_free(&grad_l0_total);
        matrix_free(&grad_l1_total);
        matrix_free(&G_skew_l0);
        matrix_free(&G_skew_l1);

        if (epoch % 100 == 0) {
            double loss = compute_xor_batch_loss(&sim, 0.0);
            printf("  Epoch %4d | Loss: %.6f\n", epoch, loss);
        }
    }

    double final_loss = compute_xor_batch_loss(&sim, 0.0);
    printf("  Final Loss: %.6f\n", final_loss);

    if (final_loss < initial_loss && final_loss < 0.1) {
        printf("  [PASSED] Test 3: XOR successfully trained using Analytic Cayley Gradients!\n\n");
    } else {
        printf("  [FAILED] Test 3: XOR training did not converge below 0.1!\n\n");
    }

    sim_free(&sim);
    return 0;
}
