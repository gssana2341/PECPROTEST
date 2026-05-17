// XOR Problem — Photonic Neural Network Example
//
// เป้าหมาย: ทดสอบ photonic forward pass ด้วยปัญหา XOR
// ต้องได้ accuracy > 99%
//
// XOR Truth Table:
//   0 XOR 0 = 0
//   0 XOR 1 = 1
//   1 XOR 0 = 1
//   1 XOR 1 = 0

#include <stdio.h>
#include <math.h>
#include "../../core/complex.h"
#include "../../core/matrix.h"
#include "../../core/photonic.h"
#include "../../core/activation.h"
#include "../../core/loss.h"
#include "../../core/memory.h"
#include "../../sim/photonic_sim.h"
#include "../../training/gradient.h"
#include "../../training/unitary_sgd.h"

// XOR training data
static double xor_inputs[4][2] = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
};

static double xor_targets[4] = {0.0, 1.0, 1.0, 0.0};

int main(void) {
    printf("=== XOR Photonic Example ===\n");
    
    int num_layers = 2;
    int layer_dim = 4;
    
    SimState sim = sim_init(num_layers, layer_dim);
    if (!sim.layers) {
        printf("Failed to initialize simulator.\n");
        sim_free(&sim);
        return 1;
    }
    
    // Enable non-linearity (Kerr effect) to allow solving XOR
    for (int i = 0; i < num_layers; i++) {
        sim.layers[i].kerr_gamma = 0.5;
    }
    
    printf("\nRunning forward pass (Pre-training / Untrained Weights)...\n");
    double total_mse = 0.0;
    
    for (int i = 0; i < 4; i++) {
        Complex input[2];
        Complex output[2];
        
        input[0] = complex_new(xor_inputs[i][0], 0.0);
        input[1] = complex_new(xor_inputs[i][1], 0.0);
        
        sim_forward(&sim, input, output);
        
        // Use intensity of port 0 as the prediction
        double pred_intensity = complex_norm_sq(output[0]);
        // Simple sigmoid-like normalization for probability
        double prob = 1.0 / (1.0 + exp(-pred_intensity));
        
        double mse = loss_mse(&prob, &xor_targets[i], 1);
        total_mse += mse;
        
        printf("  Input: [%.0f, %.0f] -> Raw Out: %+.4f%+.4fi -> Prob: %.4f (Target: %.0f) | MSE: %.4f\n",
               xor_inputs[i][0], xor_inputs[i][1], 
               output[0].real, output[0].imag, 
               prob, xor_targets[i], mse);
    }
    
    printf("\nAverage Initial MSE: %.4f\n", total_mse / 4.0);
    
    XorDataset dataset;
    for (int i = 0; i < 4; i++) {
        dataset.inputs[i][0] = xor_inputs[i][0];
        dataset.inputs[i][1] = xor_inputs[i][1];
        dataset.targets[i] = xor_targets[i];
    }
    
    printf("\nStarting Riemannian Gradient Descent Training...\n");
    TrainConfig cfg = {
        .learning_rate = 0.05,
        .max_epochs    = 5000,
        .target_mse    = 0.01,
        .log_every     = 500
    };
    
    for (int epoch = 1; epoch <= cfg.max_epochs; epoch++) {
        for (int l = 0; l < sim.num_layers; l++) {
            Matrix grad = compute_numerical_gradient(&sim, l, &dataset, 1e-5);
            if (grad.data) {
                unitary_update_cayley(&sim.layers[l].weights, &grad, cfg.learning_rate);
                matrix_free(&grad);
            }
        }
        
        double current_mse = compute_loss(&sim, &dataset);
        if (epoch % cfg.log_every == 0 || current_mse < cfg.target_mse || epoch == 1) {
            printf("  Epoch %4d | MSE: %.6f\n", epoch, current_mse);
        }
        
        if (current_mse < cfg.target_mse) {
            printf("\nTarget MSE reached at epoch %d!\n", epoch);
            break;
        }
    }
    
    printf("\nRunning forward pass (Post-training)...\n");
    for (int i = 0; i < 4; i++) {
        int dim = sim.layer_dim;
        Complex *input = (Complex*)pho_alloc(dim * sizeof(Complex), "main.in");
        Complex *output = (Complex*)pho_alloc(dim * sizeof(Complex), "main.out");
        
        if (dim >= 4) {
            input[0] = complex_new(xor_inputs[i][0], 0.0);
            input[1] = complex_new(xor_inputs[i][1], 0.0);
            input[2] = complex_new(xor_inputs[i][0] * xor_inputs[i][1], 0.0);
            input[3] = complex_new(1.0, 0.0);
            for(int k=4; k<dim; k++) input[k] = complex_new(0,0);
        } else {
            for(int k=0; k<dim; k++) {
                if (k < 2) input[k] = complex_new(xor_inputs[i][k], 0.0);
                else input[k] = complex_new(0.0, 0.0);
            }
        }
        
        sim_forward(&sim, input, output);
        
        double pred_intensity = 0.0;
        if (dim >= 2) {
            pred_intensity = complex_norm_sq(output[0]) - complex_norm_sq(output[1]);
        } else {
            pred_intensity = complex_norm_sq(output[0]);
        }
        
        double prob = 1.0 / (1.0 + exp(-pred_intensity));
        
        printf("  Input: [%.0f, %.0f] -> Prob: %.4f (Target: %.0f)\n",
               xor_inputs[i][0], xor_inputs[i][1], prob, xor_targets[i]);
               
        pho_free(input, "main.in");
        pho_free(output, "main.out");
    }
    
    printf("\nEnergy consumed: %.4f pJ\n", sim.total_energy);
    printf("Total MAC ops: %lld\n", sim.total_ops);
    
    sim_free(&sim);
    return 0;
}
