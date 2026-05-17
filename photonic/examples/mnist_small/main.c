#include "../../core/photonic.h"
#include "../../core/activation.h"
#include "../../core/loss.h"
#include "../../core/memory.h"
#include "../../sim/photonic_sim.h"
#include "../../training/data_loader.h"
#include "../../training/gradient.h"
#include "../../training/unitary_sgd.h"
#include "pooling.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Helper to compute matrix norm
double compute_matrix_norm(const Matrix *m) {
    double sum = 0.0;
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        sum += complex_norm_sq(m->data[i]);
    }
    return sqrt(sum);
}

// Batch loss calculation for MNIST
double compute_mnist_batch_loss(SimState *sim, ImageSample *samples, int start_idx, int batch_size) {
    double total_loss = 0.0;
    int dim = sim->layer_dim;
    Complex *net_input = (Complex*)pho_alloc(dim * sizeof(Complex), "loss.in");
    Complex *net_output = (Complex*)pho_alloc(dim * sizeof(Complex), "loss.out");
    double *pooled_input = (double*)pho_alloc(dim * sizeof(double), "loss.pooled");
    double *probabilities = (double*)pho_alloc(dim * sizeof(double), "loss.probs");
    
    // One-hot target buffer (max dim size)
    double *target_oh = (double*)pho_alloc(dim * sizeof(double), "loss.target");
    for (int k = 0; k < dim; k++) target_oh[k] = 0.0;

    for (int i = 0; i < batch_size; i++) {
        ImageSample *sample = &samples[start_idx + i];
        
        // Pool 28x28 -> 4x4
        optical_lens_pool_28_to_4(sample->pixels, pooled_input);
        
        for (int k = 0; k < dim; k++) {
            net_input[k] = complex_new(pooled_input[k], 0.0);
        }
        sim_forward(sim, net_input, net_output);
        
        for(int k=0; k<10; k++) target_oh[k] = 0.0;
        target_oh[sample->label] = 1.0;

        double loss = loss_cross_entropy_softmax_optical(
            net_output, 
            sim->layers[1].detector_gain, 
            target_oh, 
            10, 
            probabilities
        );
        total_loss += loss;
    }

    pho_free(net_input, "loss.in");
    pho_free(net_output, "loss.out");
    pho_free(pooled_input, "loss.pooled");
    pho_free(probabilities, "loss.probs");
    pho_free(target_oh, "loss.target");
    return total_loss / batch_size;
}

// Numerical gradient calculation for a layer on a batch using Finite Difference
Matrix compute_mnist_layer_gradient(SimState *sim, int layer_idx, ImageSample *samples, int start_idx, int batch_size, double h) {
    PhotonicLayer *layer = &sim->layers[layer_idx];
    size_t dim = layer->weights.rows;
    Matrix grad = matrix_new(dim, dim);
    if (!grad.data) return grad;

    for (size_t i = 0; i < dim * dim; i++) {
        Complex orig_w = layer->weights.data[i];

        // --- Real ---
        layer->weights.data[i] = complex_new(orig_w.real + h, orig_w.imag);
        double loss_plus_real = compute_mnist_batch_loss(sim, samples, start_idx, batch_size);

        layer->weights.data[i] = complex_new(orig_w.real - h, orig_w.imag);
        double loss_minus_real = compute_mnist_batch_loss(sim, samples, start_idx, batch_size);

        double dL_dreal = (loss_plus_real - loss_minus_real) / (2.0 * h);

        // --- Imag ---
        layer->weights.data[i] = complex_new(orig_w.real, orig_w.imag + h);
        double loss_plus_imag = compute_mnist_batch_loss(sim, samples, start_idx, batch_size);

        layer->weights.data[i] = complex_new(orig_w.real, orig_w.imag - h);
        double loss_minus_imag = compute_mnist_batch_loss(sim, samples, start_idx, batch_size);

        double dL_dimag = (loss_plus_imag - loss_minus_imag) / (2.0 * h);

        // Restore
        layer->weights.data[i] = orig_w;
        grad.data[i] = complex_new(dL_dreal, dL_dimag);
    }
    return grad;
}

int main() {
    printf("=== MNIST Small Photonic Example ===\n");
    srand(42);
    
    // 1. Load dataset
    ImageDataset dataset;
    if (!load_mnist_csv("data/mnist_small.csv", 1000, &dataset)) {
        fprintf(stderr, "Failed to load MNIST dataset. Ensure data/mnist_small.csv exists.\n");
        return 1;
    }
    printf("Loaded %zu samples.\n", dataset.num_samples);
    
    // 2. Initialize Network (2 layers: hidden + output)
    int num_layers = 2;
    int layer_dim = 16; // Using 16x16 instead of 64x64 for 256x speedup!
    
    SimState sim = sim_init(num_layers, layer_dim);
    if (!sim.layers) {
        fprintf(stderr, "Failed to initialize simulation.\n");
        free_dataset(&dataset);
        return 1;
    }
    
    // Set parameters and perturb weights slightly to break initialization symmetry
    for (int l = 0; l < num_layers; l++) {
        sim.layers[l].kerr_gamma = 0.1;      // Enable Kerr optical non-linearity
        sim.layers[l].detector_gain = 3.0;   // Set photodetector TIA Gain
        
        for (int i = 0; i < layer_dim; i++) {
            for (int j = 0; j < layer_dim; j++) {
                double noise_r = ((double)rand() / RAND_MAX - 0.5) * 0.05;
                double noise_i = ((double)rand() / RAND_MAX - 0.5) * 0.05;
                if (i == j) {
                    sim.layers[l].weights.data[i * layer_dim + j] = complex_new(1.0 + noise_r, noise_i);
                } else {
                    sim.layers[l].weights.data[i * layer_dim + j] = complex_new(noise_r, noise_i);
                }
            }
        }
    }
    
    // 3. Training Loop Setup
    int epochs = 5; 
    int num_train = 100; // Limit training samples for fast execution
    int batch_size = 10;
    double lr = 1.0;
    double h = 1e-4; // Finite difference perturbation step
    
    Complex *net_input = (Complex*)pho_alloc(layer_dim * sizeof(Complex), "net_input");
    Complex *net_output = (Complex*)pho_alloc(layer_dim * sizeof(Complex), "net_output");
    double *pooled_input = (double*)pho_alloc(layer_dim * sizeof(double), "pooled_input");
    double *probabilities = (double*)pho_alloc(layer_dim * sizeof(double), "probabilities");
    double *target_oh = (double*)pho_alloc(layer_dim * sizeof(double), "target_oh");
    for (int k = 0; k < layer_dim; k++) target_oh[k] = 0.0;
    
    printf("Starting SGD Training Loop (16x16 network, %d samples, %d epochs, batch size %d)...\n", num_train, epochs, batch_size);
    for (int epoch = 1; epoch <= epochs; epoch++) {
        shuffle_dataset(&dataset);
        
        double total_loss = 0.0;
        int correct = 0;
        
        // 4. Batch Training
        for (int start_idx = 0; start_idx < num_train; start_idx += batch_size) {
            
            // Print progress for first sample first epoch
            if (epoch == 1 && start_idx == 0) {
                // Readout first sample data
                ImageSample *sample = &dataset.samples[0];
                double pixel_sum = 0.0;
                for (int p = 0; p < 784; p++) pixel_sum += sample->pixels[p];
                printf("  [Debug] Sample 0 (Label: %d) Sum of raw pixels: %.4f\n", sample->label, pixel_sum);
                
                optical_lens_pool_28_to_4(sample->pixels, pooled_input);
                printf("  [Debug] First Sample Pooling outputs (4x4 grid):\n");
                for (int r = 0; r < 4; r++) {
                    printf("    ");
                    for (int c = 0; c < 4; c++) {
                        printf("%.4f ", pooled_input[r * 4 + c]);
                    }
                    printf("\n");
                }
            }
            
            // Compute numerical gradients for both layers
            Matrix grad_l0 = compute_mnist_layer_gradient(&sim, 0, dataset.samples, start_idx, batch_size, h);
            Matrix grad_l1 = compute_mnist_layer_gradient(&sim, 1, dataset.samples, start_idx, batch_size, h);
            
            if (epoch == 1 && start_idx == 0) {
                printf("  [Debug] Grad norm layer0: %.8f | layer1: %.8f (Symmetry Broken!)\n", 
                       compute_matrix_norm(&grad_l0), compute_matrix_norm(&grad_l1));
            }
            
            // Perform Cayley updates
            unitary_update_cayley(&sim.layers[0].weights, &grad_l0, lr);
            unitary_update_cayley(&sim.layers[1].weights, &grad_l1, lr);
            
            matrix_free(&grad_l0);
            matrix_free(&grad_l1);
        }
        
        // 5. Evaluate after each epoch
        for (int i = 0; i < num_train; i++) {
            ImageSample *sample = &dataset.samples[i];
            
            optical_lens_pool_28_to_4(sample->pixels, pooled_input);
            for (int k = 0; k < layer_dim; k++) {
                net_input[k] = complex_new(pooled_input[k], 0.0);
            }
            
            sim_forward(&sim, net_input, net_output);
            
            for(int k=0; k<10; k++) target_oh[k] = 0.0;
            target_oh[sample->label] = 1.0;
            
            double loss = loss_cross_entropy_softmax_optical(
                net_output, 
                sim.layers[1].detector_gain, 
                target_oh, 
                10, 
                probabilities
            );
            total_loss += loss;
            
            int pred_class = 0;
            double max_prob = probabilities[0];
            for (int k = 1; k < 10; k++) {
                if (probabilities[k] > max_prob) {
                    max_prob = probabilities[k];
                    pred_class = k;
                }
            }
            
            if (pred_class == sample->label) {
                correct++;
            }
        }
        
        double accuracy = (double)correct / num_train * 100.0;
        printf("Epoch %3d | Loss: %.4f | Accuracy: %.2f%%\n", epoch, total_loss / num_train, accuracy);
    }
    
    printf("\nEnergy consumed: %.4f pJ\n", sim.total_energy);
    printf("Total MAC ops: %lld\n", sim.total_ops);
    
    pho_free(net_input, "net_input");
    pho_free(net_output, "net_output");
    pho_free(pooled_input, "pooled_input");
    pho_free(probabilities, "probabilities");
    pho_free(target_oh, "target_oh");
    
    sim_free(&sim);
    free_dataset(&dataset);
    
    return 0;
}
