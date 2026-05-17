#include "insitu_loop.h"
#include "hal.h"
#include "calibration.h"
#include "data_loader.h"
#include "pooling.h"
#include "loss.h"
#include "unitary_sgd.h"
#include "gradient_analytic.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int run_insitu_training(SimState *sim, const InSituConfig *cfg) {
    if (!sim || !cfg || !cfg->data_csv) {
        fprintf(stderr, "[In-Situ Error] Invalid simulator state or configurations.\n");
        return -1;
    }

    // Step 1: Initialize the physical or emulated chip interface
    printf("\n=== Initializing Silicon Photonics Chip Driver via HAL ===\n");
    if (hal_init_chip("EMULATION") != 0) {
        fprintf(stderr, "[In-Situ Error] Failed to initialize hardware interface.\n");
        return -1;
    }

    // Step 2: Load the training dataset
    ImageDataset dataset = {0};
    printf("Loading in-situ training dataset from %s...\n", cfg->data_csv);
    if (!load_mnist_csv(cfg->data_csv, 1000, &dataset)) {
        fprintf(stderr, "[In-Situ Error] Failed to load dataset.\n");
        hal_shutdown();
        return -1;
    }

    int dim = sim->layer_dim;
    int num_layers = sim->num_layers;
    int n_heaters = dim * dim;

    // Allocate memory for DAC control voltages
    double *dac_voltages = (double *)calloc(n_heaters, sizeof(double));
    double *adc_intensities = (double *)calloc(dim, sizeof(double));
    if (!dac_voltages || !adc_intensities) {
        fprintf(stderr, "[In-Situ Error] Memory allocation failed.\n");
        free(dac_voltages);
        free(adc_intensities);
        free_dataset(&dataset);
        hal_shutdown();
        return -1;
    }

    printf("\n=== Starting Silicon-on-Insulator In-Situ Hybrid Training Loop ===\n");
    printf("Epochs: %d, Batch Size: %d, Learning Rate: %.4f, Heaters/Layer: %d\n",
           cfg->epochs, cfg->batch_size, cfg->lr, n_heaters);

    for (int epoch = 1; epoch <= cfg->epochs; epoch++) {
        shuffle_dataset(&dataset);

        double total_loss = 0.0;
        int correct_predictions = 0;

        for (size_t i = 0; i < dataset.num_samples; i += cfg->batch_size) {
            size_t current_batch_size = (i + cfg->batch_size > dataset.num_samples) ? 
                                        (dataset.num_samples - i) : (size_t)cfg->batch_size;

            // Allocate gradient accumulators for each layer in C
            Matrix *batch_grads = (Matrix *)malloc(num_layers * sizeof(Matrix));
            for (int l = 0; l < num_layers; l++) {
                batch_grads[l] = matrix_new(dim, dim);
            }

            // Calibrate current complex weight matrices to micro-heater DAC voltages
            for (int l = 0; l < num_layers; l++) {
                calibrate_direct_phases_to_voltages(
                    &sim->layers[l].weights, 
                    &cfg->cal_params, 
                    dac_voltages
                );

                // Push calibrated DAC control voltages physically to MZI arrays
                hal_write_voltages(dac_voltages, n_heaters);
            }

            // Perform batch forward and backpropagation
            for (size_t b = 0; b < current_batch_size; b++) {
                size_t sample_idx = i + b;
                const double *img = dataset.samples[sample_idx].pixels;
                int label = dataset.samples[sample_idx].label;

                double target[10] = {0};
                target[label] = 1.0;

                // Perform optical pooling from 28x28 pixel input to 8x8 input channels
                double pool_buf[64];
                optical_lens_pool_28_to_8(img, pool_buf);

                Complex *input_c = (Complex *)calloc(dim, sizeof(Complex));
                for (int k = 0; k < dim; k++) {
                    if (k < 64) {
                        input_c[k] = complex_new(pool_buf[k], 0.0);
                    }
                }

                // Simulate physical chip forward pass under DAC thermal modulation
                Complex *output_c = (Complex *)calloc(dim, sizeof(Complex));
                sim_forward(sim, input_c, output_c);

                // Read physical ADC photodetector intensities
                hal_read_photodetectors(adc_intensities, dim);

                // Apply dynamic softmax scaling based on photodetector responsivity
                double probs[10] = {0};
                double gain = sim->layers[num_layers - 1].detector_gain;
                double loss = loss_cross_entropy_softmax_optical(output_c, gain, target, 10, probs);
                total_loss += loss;

                int max_class = 0;
                double max_prob = -1.0;
                for (int c = 0; c < 10; c++) {
                    if (probs[c] > max_prob) {
                        max_prob = probs[c];
                        max_class = c;
                    }
                }
                if (max_class == label) {
                    correct_predictions++;
                }

                // Compute CPU backpropagation gradients
                Complex *current_grad = (Complex *)calloc(dim, sizeof(Complex));
                loss_cross_entropy_softmax_optical_grad(output_c, gain, target, 10, current_grad);

                for (int l = num_layers - 1; l >= 0; l--) {
                    Complex *g_buf = (Complex *)calloc(dim * dim, sizeof(Complex));
                    Matrix g_mat = { .rows = dim, .cols = dim, .data = g_buf };
                    Complex *next_grad = (Complex *)calloc(dim, sizeof(Complex));

                    const Complex *l_input = (l == 0) ? input_c : sim->layer_outputs[l - 1];

                    photonic_layer_backward(
                        &sim->layers[l].weights,
                        l_input,
                        current_grad,
                        sim->layers[l].kerr_gamma,
                        &g_mat,
                        next_grad,
                        dim
                    );

                    // Accumulate gradients for dynamic batch update
                    for (int idx = 0; idx < dim * dim; idx++) {
                        batch_grads[l].data[idx] = complex_add(batch_grads[l].data[idx], g_mat.data[idx]);
                    }

                    memcpy(current_grad, next_grad, dim * sizeof(Complex));
                    free(g_buf);
                    free(next_grad);
                }

                free(input_c);
                free(output_c);
                free(current_grad);
            }

            // Apply C Riemannian Cayley Unitary Updates
            for (int l = 0; l < num_layers; l++) {
                unitary_update_cayley(&sim->layers[l].weights, &batch_grads[l], cfg->lr);
                matrix_free(&batch_grads[l]);
            }
            free(batch_grads);
        }

        double epoch_loss = total_loss / dataset.num_samples;
        double epoch_acc = (double)correct_predictions / dataset.num_samples * 100.0;
        printf("In-Situ Epoch %2d | Loss: %.6f | Accuracy: %.2f%%\n", epoch, epoch_loss, epoch_acc);
    }

    // Free buffers and safely shutdown chip heaters
    free(dac_voltages);
    free(adc_intensities);
    free_dataset(&dataset);
    hal_shutdown();

    printf("=== In-Situ Training Completed Successfully (Chip reset to 0.0V) ===\n");
    return 0;
}
