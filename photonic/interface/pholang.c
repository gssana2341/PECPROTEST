// pholang.c — Implementation of the public C API for the Photonic library
#include "pholang.h"
#include "../core/complex.h"
#include "../core/matrix.h"
#include "../core/photonic.h"
#include "../core/activation.h"
#include "../core/loss.h"
#include "../core/memory.h"
#include "../lang/ast.h"
#include "../lang/lexer.h"
ASTNode *parser_parse(const char *source);
#include "../examples/mnist_small/pooling.h"
#include "../training/unitary_sgd.h"
#include "../training/gradient_analytic.h"
#include "../training/data_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Concrete PhoNetwork implementation
struct PhoNetwork {
    SimState sim;
    double   learning_rate;
    int      epochs;
    int      batch_size;
    int      early_stop;
    double   phase_noise;
    double   shot_noise;
};

// Thread-local or global error storage
static char g_last_error[512] = "";

const char* pho_last_error(void) {
    return g_last_error;
}

static void set_last_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, args);
    va_end(args);
}

// Helper to inject phase noise to weights
static void inject_weights_phase_noise(SimState *sim, double noise_std) {
    if (noise_std <= 0.0) return;
    int dim = sim->layer_dim;
    for (int l = 0; l < sim->num_layers; l++) {
        for (int i = 0; i < dim; i++) {
            // Sample random phase drift using Box-Muller transform
            double u1 = (double)rand() / RAND_MAX;
            double u2 = (double)rand() / RAND_MAX;
            if (u1 < 1e-15) u1 = 1e-15;
            double z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.141592653589793 * u2);
            double phase_drift = z * noise_std;
            
            Complex phase_factor = complex_new(cos(phase_drift), sin(phase_drift));
            for (int j = 0; j < dim; j++) {
                sim->layers[l].weights.data[i * dim + j] = complex_mul(sim->layers[l].weights.data[i * dim + j], phase_factor);
            }
        }
    }
}

// 1. Create a network by parsing a .pho definition file
PhoNetwork* pho_network_load(const char* pho_file) {
    FILE *f = fopen(pho_file, "r");
    if (!f) {
        set_last_error("Failed to open file: %s", pho_file);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(size + 1);
    if (!buf) {
        set_last_error("Out of memory reading file: %s", pho_file);
        fclose(f);
        return NULL;
    }
    size_t read_bytes = fread(buf, 1, size, f);
    buf[read_bytes] = '\0';
    fclose(f);

    ASTNode *ast = parser_parse(buf);
    free(buf);
    if (!ast) {
        set_last_error("Failed to parse PhoLang code");
        return NULL;
    }

    PhoNetwork *net = (PhoNetwork *)malloc(sizeof(PhoNetwork));
    if (!net) {
        set_last_error("Out of memory allocating PhoNetwork");
        ast_free(ast);
        return NULL;
    }
    memset(net, 0, sizeof(PhoNetwork));

    // Default metadata
    net->learning_rate = 0.01;
    net->epochs = 100;
    net->batch_size = 32;
    net->early_stop = 10;
    net->phase_noise = 0.0;
    net->shot_noise = 0.0;

    // Count layers
    int num_layers = 0;
    int layer_dim = 64; // default
    ASTNode *child = ast->children;
    while (child) {
        if (child->type == AST_LAYER_DECL) {
            if (strcmp(child->value, "unitary") == 0) {
                num_layers++;
                if (child->children && child->children->type == AST_NUMBER_LITERAL) {
                    layer_dim = (int)child->children->num_value;
                }
            }
        }
        child = child->next;
    }

    if (num_layers == 0) {
        set_last_error("No photonic unitary layers found in network definition");
        ast_free(ast);
        free(net);
        return NULL;
    }

    net->sim = sim_init(num_layers, layer_dim);

    // Second pass: configure layers and training block
    int current_layer = 0;
    child = ast->children;
    while (child) {
        if (child->type == AST_LAYER_DECL && strcmp(child->value, "unitary") == 0) {
            double kerr = 0.5; // default
            double gain = 15.0; // default
            ASTNode *attr = child->children;
            if (attr && attr->type == AST_NUMBER_LITERAL) {
                attr = attr->next;
            }
            while (attr) {
                if (attr->type == AST_CONFIG_PAIR) {
                    if (strcmp(attr->value, "kerr") == 0) {
                        kerr = attr->num_value;
                    } else if (strcmp(attr->value, "gain") == 0) {
                        gain = attr->num_value;
                    }
                }
                attr = attr->next;
            }
            net->sim.layers[current_layer].kerr_gamma = kerr;
            net->sim.layers[current_layer].detector_gain = gain;
            
            // Initialize with unitary matrices perturbed by slight noise to break symmetry
            for (int i = 0; i < layer_dim; i++) {
                for (int j = 0; j < layer_dim; j++) {
                    double noise_r = ((double)rand() / RAND_MAX - 0.5) * 0.1;
                    double noise_i = ((double)rand() / RAND_MAX - 0.5) * 0.1;
                    if (i == j) {
                        net->sim.layers[current_layer].weights.data[i * layer_dim + j] = complex_new(1.0 + noise_r, noise_i);
                    } else {
                        net->sim.layers[current_layer].weights.data[i * layer_dim + j] = complex_new(noise_r, noise_i);
                    }
                }
            }
            current_layer++;
        } else if (child->type == AST_TRAIN_BLOCK) {
            ASTNode *pair = child->children;
            while (pair) {
                if (pair->type == AST_CONFIG_PAIR) {
                    if (strcmp(pair->value, "lr") == 0) {
                        net->learning_rate = pair->num_value;
                    } else if (strcmp(pair->value, "epochs") == 0) {
                        net->epochs = (int)pair->num_value;
                    } else if (strcmp(pair->value, "batch_size") == 0) {
                        net->batch_size = (int)pair->num_value;
                    } else if (strcmp(pair->value, "early_stop") == 0) {
                        net->early_stop = (int)pair->num_value;
                    } else if (strcmp(pair->value, "phase_noise") == 0) {
                        net->phase_noise = pair->num_value;
                    } else if (strcmp(pair->value, "shot_noise") == 0) {
                        net->shot_noise = pair->num_value;
                    }
                }
                pair = pair->next;
            }
        }
        child = child->next;
    }

    ast_free(ast);
    return net;
}

// 2. Train the network on a balanced class CSV dataset
PhoResult pho_network_train(PhoNetwork* net, const char* data_csv, PhoTrainConfig cfg) {
    PhoResult final_res = {0.0, 0.0};
    if (!net || !data_csv) {
        set_last_error("Invalid network or dataset path");
        return final_res;
    }

    ImageDataset dataset = {0};
    // Load dataset dynamically
    if (!load_mnist_csv(data_csv, 6000, &dataset)) {
        set_last_error("Failed to load dataset from path: %s", data_csv);
        return final_res;
    }

    shuffle_dataset(&dataset);

    // Split 80% train, 20% validation test
    int num_train = (int)(dataset.num_samples * 0.8);
    int num_test = dataset.num_samples - num_train;

    int epochs = (cfg.epochs > 0) ? cfg.epochs : net->epochs;
    double lr = (cfg.lr > 0.0) ? cfg.lr : net->learning_rate;
    int batch_size = (cfg.batch_size > 0) ? cfg.batch_size : net->batch_size;

    int dim = net->sim.layer_dim;
    int num_layers = net->sim.num_layers;

    double best_test_loss = 1e9;
    double best_test_acc = 0.0;
    int no_improvement_epochs = 0;

    // Allocate backup weights for early stopping
    Matrix *best_weights = (Matrix *)malloc(num_layers * sizeof(Matrix));
    for (int l = 0; l < num_layers; l++) {
        best_weights[l] = matrix_copy(&net->sim.layers[l].weights);
    }

    printf("=== Starting PhoLang Public C API Dynamic Training Loop ===\n");
    printf("Layers: %d, Dimension: %dx%d, lr: %.4f, batch: %d, epochs: %d\n", 
           num_layers, dim, dim, lr, batch_size, epochs);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        double total_loss = 0.0;
        int correct_predictions = 0;

        for (int i = 0; i < num_train; i += batch_size) {
            int current_batch_size = (i + batch_size > num_train) ? (num_train - i) : batch_size;

            // Batch accumulators for weight gradients
            Matrix *batch_grad_accum = (Matrix *)malloc(num_layers * sizeof(Matrix));
            for (int l = 0; l < num_layers; l++) {
                batch_grad_accum[l] = matrix_new(dim, dim);
            }

            // Create temporary backup of weights to inject phase noise
            SimState sim_noisy = sim_init(num_layers, dim);
            for (int l = 0; l < num_layers; l++) {
                matrix_free(&sim_noisy.layers[l].weights);
                sim_noisy.layers[l].weights = matrix_copy(&net->sim.layers[l].weights);
                sim_noisy.layers[l].kerr_gamma = net->sim.layers[l].kerr_gamma;
                sim_noisy.layers[l].detector_gain = net->sim.layers[l].detector_gain;
            }
            sim_noisy.shot_noise = net->shot_noise;

            inject_weights_phase_noise(&sim_noisy, net->phase_noise);

            #pragma omp parallel for reduction(+:total_loss, correct_predictions)
            for (int b = 0; b < current_batch_size; b++) {
                int sample_idx = i + b;
                const double *img = dataset.samples[sample_idx].pixels;
                int label = dataset.samples[sample_idx].label;

                double target[10] = {0};
                target[label] = 1.0;

                // Process optical pooling
                double pool_buf[64];
                optical_lens_pool_28_to_8(img, pool_buf);

                Complex input_c[dim];
                for (int k = 0; k < dim; k++) {
                    if (k < 64) {
                        input_c[k] = complex_new(pool_buf[k], 0.0);
                    } else {
                        input_c[k] = complex_new(0.0, 0.0);
                    }
                }

                Complex output_c[dim];
                sim_forward(&sim_noisy, input_c, output_c);

                double probs[10] = {0};
                double gain = sim_noisy.layers[num_layers - 1].detector_gain;
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

                // Dynamic analytical backpropagation
                Complex current_grad[dim];
                memset(current_grad, 0, dim * sizeof(Complex));
                loss_cross_entropy_softmax_optical_grad(output_c, gain, target, 10, current_grad);

                for (int l = num_layers - 1; l >= 0; l--) {
                    Complex g_buf[dim * dim];
                    Matrix g_mat = { .rows = dim, .cols = dim, .data = g_buf };
                    Complex next_grad[dim];

                    const Complex *l_input = (l == 0) ? input_c : sim_noisy.layer_outputs[l - 1];

                    photonic_layer_backward(
                        &sim_noisy.layers[l].weights,
                        l_input,
                        current_grad,
                        sim_noisy.layers[l].kerr_gamma,
                        &g_mat,
                        next_grad,
                        dim
                    );

                    #pragma omp critical
                    {
                        for (int idx = 0; idx < dim * dim; idx++) {
                            batch_grad_accum[l].data[idx] = complex_add(batch_grad_accum[l].data[idx], g_mat.data[idx]);
                        }
                    }

                    memcpy(current_grad, next_grad, dim * sizeof(Complex));
                }
            }

            // Apply Riemannian SGD Unitary Updates (Cayley Transform)
            for (int l = 0; l < num_layers; l++) {
                // Average the accumulated batch gradients over the batch size
                for (int idx = 0; idx < dim * dim; idx++) {
                    batch_grad_accum[l].data[idx] = complex_new(
                        batch_grad_accum[l].data[idx].real / (double)current_batch_size,
                        batch_grad_accum[l].data[idx].imag / (double)current_batch_size
                    );
                }
                unitary_update_cayley(&net->sim.layers[l].weights, &batch_grad_accum[l], lr);
                matrix_free(&batch_grad_accum[l]);
            }
            free(batch_grad_accum);
            sim_free(&sim_noisy);
        }

        double train_loss = total_loss / num_train;
        double train_acc = (double)correct_predictions / num_train * 100.0;

        // Validation testing pass
        double test_loss = 0.0;
        int test_correct = 0;

        #pragma omp parallel for reduction(+:test_loss, test_correct)
        for (int i = 0; i < num_test; i++) {
            int sample_idx = num_train + i;
            const double *img = dataset.samples[sample_idx].pixels;
            int label = dataset.samples[sample_idx].label;

            double target[10] = {0};
            target[label] = 1.0;

            double pool_buf[64];
            optical_lens_pool_28_to_8(img, pool_buf);

            Complex input_c[dim];
            for (int k = 0; k < dim; k++) {
                if (k < 64) {
                    input_c[k] = complex_new(pool_buf[k], 0.0);
                } else {
                    input_c[k] = complex_new(0.0, 0.0);
                }
            }

            Complex output_c[dim];
            sim_forward(&net->sim, input_c, output_c);

            double probs[10] = {0};
            double gain = net->sim.layers[num_layers - 1].detector_gain;
            double loss = loss_cross_entropy_softmax_optical(output_c, gain, target, 10, probs);
            test_loss += loss;

            int max_class = 0;
            double max_prob = -1.0;
            for (int c = 0; c < 10; c++) {
                if (probs[c] > max_prob) {
                    max_prob = probs[c];
                    max_class = c;
                }
            }
            if (max_class == label) {
                test_correct++;
            }
        }

        double val_loss = test_loss / num_test;
        double val_acc = (double)test_correct / num_test * 100.0;

        printf("Epoch %3d | Train Loss: %.6f | Train Acc: %.2f%% | Test Loss: %.6f | Test Acc: %.2f%%\n",
               epoch, train_loss, train_acc, val_loss, val_acc);

        // Early stopping check
        if (val_loss < best_test_loss) {
            best_test_loss = val_loss;
            best_test_acc = val_acc;
            no_improvement_epochs = 0;
            for (int l = 0; l < num_layers; l++) {
                matrix_free(&best_weights[l]);
                best_weights[l] = matrix_copy(&net->sim.layers[l].weights);
            }
        } else {
            no_improvement_epochs++;
            if (net->early_stop > 0 && no_improvement_epochs >= net->early_stop) {
                printf("Early stopping triggered! Restoring best weights from epoch with Test Loss: %.6f\n", best_test_loss);
                for (int l = 0; l < num_layers; l++) {
                    matrix_free(&net->sim.layers[l].weights);
                    net->sim.layers[l].weights = matrix_copy(&best_weights[l]);
                }
                break;
            }
        }
    }

    for (int l = 0; l < num_layers; l++) {
        matrix_free(&best_weights[l]);
    }
    free(best_weights);
    free_dataset(&dataset);

    final_res.final_loss = best_test_loss;
    final_res.final_accuracy = best_test_acc;
    return final_res;
}

// 3. Predict the digit category for a raw single image input array
int pho_network_predict(PhoNetwork* net, const double* input, int input_len) {
    if (!net || !input) return -1;
    if (input_len != 784) {
        set_last_error("Invalid input array length (Expected 784, got %d)", input_len);
        return -1;
    }

    double pool_buf[64];
    optical_lens_pool_28_to_8(input, pool_buf);

    int dim = net->sim.layer_dim;
    Complex *input_c = (Complex *)malloc(dim * sizeof(Complex));
    Complex *output_c = (Complex *)malloc(dim * sizeof(Complex));

    for (int i = 0; i < dim; i++) {
        if (i < 64) {
            input_c[i] = complex_new(pool_buf[i], 0.0);
        } else {
            input_c[i] = complex_new(0.0, 0.0);
        }
    }

    sim_forward(&net->sim, input_c, output_c);

    double gain = net->sim.layers[net->sim.num_layers - 1].detector_gain;
    double max_val = -1e9;
    double *vals = (double *)malloc(dim * sizeof(double));
    for (int i = 0; i < dim; i++) {
        double intensity = complex_norm_sq(output_c[i]);
        vals[i] = intensity * gain;
        if (vals[i] > max_val) {
            max_val = vals[i];
        }
    }

    double sum_exp = 0.0;
    for (int i = 0; i < dim; i++) {
        double e = exp(vals[i] - max_val);
        vals[i] = e;
        sum_exp += e;
    }

    int pred_class = 0;
    double max_prob = -1.0;
    for (int i = 0; i < 10; i++) {
        double prob = vals[i] / sum_exp;
        if (prob > max_prob) {
            max_prob = prob;
            pred_class = i;
        }
    }

    free(input_c);
    free(output_c);
    free(vals);

    return pred_class;
}

// 4. Save and restore trained weight matrices
int pho_network_save(PhoNetwork* net, const char* path) {
    if (!net || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) {
        set_last_error("Failed to open file for writing weights: %s", path);
        return -1;
    }

    fwrite("PHOM", 1, 4, f);
    fwrite(&net->sim.num_layers, sizeof(int), 1, f);
    fwrite(&net->sim.layer_dim, sizeof(int), 1, f);

    int dim = net->sim.layer_dim;
    for (int l = 0; l < net->sim.num_layers; l++) {
        fwrite(&net->sim.layers[l].kerr_gamma, sizeof(double), 1, f);
        fwrite(&net->sim.layers[l].detector_gain, sizeof(double), 1, f);
        fwrite(net->sim.layers[l].weights.data, sizeof(Complex), dim * dim, f);
    }

    fclose(f);
    return 0;
}

PhoNetwork* pho_network_load_weights(const char* pho_file, const char* weights_path) {
    PhoNetwork *net = pho_network_load(pho_file);
    if (!net) return NULL;

    FILE *f = fopen(weights_path, "rb");
    if (!f) {
        set_last_error("Failed to open weights file: %s", weights_path);
        pho_network_free(net);
        return NULL;
    }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "PHOM", 4) != 0) {
        set_last_error("Invalid weight file magic number");
        fclose(f);
        pho_network_free(net);
        return NULL;
    }

    int num_layers = 0;
    int layer_dim = 0;
    if (fread(&num_layers, sizeof(int), 1, f) != 1 ||
        fread(&layer_dim, sizeof(int), 1, f) != 1) {
        set_last_error("Failed to read weights file header");
        fclose(f);
        pho_network_free(net);
        return NULL;
    }

    if (num_layers != net->sim.num_layers || layer_dim != net->sim.layer_dim) {
        set_last_error("Weights structure mismatch (Expected layers=%d, dim=%d; Got layers=%d, dim=%d)",
                       net->sim.num_layers, net->sim.layer_dim, num_layers, layer_dim);
        fclose(f);
        pho_network_free(net);
        return NULL;
    }

    for (int l = 0; l < num_layers; l++) {
        double kerr = 0.0;
        double gain = 0.0;
        fread(&kerr, sizeof(double), 1, f);
        fread(&gain, sizeof(double), 1, f);
        net->sim.layers[l].kerr_gamma = kerr;
        net->sim.layers[l].detector_gain = gain;

        int read_count = fread(net->sim.layers[l].weights.data, sizeof(Complex), layer_dim * layer_dim, f);
        if (read_count != layer_dim * layer_dim) {
            set_last_error("Failed to read complete weight matrix for layer %d", l);
            fclose(f);
            pho_network_free(net);
            return NULL;
        }
    }

    fclose(f);
    return net;
}

// 5. Free network structure allocations
void pho_network_free(PhoNetwork* net) {
    if (net) {
        sim_free(&net->sim);
        free(net);
    }
}
