#include "gradient.h"
#include "../core/loss.h"
#include "../core/complex.h"
#include "../core/memory.h"
#include <math.h>

double compute_loss(SimState *sim, const XorDataset *dataset) {
    double total_mse = 0.0;
    int dim = sim->layer_dim;
    for (int i = 0; i < 4; i++) {
        Complex *input = (Complex*)pho_alloc(dim * sizeof(Complex), "loss.in");
        Complex *output = (Complex*)pho_alloc(dim * sizeof(Complex), "loss.out");
        
        // 4x4 Photonic Embedding
        if (dim >= 4) {
            input[0] = complex_new(dataset->inputs[i][0], 0.0);
            input[1] = complex_new(dataset->inputs[i][1], 0.0);
            input[2] = complex_new(dataset->inputs[i][0] * dataset->inputs[i][1], 0.0);
            input[3] = complex_new(1.0, 0.0);
            for(int k=4; k<dim; k++) input[k] = complex_new(0,0);
        } else {
            for(int k=0; k<dim; k++) {
                if (k < 2) input[k] = complex_new(dataset->inputs[i][k], 0.0);
                else input[k] = complex_new(0.0, 0.0);
            }
        }
        
        sim_forward(sim, input, output);
        
        // Dual-rail readout: Int(port0) - Int(port1)
        double pred_intensity = 0.0;
        if (dim >= 2) {
            pred_intensity = complex_norm_sq(output[0]) - complex_norm_sq(output[1]);
        } else {
            pred_intensity = complex_norm_sq(output[0]);
        }
        
        double prob = 1.0 / (1.0 + exp(-pred_intensity));
        
        double mse = loss_mse(&prob, &dataset->targets[i], 1);
        total_mse += mse;
        
        pho_free(input, "loss.in");
        pho_free(output, "loss.out");
    }
    return total_mse / 4.0;
}

Matrix compute_numerical_gradient(SimState *sim, int layer_idx, const XorDataset *dataset, double h) {
    PhotonicLayer *layer = &sim->layers[layer_idx];
    size_t dim = layer->weights.rows;
    
    Matrix grad = matrix_new(dim, dim);
    if (!grad.data) return grad;
    
    for (size_t i = 0; i < dim * dim; i++) {
        Complex orig_w = layer->weights.data[i];
        
        // --- Real part gradient ---
        layer->weights.data[i] = complex_new(orig_w.real + h, orig_w.imag);
        double loss_plus_real = compute_loss(sim, dataset);
        
        layer->weights.data[i] = complex_new(orig_w.real - h, orig_w.imag);
        double loss_minus_real = compute_loss(sim, dataset);
        
        double dL_dreal = (loss_plus_real - loss_minus_real) / (2.0 * h);
        
        // --- Imaginary part gradient ---
        layer->weights.data[i] = complex_new(orig_w.real, orig_w.imag + h);
        double loss_plus_imag = compute_loss(sim, dataset);
        
        layer->weights.data[i] = complex_new(orig_w.real, orig_w.imag - h);
        double loss_minus_imag = compute_loss(sim, dataset);
        
        double dL_dimag = (loss_plus_imag - loss_minus_imag) / (2.0 * h);
        
        // Restore original weight
        layer->weights.data[i] = orig_w;
        
        // Store complex gradient: G = dL/dRe + i * dL/dIm
        grad.data[i] = complex_new(dL_dreal, dL_dimag);
    }
    
    return grad;
}
