#include "photonic.h"
#include "activation.h"
#include <stddef.h>

PhotonicLayer photonic_layer_init(size_t dim) {
    PhotonicLayer layer;
    layer.kerr_gamma = 0.0; // Default: no non-linear phase shift
    layer.detector_gain = 1.0; // Default TIA gain
    layer.phase_noise = 0.0; // Default: no phase noise
    
    if (dim == 0) {
        layer.weights = matrix_new(0, 0);
        return layer;
    }
    
    layer.weights = matrix_new(dim, dim);
    if (!layer.weights.data) return layer;
    
    // Initialize to identity matrix (unitary)
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            if (i == j) {
                layer.weights.data[i * dim + j] = complex_new(1.0, 0.0);
            } else {
                layer.weights.data[i * dim + j] = complex_new(0.0, 0.0);
            }
        }
    }
    
    return layer;
}

int photonic_layer_is_valid(const PhotonicLayer *layer, double epsilon) {
    if (!layer || !layer->weights.data) return 0;
    return matrix_is_unitary(&layer->weights, epsilon);
}

void photonic_layer_free(PhotonicLayer *layer) {
    if (layer) {
        matrix_free(&layer->weights);
    }
}

#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double rand_normal_photonic(double mean, double std) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    if (u1 < 1e-15) u1 = 1e-15;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + z * std;
}

void photonic_layer_forward(const PhotonicLayer *layer, const Complex *input, Complex *output) {
    if (!layer || !input || !output) return;
    
    size_t dim = layer->weights.rows;
    
    // Matrix-vector multiplication for optical interference
    for (size_t i = 0; i < dim; i++) {
        Complex sum = complex_new(0.0, 0.0);
        for (size_t j = 0; j < dim; j++) {
            Complex w = layer->weights.data[i * dim + j];
            sum = complex_add(sum, complex_mul(w, input[j]));
        }
        
        // Apply waveguide phase noise before Kerr effect/readout
        if (layer->phase_noise > 0.0) {
            double phi = rand_normal_photonic(0.0, layer->phase_noise);
            Complex phase_factor = complex_new(cos(phi), sin(phi));
            sum = complex_mul(sum, phase_factor);
        }
        
        // Apply Kerr effect if gamma is set
        if (layer->kerr_gamma != 0.0) {
            output[i] = activation_kerr(sum, layer->kerr_gamma);
        } else {
            output[i] = sum;
        }
    }
}
