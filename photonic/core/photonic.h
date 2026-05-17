// photonic.h – abstraction for photonic layers
#ifndef PHOTONIC_H
#define PHOTONIC_H

#include "complex.h"
#include "matrix.h"

// Define a photonic layer type (placeholder)
typedef struct {
    Matrix weights;
    double kerr_gamma; // Optional Kerr effect coefficient
    double detector_gain; // TIA gain after photodetector
    double phase_noise; // Waveguide phase noise std dev
} PhotonicLayer;

// Create a new photonic layer with identity weights
PhotonicLayer photonic_layer_init(size_t dim);

// Free layer resources
void photonic_layer_free(PhotonicLayer *layer);

// Check if a layer's weight matrix is unitary
int photonic_layer_is_valid(const PhotonicLayer *layer, double epsilon);

// Forward pass for a single layer
void photonic_layer_forward(const PhotonicLayer *layer, const Complex *input, Complex *output);

#endif // PHOTONIC_H
