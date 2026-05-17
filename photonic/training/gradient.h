#ifndef GRADIENT_H
#define GRADIENT_H

#include "../core/matrix.h"
#include "../sim/photonic_sim.h"

// Define XOR dataset type for gradient functions
typedef struct {
    double inputs[4][2];
    double targets[4];
} XorDataset;

// Calculate the MSE loss over the entire dataset
double compute_loss(SimState *sim, const XorDataset *dataset);

// Compute the numerical gradient (dL / dW) for a specific layer using Finite Difference
Matrix compute_numerical_gradient(SimState *sim, int layer_idx, const XorDataset *dataset, double h);

#endif // GRADIENT_H
