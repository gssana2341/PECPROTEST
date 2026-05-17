#ifndef UNITARY_SGD_H
#define UNITARY_SGD_H

#include "../core/matrix.h"
#include "../sim/photonic_sim.h"

typedef struct {
    double learning_rate;
    int    max_epochs;
    double target_mse;
    int    log_every;
} TrainConfig;

// Perform Riemannian Gradient Descent update using Cayley Transform
// W_new = (I - (lr/2)*G_skew) * (I + (lr/2)*G_skew)^-1 * W
// This maintains the unitarity of the matrix.
void unitary_update_cayley(Matrix *W, const Matrix *grad, double lr);

#endif // UNITARY_SGD_H
