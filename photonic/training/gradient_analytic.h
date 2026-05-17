#ifndef GRADIENT_ANALYTIC_H
#define GRADIENT_ANALYTIC_H

#include "../core/complex.h"
#include "../core/matrix.h"
#include "../sim/photonic_sim.h"

// Backpropagate gradient through a single photonic layer
// Computes dL/dW (weight_grad) and dL/din (input_grad)
void photonic_layer_backward(
    const Matrix   *W,            // weights of this layer
    const Complex  *input,        // input signals of this layer
    const Complex  *output_grad,  // output gradients (dL/dout)
    double          kerr_gamma,   // Kerr nonlinearity
    Matrix         *weight_grad,  // output: weight gradient (dL/dW)
    Complex        *input_grad,   // output: input gradient (dL/din)
    int             dim
);

// Skew-hermitian matrix projection for Unitary manifold
void compute_skew_hermitian(
    const Matrix *G,      // dL/dW
    const Matrix *W,      // current weights
    Matrix       *G_skew  // output: G * W^dag - W * G^dag
);

// Cayley update
void cayley_update(
    Matrix       *W,
    const Matrix *G_skew,
    double        lr
);

// Gradient of optical cross-entropy softmax with TIA gain
void loss_cross_entropy_softmax_optical_grad(
    const Complex *output,
    double detector_gain,
    const double *target,
    size_t length,
    Complex *grad_out
);

#endif // GRADIENT_ANALYTIC_H
