// loss.h — loss functions for photonic neural networks
#ifndef LOSS_H
#define LOSS_H

#include <stddef.h>
#include "complex.h"

// Mean Squared Error
// Returns scalar loss value
double loss_mse(const double *predicted, const double *target, size_t length);

// MSE gradient (dL/d_predicted)
void loss_mse_grad(const double *predicted, const double *target, double *grad_out, size_t length);

// Cross-Entropy Loss
double loss_cross_entropy(const double *predicted, const double *target, size_t length);

// Cross-Entropy gradient
void loss_cross_entropy_grad(const double *predicted, const double *target, double *grad_out, size_t length);

// Cross-Entropy Loss with Softmax
double loss_cross_entropy_softmax(const double *predicted, const double *target, size_t length);

// Cross-Entropy Loss with Softmax gradient
void loss_cross_entropy_softmax_grad(const double *predicted, const double *target, double *grad_out, size_t length);

// Calculate Softmax probabilities directly from Complex output intensities,
// applying detector_gain, and then calculate Cross Entropy loss.
double loss_cross_entropy_softmax_optical(const Complex *output, 
                                          double detector_gain, 
                                          const double *target, 
                                          size_t length, 
                                          double *probabilities_out);

#endif // LOSS_H
