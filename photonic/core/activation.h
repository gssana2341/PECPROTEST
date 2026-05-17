// activation.h – optical activation functions
#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stddef.h>
#include "complex.h"

// Sigmoid optical activation (applied element‑wise)
Complex activation_sigmoid_optical(Complex z);

// Gradient for backprop / feedback alignment
Complex activation_sigmoid_optical_grad(Complex z);

// Kerr effect (z * e^(i * gamma * |z|^2))
Complex activation_kerr(Complex z, double gamma);

// Softmax optical activation for a vector of Complex numbers
void activation_softmax_optical(const Complex *input, Complex *output, size_t length);

#endif // ACTIVATION_H
