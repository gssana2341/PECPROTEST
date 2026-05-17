#ifndef POOLING_H
#define POOLING_H

// Performs 28x28 -> 8x8 average pooling simulating an optical multimode lens
// Input array must have 784 elements, output must have 64 elements
void optical_lens_pool_28_to_8(const double *input_784, double *output_64);
void optical_lens_pool_28_to_4(const double *input_784, double *output_16);

#endif // POOLING_H
