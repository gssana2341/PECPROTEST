#include "gradient_analytic.h"
#include "../core/activation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Backpropagate gradient through a single photonic layer
void photonic_layer_backward(
    const Matrix   *W,            // weights of this layer
    const Complex  *input,        // input signals of this layer
    const Complex  *output_grad,  // output gradients (dL/dout)
    double          kerr_gamma,   // Kerr nonlinearity
    Matrix         *weight_grad,  // output: weight gradient (dL/dW)
    Complex        *input_grad,   // output: input gradient (dL/din)
    int             dim
) {
    Complex s[dim];
    Complex y[dim];
    Complex delta_s[dim];
    
    // 1. Compute intermediate sum and forward activation
    for (int i = 0; i < dim; i++) {
        Complex sum = complex_new(0.0, 0.0);
        for (int j = 0; j < dim; j++) {
            sum = complex_add(sum, complex_mul(W->data[i * dim + j], input[j]));
        }
        s[i] = sum;
        
        if (kerr_gamma != 0.0) {
            y[i] = activation_kerr(s[i], kerr_gamma);
        } else {
            y[i] = s[i];
        }
        
        // 2. Backprop through Kerr activation to get delta_s_i
        if (kerr_gamma != 0.0) {
            double intensity = complex_norm_sq(s[i]);
            double phase = kerr_gamma * intensity;
            Complex phase_factor_inv = complex_new(cos(-phase), sin(-phase));
            
            Complex term1 = complex_mul(output_grad[i], phase_factor_inv);
            Complex y_conj = complex_new(y[i].real, -y[i].imag);
            Complex out_y = complex_mul(output_grad[i], y_conj);
            double im_part = out_y.imag; // Im(out * y_conj)
            
            Complex term2 = complex_new(2.0 * kerr_gamma * s[i].real * im_part,
                                        2.0 * kerr_gamma * s[i].imag * im_part);
            
            delta_s[i] = complex_add(term1, term2);
        } else {
            delta_s[i] = output_grad[i];
        }
    }
    
    // 3. Compute weight gradient: dL/dW_ij = delta_s_i * input_j^*
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            Complex input_conj = complex_new(input[j].real, -input[j].imag);
            weight_grad->data[i * dim + j] = complex_mul(delta_s[i], input_conj);
        }
    }
    
    // 4. Compute input gradient: dL/dinput_j = sum_i W_ij^* * delta_s_i
    for (int j = 0; j < dim; j++) {
        Complex sum = complex_new(0.0, 0.0);
        for (int i = 0; i < dim; i++) {
            Complex w_conj = complex_new(W->data[i * dim + j].real, -W->data[i * dim + j].imag);
            sum = complex_add(sum, complex_mul(w_conj, delta_s[i]));
        }
    input_grad[j] = sum;
    }
}

// Skew-hermitian matrix projection for Unitary manifold
void compute_skew_hermitian(
    const Matrix *G,      // dL/dW
    const Matrix *W,      // current weights
    Matrix       *G_skew  // output: G * W^dag - W * G^dag
) {
    size_t dim = W->rows;
    Matrix W_dag = matrix_adjoint(W);
    Matrix G_dag = matrix_adjoint(G);
    
    Matrix GWd = matrix_mul(G, &W_dag);
    Matrix WGd = matrix_mul(W, &G_dag);
    
    for (size_t i = 0; i < dim * dim; i++) {
        G_skew->data[i] = complex_sub(GWd.data[i], WGd.data[i]);
    }
    
    matrix_free(&W_dag);
    matrix_free(&G_dag);
    matrix_free(&GWd);
    matrix_free(&WGd);
}

// Cayley update
void cayley_update(
    Matrix       *W,
    const Matrix *G_skew,
    double        lr
) {
    size_t dim = W->rows;
    
    // A = (lr/2) * G_skew
    Matrix A = matrix_copy(G_skew);
    matrix_scale(&A, lr / 2.0);
    
    Matrix I = matrix_identity(dim);
    
    // (I - A)
    Matrix I_minus_A = matrix_new(dim, dim);
    for (size_t i = 0; i < dim * dim; i++) {
        I_minus_A.data[i] = complex_sub(I.data[i], A.data[i]);
    }
    
    // (I + A)
    Matrix I_plus_A = matrix_new(dim, dim);
    for (size_t i = 0; i < dim * dim; i++) {
        I_plus_A.data[i] = complex_add(I.data[i], A.data[i]);
    }
    
    // Inverse of (I + A)
    Matrix I_plus_A_inv;
    if (!matrix_inverse(&I_plus_A, &I_plus_A_inv)) {
        fprintf(stderr, "Cayley update failed: (I+A) is singular.\n");
        matrix_free(&A);
        matrix_free(&I);
        matrix_free(&I_minus_A);
        matrix_free(&I_plus_A);
        return;
    }
    
    // W_new = (I - A) * (I + A)^-1 * W
    Matrix Cayley = matrix_mul(&I_minus_A, &I_plus_A_inv);
    Matrix W_new = matrix_mul(&Cayley, W);
    
    // Copy W_new back
    for (size_t i = 0; i < dim * dim; i++) {
        W->data[i] = W_new.data[i];
    }
    
    matrix_free(&A);
    matrix_free(&I);
    matrix_free(&I_minus_A);
    matrix_free(&I_plus_A);
    matrix_free(&I_plus_A_inv);
    matrix_free(&Cayley);
    matrix_free(&W_new);
}

// Gradient of optical cross-entropy softmax with TIA gain
void loss_cross_entropy_softmax_optical_grad(
    const Complex *output,
    double detector_gain,
    const double *target,
    size_t length,
    Complex *grad_out
) {
    double max_val = -1e9;
    double vals[length];
    
    for (size_t i = 0; i < length; i++) {
        double intensity = complex_norm_sq(output[i]);
        vals[i] = intensity * detector_gain;
        if (vals[i] > max_val) {
            max_val = vals[i];
        }
    }
    
    double sum_exp = 0.0;
    for (size_t i = 0; i < length; i++) {
        double e = exp(vals[i] - max_val);
        vals[i] = e;
        sum_exp += e;
    }
    
    for (size_t i = 0; i < length; i++) {
        double prob = vals[i] / sum_exp;
        double diff = prob - target[i];
        double scale = 2.0 * detector_gain * diff;
        grad_out[i] = complex_new(output[i].real * scale, output[i].imag * scale);
    }
}
