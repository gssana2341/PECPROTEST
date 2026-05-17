#include "unitary_sgd.h"
#include "../core/complex.h"
#include <stdio.h>

void unitary_update_cayley(Matrix *W, const Matrix *grad, double lr) {
    if (!W || !W->data || !grad || !grad->data) return;
    
    size_t dim = W->rows;
    
    // 1. Project gradient to skew-Hermitian tangent space
    // G_skew = G * W^dag - W * G^dag
    Matrix W_dag = matrix_adjoint(W);
    Matrix G_dag = matrix_adjoint(grad);
    
    Matrix GWd = matrix_mul(grad, &W_dag);
    Matrix WGd = matrix_mul(W, &G_dag);
    
    // GWd - WGd
    Matrix G_skew = matrix_new(dim, dim);
    for (size_t i = 0; i < dim * dim; i++) {
        G_skew.data[i] = complex_sub(GWd.data[i], WGd.data[i]);
    }
    
    // 2. Compute Cayley transform terms
    // A = (lr/2) * G_skew
    Matrix A = matrix_copy(&G_skew);
    matrix_scale(&A, lr / 2.0);
    
    Matrix I = matrix_identity(dim);
    
    // Term 1: (I - A)
    Matrix I_minus_A = matrix_new(dim, dim);
    for (size_t i = 0; i < dim * dim; i++) {
        I_minus_A.data[i] = complex_sub(I.data[i], A.data[i]);
    }
    
    // Term 2: (I + A)
    Matrix I_plus_A = matrix_new(dim, dim);
    for (size_t i = 0; i < dim * dim; i++) {
        I_plus_A.data[i] = complex_add(I.data[i], A.data[i]);
    }
    
    // Inverse of (I + A)
    Matrix I_plus_A_inv;
    if (!matrix_inverse(&I_plus_A, &I_plus_A_inv)) {
        fprintf(stderr, "Cayley transform failed: (I+A) is singular.\n");
        // Clean up
        matrix_free(&W_dag); matrix_free(&G_dag);
        matrix_free(&GWd); matrix_free(&WGd); matrix_free(&G_skew);
        matrix_free(&A); matrix_free(&I);
        matrix_free(&I_minus_A); matrix_free(&I_plus_A);
        return;
    }
    
    // 3. Final update: W_new = (I - A) * (I + A)^-1 * W
    Matrix Cayley = matrix_mul(&I_minus_A, &I_plus_A_inv);
    Matrix W_new = matrix_mul(&Cayley, W);
    
    // Copy new weights back to W
    for (size_t i = 0; i < dim * dim; i++) {
        W->data[i] = W_new.data[i];
    }
    
    // Free everything
    matrix_free(&W_dag);
    matrix_free(&G_dag);
    matrix_free(&GWd);
    matrix_free(&WGd);
    matrix_free(&G_skew);
    matrix_free(&A);
    matrix_free(&I);
    matrix_free(&I_minus_A);
    matrix_free(&I_plus_A);
    matrix_free(&I_plus_A_inv);
    matrix_free(&Cayley);
    matrix_free(&W_new);
}
