#include "matrix.h"
#include "memory.h"
#include <math.h>
#include <stdio.h>
#include <stdint.h>

Matrix matrix_new(size_t rows, size_t cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    if (rows > 0 && cols > 0) {
        // Guard against integer overflow: rows * cols * sizeof(Complex)
        if (rows > SIZE_MAX / cols || (rows * cols) > SIZE_MAX / sizeof(Complex)) {
            fprintf(stderr, "[matrix_new] Overflow: %zu x %zu exceeds addressable memory\n", rows, cols);
            m.data = NULL;
            m.rows = 0;
            m.cols = 0;
            return m;
        }
        m.data = (Complex *)pho_alloc(rows * cols * sizeof(Complex), "Matrix.data");
        if (!m.data) { m.rows = m.cols = 0; return m; }
        for (size_t i = 0; i < rows * cols; i++) {
            m.data[i] = complex_new(0.0, 0.0);
        }
    } else {
        m.data = NULL;
    }
    return m;
}

Matrix matrix_identity(size_t dim) {
    Matrix m = matrix_new(dim, dim);
    if (!m.data) return m;
    for (size_t i = 0; i < dim; i++) {
        m.data[i * dim + i] = complex_new(1.0, 0.0);
    }
    return m;
}

void matrix_free(Matrix *m) {
    if (m->data) {
        pho_free(m->data, "Matrix.data");
        m->data = NULL;
    }
    m->rows = 0;
    m->cols = 0;
}

Matrix matrix_mul(const Matrix *a, const Matrix *b) {
    if (a->cols != b->rows) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication\n");
        return matrix_new(0, 0); // returns Matrix with data = NULL
    }
    
    Matrix result = matrix_new(a->rows, b->cols);
    
    // Transpose B for cache-friendly access (O(n^3) optimized)
    Matrix b_t = matrix_transpose(b);
    
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b_t.rows; j++) { // b_t.rows == b.cols
            Complex sum = complex_new(0.0, 0.0);
            for (size_t k = 0; k < a->cols; k++) {
                // Now both a and b_t are accessed sequentially in the inner loop
                sum = complex_add(sum, complex_mul(a->data[i * a->cols + k], b_t.data[j * b_t.cols + k]));
            }
            result.data[i * result.cols + j] = sum;
        }
    }
    
    matrix_free(&b_t);
    return result;
}

Matrix matrix_transpose(const Matrix *m) {
    Matrix result = matrix_new(m->cols, m->rows);
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            result.data[j * result.cols + i] = m->data[i * m->cols + j];
        }
    }
    return result;
}

Matrix matrix_adjoint(const Matrix *m) {
    Matrix result = matrix_new(m->cols, m->rows);
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            result.data[j * result.cols + i] = complex_conj(m->data[i * m->cols + j]);
        }
    }
    return result;
}

int matrix_is_unitary(const Matrix *m, double epsilon) {
    if (m->rows != m->cols) return 0;
    Matrix adjoint = matrix_adjoint(m);
    // Check W^dag * W = I
    Matrix wt_w = matrix_mul(&adjoint, m);
    int is_unitary = 1;
    
    // Fix: Break out of both loops efficiently using flag condition
    for (size_t i = 0; i < wt_w.rows && is_unitary; i++) {
        for (size_t j = 0; j < wt_w.cols && is_unitary; j++) {
            double expected_real = (i == j) ? 1.0 : 0.0;
            Complex val = wt_w.data[i * wt_w.cols + j];
            if (fabs(val.real - expected_real) > epsilon || fabs(val.imag) > epsilon) {
                is_unitary = 0;
            }
        }
    }
    
    matrix_free(&adjoint);
    matrix_free(&wt_w);
    return is_unitary;
}

void matrix_scale(Matrix *m, double scalar) {
    if (!m || !m->data) return;
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        m->data[i].real *= scalar;
        m->data[i].imag *= scalar;
    }
}

Matrix matrix_add(const Matrix *a, const Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Matrix dimensions do not match for addition\n");
        return matrix_new(0, 0);
    }
    Matrix result = matrix_new(a->rows, a->cols);
    for (size_t i = 0; i < a->rows * a->cols; i++) {
        result.data[i] = complex_add(a->data[i], b->data[i]);
    }
    return result;
}

double matrix_norm(const Matrix *m) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        double r = m->data[i].real;
        double im = m->data[i].imag;
        sum_sq += r * r + im * im;
    }
    return sqrt(sum_sq);
}

void matrix_print(const Matrix *m, const char *name) {
    printf("Matrix %s (%zu x %zu):\n", name ? name : "", m->rows, m->cols);
    for (size_t i = 0; i < m->rows; i++) {
        printf("  [ ");
        for (size_t j = 0; j < m->cols; j++) {
            Complex val = m->data[i * m->cols + j];
            printf("%.4f%+.4fi ", val.real, val.imag);
        }
        printf("]\n");
    }
}

Matrix matrix_copy(const Matrix *m) {
    Matrix copy = matrix_new(m->rows, m->cols);
    if (copy.data && m->data) {
        for (size_t i = 0; i < m->rows * m->cols; i++) {
            copy.data[i] = m->data[i];
        }
    }
    return copy;
}

int matrix_inverse(const Matrix *m, Matrix *out) {
    if (!m || !m->data || !out || m->rows != m->cols || m->rows == 0) return 0;
    
    size_t n = m->rows;
    *out = matrix_new(n, n);
    if (!out->data) return 0;
    
    // Create augmented matrix [A | I]
    Matrix aug = matrix_new(n, 2 * n);
    if (!aug.data) {
        matrix_free(out);
        return 0;
    }
    
    // Initialize [A | I]
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            aug.data[i * (2 * n) + j] = m->data[i * n + j];
            if (i == j) {
                aug.data[i * (2 * n) + (n + j)] = complex_new(1.0, 0.0);
            } else {
                aug.data[i * (2 * n) + (n + j)] = complex_new(0.0, 0.0);
            }
        }
    }
    
    double epsilon = 1e-12;
    
    // Gauss-Jordan Elimination with partial pivoting
    for (size_t i = 0; i < n; i++) {
        // Partial pivoting based on complex norm
        size_t max_row = i;
        double max_norm = complex_norm(aug.data[i * (2 * n) + i]);
        
        for (size_t k = i + 1; k < n; k++) {
            double current_norm = complex_norm(aug.data[k * (2 * n) + i]);
            if (current_norm > max_norm) {
                max_norm = current_norm;
                max_row = k;
            }
        }
        
        if (max_norm < epsilon) {
            matrix_free(&aug);
            matrix_free(out);
            return 0; // Singular matrix
        }
        
        // Swap rows if a larger pivot is found
        if (max_row != i) {
            for (size_t j = 0; j < 2 * n; j++) {
                Complex temp = aug.data[i * (2 * n) + j];
                aug.data[i * (2 * n) + j] = aug.data[max_row * (2 * n) + j];
                aug.data[max_row * (2 * n) + j] = temp;
            }
        }
        
        // Normalize pivot row
        Complex pivot = aug.data[i * (2 * n) + i];
        for (size_t j = 0; j < 2 * n; j++) {
            aug.data[i * (2 * n) + j] = complex_div(aug.data[i * (2 * n) + j], pivot);
        }
        
        // Eliminate all other rows
        for (size_t k = 0; k < n; k++) {
            if (k != i) {
                Complex factor = aug.data[k * (2 * n) + i];
                for (size_t j = 0; j < 2 * n; j++) {
                    Complex sub = complex_mul(factor, aug.data[i * (2 * n) + j]);
                    aug.data[k * (2 * n) + j] = complex_sub(aug.data[k * (2 * n) + j], sub);
                }
            }
        }
    }
    
    // Extract the right half [I | A^-1]
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            out->data[i * n + j] = aug.data[i * (2 * n) + (n + j)];
        }
    }
    
    matrix_free(&aug);
    return 1;
}
