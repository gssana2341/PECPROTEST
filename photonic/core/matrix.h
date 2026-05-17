// matrix.h – basic matrix operations for photonic computing
#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include "complex.h"

typedef struct {
    size_t rows;
    size_t cols;
    Complex *data; // row-major order
} Matrix;

// Create a matrix (allocates memory)
Matrix matrix_new(size_t rows, size_t cols);
Matrix matrix_identity(size_t dim);

// Free matrix memory
void matrix_free(Matrix *m);
Matrix matrix_copy(const Matrix *m);

// Basic operations
Matrix matrix_mul(const Matrix *a, const Matrix *b);
Matrix matrix_transpose(const Matrix *m);
int matrix_is_unitary(const Matrix *m, double epsilon);

// Photonic-specific additions
Matrix matrix_adjoint(const Matrix *m);
void matrix_scale(Matrix *m, double scalar);
Matrix matrix_add(const Matrix *a, const Matrix *b);

// Inverse via Gauss-Jordan. Returns 1 if successful, 0 if singular.
int matrix_inverse(const Matrix *m, Matrix *out);
double matrix_norm(const Matrix *m);
void matrix_print(const Matrix *m, const char *name);

#endif // MATRIX_H
