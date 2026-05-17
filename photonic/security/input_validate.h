// input_validate.h — input validation + sanitization
//
// ทุก input ต้องผ่าน validate ก่อนเสมอ
// ออกแบบสำหรับ safety-critical environments
#ifndef INPUT_VALIDATE_H
#define INPUT_VALIDATE_H

#include "../core/complex.h"
#include "../core/matrix.h"

typedef enum {
    VALIDATE_OK,
    VALIDATE_NULL_PTR,
    VALIDATE_OUT_OF_RANGE,
    VALIDATE_DIMENSION_MISMATCH,
    VALIDATE_NON_UNITARY,        // photonic gate ต้อง unitary
    VALIDATE_ENERGY_OVERFLOW,    // ป้องกัน energy blow-up
    VALIDATE_NAN_DETECTED,
    VALIDATE_INF_DETECTED
} ValidationResult;

// Validate a signal vector
ValidationResult validate_signal(const Complex *signal, int expected_dim);

// Validate a weight matrix
ValidationResult validate_weights(const Matrix *w, size_t expected_rows, size_t expected_cols);

// Validate that a matrix is unitary (W†W ≈ I)
ValidationResult validate_unitary(const Matrix *w, double epsilon);

// Validate a scalar parameter is within range
ValidationResult validate_param(double value, double min_val, double max_val);

// Get human-readable error message
const char *validate_error_string(ValidationResult result);

#endif // INPUT_VALIDATE_H
