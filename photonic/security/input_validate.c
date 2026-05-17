// input_validate.c — Input validation implementation for safety-critical environments
//
// Implements all validation functions declared in input_validate.h.
// Every input to the photonic engine must pass through these checks.

#include "input_validate.h"
#include <math.h>
#include <stddef.h>

// Validate a signal vector for NULL, dimension, NaN, and Inf
ValidationResult validate_signal(const Complex *signal, int expected_dim) {
    if (!signal) return VALIDATE_NULL_PTR;
    if (expected_dim <= 0) return VALIDATE_OUT_OF_RANGE;

    for (int i = 0; i < expected_dim; i++) {
        if (isnan(signal[i].real) || isnan(signal[i].imag)) {
            return VALIDATE_NAN_DETECTED;
        }
        if (isinf(signal[i].real) || isinf(signal[i].imag)) {
            return VALIDATE_INF_DETECTED;
        }
    }

    // Check total energy (sum of |z|^2) to prevent blow-up
    double total_energy = 0.0;
    for (int i = 0; i < expected_dim; i++) {
        total_energy += signal[i].real * signal[i].real + signal[i].imag * signal[i].imag;
    }
    if (total_energy > 1e12) {
        return VALIDATE_ENERGY_OVERFLOW;
    }

    return VALIDATE_OK;
}

// Validate a weight matrix for NULL, dimensions, NaN, and Inf
ValidationResult validate_weights(const Matrix *w, size_t expected_rows, size_t expected_cols) {
    if (!w || !w->data) return VALIDATE_NULL_PTR;
    if (w->rows != expected_rows || w->cols != expected_cols) {
        return VALIDATE_DIMENSION_MISMATCH;
    }

    size_t total = w->rows * w->cols;
    for (size_t i = 0; i < total; i++) {
        if (isnan(w->data[i].real) || isnan(w->data[i].imag)) {
            return VALIDATE_NAN_DETECTED;
        }
        if (isinf(w->data[i].real) || isinf(w->data[i].imag)) {
            return VALIDATE_INF_DETECTED;
        }
    }

    return VALIDATE_OK;
}

// Validate that a matrix is unitary: ||W†W - I||_F < epsilon
ValidationResult validate_unitary(const Matrix *w, double epsilon) {
    if (!w || !w->data) return VALIDATE_NULL_PTR;
    if (w->rows != w->cols || w->rows == 0) return VALIDATE_DIMENSION_MISMATCH;

    size_t N = w->rows;
    double frob_sq = 0.0;

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            // Compute (W†W)_{ij} = sum_k conj(W_{ki}) * W_{kj}
            double re = 0.0, im = 0.0;
            for (size_t k = 0; k < N; k++) {
                Complex wki = w->data[k * N + i];
                Complex wkj = w->data[k * N + j];
                // conj(wki) * wkj
                re += wki.real * wkj.real + wki.imag * wkj.imag;
                im += wki.real * wkj.imag - wki.imag * wkj.real;
            }
            // Subtract identity
            if (i == j) re -= 1.0;

            frob_sq += re * re + im * im;
        }
    }

    if (sqrt(frob_sq) > epsilon) {
        return VALIDATE_NON_UNITARY;
    }

    return VALIDATE_OK;
}

// Validate a scalar parameter is within [min_val, max_val]
ValidationResult validate_param(double value, double min_val, double max_val) {
    if (isnan(value)) return VALIDATE_NAN_DETECTED;
    if (isinf(value)) return VALIDATE_INF_DETECTED;
    if (value < min_val || value > max_val) return VALIDATE_OUT_OF_RANGE;
    return VALIDATE_OK;
}

// Get human-readable error message
const char *validate_error_string(ValidationResult result) {
    switch (result) {
        case VALIDATE_OK:                 return "Validation passed";
        case VALIDATE_NULL_PTR:           return "Null pointer detected";
        case VALIDATE_OUT_OF_RANGE:       return "Parameter out of valid range";
        case VALIDATE_DIMENSION_MISMATCH: return "Matrix/vector dimension mismatch";
        case VALIDATE_NON_UNITARY:        return "Matrix is not unitary (W†W ≠ I)";
        case VALIDATE_ENERGY_OVERFLOW:    return "Signal energy exceeds safe threshold";
        case VALIDATE_NAN_DETECTED:       return "NaN value detected in input";
        case VALIDATE_INF_DETECTED:       return "Infinity value detected in input";
    }
    return "Unknown validation error";
}
