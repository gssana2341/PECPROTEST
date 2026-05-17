// runtime.h — PhoLang runtime (inline C)
//
// Provides runtime support functions that PhoLang-generated C code
// calls at execution time. Designed to be lightweight and inlineable.
#ifndef PHOLANG_RUNTIME_H
#define PHOLANG_RUNTIME_H

#include "../core/complex.h"
#include "../core/matrix.h"
#include "../core/photonic.h"
#include "../core/activation.h"
#include "../core/loss.h"
#include "../core/memory.h"

// ─── Signal Type (PhoLang `signal[N]`) ──────────────────────────────

typedef struct {
    Complex *data;
    size_t   length;
} PhoSignal;

// Create a signal of given length
static inline PhoSignal pho_signal_new(size_t length) {
    PhoSignal s;
    s.length = length;
    s.data   = (Complex *)pho_alloc(length * sizeof(Complex), "PhoSignal");
    return s;
}

// Free a signal
static inline void pho_signal_free(PhoSignal *s) {
    if (s->data) {
        pho_free(s->data, "PhoSignal");
        s->data   = NULL;
        s->length = 0;
    }
}

// ─── Runtime Helpers ────────────────────────────────────────────────

// interfere: apply matrix transformation to signal (optical interference)
static inline void pho_interfere(PhoSignal *signal, const Matrix *weights) {
    (void)signal;
    (void)weights;
    // TODO: implement matrix-signal multiplication
}

// activate: apply optical activation function to signal
static inline void pho_activate_sigmoid(PhoSignal *signal) {
    for (size_t i = 0; i < signal->length; i++) {
        signal->data[i] = activation_sigmoid_optical(signal->data[i]);
    }
}

#endif // PHOLANG_RUNTIME_H
