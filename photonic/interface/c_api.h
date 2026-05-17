// c_api.h — Clean public C API for the Photonic library
//
// This is the ONLY header external users need to include.
// ทุกอย่างถูก namespace ด้วย prefix `pho_`
#ifndef C_API_H
#define C_API_H

#include "../core/complex.h"
#include "../core/matrix.h"
#include "../core/photonic.h"
#include "../core/activation.h"
#include "../core/loss.h"
#include "../core/memory.h"

// ─── Library Lifecycle ──────────────────────────────────────────────

// Initialize the photonic library (call once at startup)
int pho_init(void);

// Shutdown the photonic library (call once at exit)
void pho_shutdown(void);

// Get version string
const char *pho_version(void);

// ─── Layer Operations ───────────────────────────────────────────────

// Create a photonic layer with given dimensions
PhotonicLayer *pho_layer_create(size_t input_dim, size_t output_dim);

// Destroy a photonic layer
void pho_layer_destroy(PhotonicLayer *layer);

// Run forward pass
int pho_forward(const PhotonicLayer *layer, const Complex *input,
                Complex *output, size_t dim);

// ─── Training ───────────────────────────────────────────────────────

// Train using feedback alignment
int pho_train_fa(PhotonicLayer *layer, const double *target,
                 double learning_rate, int epochs);

// Train using multiplexed gradient descent
int pho_train_mgd(PhotonicLayer *layer, const double *target,
                  double learning_rate, int iterations);

#endif // C_API_H
