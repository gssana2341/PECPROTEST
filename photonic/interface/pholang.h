// pholang.h — Clean public C API for the Photonic library and Python bindings
#ifndef PHOLANG_H
#define PHOLANG_H

#include "../sim/photonic_sim.h"

// Opaque structure wrapping the core SimState and metadata
typedef struct PhoNetwork PhoNetwork;

// Configuration structure for training parameters
typedef struct {
    int    epochs;
    double lr;
    int    batch_size;
} PhoTrainConfig;

// Result structure returning final training metrics
typedef struct {
    double final_loss;
    double final_accuracy;
} PhoResult;

// ─── Core Library Public API ────────────────────────────────────────

// 1. Create a network by parsing a .pho definition file
PhoNetwork* pho_network_load(const char* pho_file);

// 2. Train the network on a balanced class CSV dataset
PhoResult   pho_network_train(PhoNetwork* net,
                               const char* data_csv,
                               PhoTrainConfig cfg);

// 3. Predict the digit category for a raw single image input array
int         pho_network_predict(PhoNetwork* net,
                                 const double* input,
                                 int input_len);

// 4. Save and restore trained weight matrices
int         pho_network_save(PhoNetwork* net, const char* path);
PhoNetwork* pho_network_load_weights(const char* pho_file,
                                      const char* weights_path);

// 5. Model Compression & Quantization API
double      pho_network_compress(PhoNetwork* net, int dac_bits, double pruning_threshold);

// 6. Free network structure allocations
void        pho_network_free(PhoNetwork* net);

// 7. Thread-safe error reporting API
const char* pho_last_error(void);

#endif // PHOLANG_H
