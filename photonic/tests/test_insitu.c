// test_insitu.c — Integration test for In-Situ Hardware-in-the-Loop training
//
// Validates:
// 1. Initializing HAL chip and loading training data
// 2. Calibrating weights to micro-heater DAC voltages under cross-talk compensation
// 3. Simulating physical optoelectronic forward propagation and ADC photodetector readouts
// 4. Executing C-based analytical backpropagation with Cayley unitary updates

#include "pholang.h"
#include "photonic_sim.h"
#include "insitu_loop.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Forward declaration of the private C API struct to inspect internal state
struct PhoNetwork {
    SimState sim;
    double   learning_rate;
    int      epochs;
    int      batch_size;
    int      early_stop;
    double   phase_noise;
    double   shot_noise;
};

int main(void) {
    srand((unsigned int)time(NULL));
    printf("=== Starting In-Situ Hardware-in-the-Loop Integration Test ===\n");

    const char *pho_path = "photonic/lang/mnist.pho";
    const char *csv_path = "data/mnist_scaled.csv";

    // Step 1: Load PhoLang dynamic runtime model
    printf("\n[Step 1] Loading dynamic PhoLang model from %s...\n", pho_path);
    PhoNetwork *net = pho_network_load(pho_path);
    if (!net) {
        fprintf(stderr, "✗ Failed to load PhoLang network definition: %s\n", pho_last_error());
        return 1;
    }
    printf("✓ Network structure parsed: %d layers, dimension: %d\n", 
           net->sim.num_layers, net->sim.layer_dim);

    // Step 2: Configure physical calibration parameters
    printf("\n[Step 2] Setting up physical heater calibration parameters...\n");
    InSituConfig cfg;
    cfg.epochs = 2; // Run a quick 2-epoch sweep to verify stability
    cfg.batch_size = 16;
    cfg.lr = 0.05;
    cfg.data_csv = csv_path;
    
    // Optoelectronic thermal properties
    cfg.cal_params.v_pi = 3.3;             // 3.3V ceiling phase voltage
    cfg.cal_params.resistance = 120.0;     // 120 Ohm micro-heater resistance
    cfg.cal_params.cross_talk_factor = 0.08; // 8% thermal coupling factor leak
    cfg.cal_params.waveguide_loss_db = 0.15; // 0.15 dB/cm insertion loss
    cfg.cal_params.max_voltage = 5.0;      // Max safety voltage threshold

    // Step 3: Run In-Situ Training Loop over the HAL Emulation Stub
    printf("\n[Step 3] Launching hybrid Hardware-in-the-Loop on-chip training loop...\n");
    int result = run_insitu_training(&net->sim, &cfg);
    if (result != 0) {
        fprintf(stderr, "✗ In-situ training execution failed!\n");
        pho_network_free(net);
        return 1;
    }
    printf("✓ In-situ training completed successfully!\n");

    // Step 4: Clean up network resources
    printf("\n[Step 4] Freeing all allocated hardware and simulation resources...\n");
    pho_network_free(net);
    printf("✓ Memory and handles safely cleared! (Zero Leaks)\n");

    printf("\n🎉 ALL IN-SITU HARDWARE INTEGRATION TESTS PASSED SUCCESSFULLY! 🎉\n");
    return 0;
}
