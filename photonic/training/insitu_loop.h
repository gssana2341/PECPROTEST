// insitu_loop.h — In-Situ (Hardware-in-the-Loop) Optical Training Loop
//
// Performs hybrid on-chip optical propagation and off-chip CPU gradient updates.
// Operates seamlessly over either physical control boards or active HAL emulator.

#ifndef INSITU_LOOP_H
#define INSITU_LOOP_H

#include "photonic_sim.h"
#include "calibration.h"

// Hardware-in-the-Loop training configurations
typedef struct {
    int epochs;
    int batch_size;
    double lr;
    const char *data_csv;
    CalibrationParams cal_params;
} InSituConfig;

// Run hybrid on-chip dynamic training utilizing HAL write/read interfaces
int run_insitu_training(SimState *sim, const InSituConfig *cfg);

#endif // INSITU_LOOP_H
