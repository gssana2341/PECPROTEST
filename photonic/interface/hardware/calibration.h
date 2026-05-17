// calibration.h — MZI Heater Voltage Calibration Engine
//
// Bridges abstract complex weights to physical thermo-optic control voltages
// incorporating physical constants, thermal cross-talk compensation, and loss balancing.

#ifndef CALIBRATION_H
#define CALIBRATION_H

#include "matrix.h"

// Thermo-optic heater calibration parameters
typedef struct {
    double v_pi;              // Voltage required for a pi phase shift (Volts)
    double resistance;        // Micro-heater electrical resistance (Ohms)
    double cross_talk_factor; // Thermal coupling factor between adjacent heaters (0.0 to 0.2)
    double waveguide_loss_db; // Attenuation coefficient of the waveguide (dB/cm)
    double max_voltage;       // Hardware ceiling voltage to prevent heater burnout (e.g. 5.0V)
} CalibrationParams;

// Perform direct phase extraction and convert to control voltages.
// Uses inverse thermal coupling matrix to compensate for thermal leakages.
int calibrate_direct_phases_to_voltages(
    const Matrix *W,
    const CalibrationParams *params,
    double *out_voltages
);

// Decomposes a unitary weight matrix W into Clements MZI mesh phase shifts (theta, phi)
// and calibrates them to their respective heater voltages with cross-talk compensation.
int calibrate_unitary_to_mzi_voltages(
    const Matrix *W,
    const CalibrationParams *params,
    double *out_theta_voltages,
    double *out_phi_voltages
);

#endif // CALIBRATION_H
