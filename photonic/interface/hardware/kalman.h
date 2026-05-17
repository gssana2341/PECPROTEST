// kalman.h — Extended Kalman Filter (EKF) for Silicon Photonics Thermal Drift Compensation
//
// Designed to track and compensate for real-time waveguide phase fluctuations
// over dynamic Mach-Zehnder Interferometers (MZIs).

#ifndef PHOTONIC_KALMAN_H
#define PHOTONIC_KALMAN_H

#ifdef __cplusplus
extern "C" {
#endif

// Scalar Kalman Filter structure for tracking localized 1D heater phase drift
typedef struct {
    double x; // State estimate: phase drift (in radians)
    double P; // Estimation error covariance
    double Q; // Process noise covariance (thermal walk drift variance)
    double R; // Measurement noise covariance (photodetector/ADC noise variance)
} KalmanFilter;

/**
 * Initialize the Extended Kalman Filter parameters
 */
void kalman_init(KalmanFilter *kf, double init_x, double init_P, double Q, double R);

/**
 * Predict step: updates state covariance based on process noise
 */
void kalman_predict(KalmanFilter *kf);

/**
 * Extended Kalman Update step: updates estimated drift based on raw optical readout
 * non-linear observation: z = cos^2((target_phase + drift) / 2)
 */
void kalman_update(KalmanFilter *kf, double target_phase, double measured_intensity);

/**
 * Compute the corrected control voltage required to achieve target_phase
 * under active estimated drift.
 */
double kalman_compensate_voltage(double target_phase, double estimated_drift, double v_pi);

#ifdef __cplusplus
}
#endif

#endif // PHOTONIC_KALMAN_H
