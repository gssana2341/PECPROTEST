// kalman.c — Extended Kalman Filter (EKF) C implementation for MZI thermal drift compensation

#include "kalman.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void kalman_init(KalmanFilter *kf, double init_x, double init_P, double Q, double R) {
    if (!kf) return;
    kf->x = init_x;
    kf->P = init_P;
    kf->Q = Q;
    kf->R = R;
}

void kalman_predict(KalmanFilter *kf) {
    if (!kf) return;
    // Scalar Random Walk process update:
    // State remains constant (x_t = x_{t-1}), error covariance increases by process noise Q
    kf->P += kf->Q;
}

void kalman_update(KalmanFilter *kf, double target_phase, double measured_intensity) {
    if (!kf) return;

    // 1. Calculate prediction residual (innovation)
    // Non-linear observation model: h(x) = cos^2((target_phase + x) / 2)
    double phase_total = target_phase + kf->x;
    double predicted_intensity = cos(phase_total / 2.0) * cos(phase_total / 2.0);
    double y = measured_intensity - predicted_intensity;

    // 2. Compute Jacobian H_t = dh/dx
    // dh/dx = -cos((target_phase + x)/2) * sin((target_phase + x)/2) = -0.5 * sin(target_phase + x)
    double H = -0.5 * sin(phase_total);

    // 3. Epsilon Floor to avoid singularity / sensitivity loss near 0 and Pi
    double eps = 1e-4;
    if (fabs(H) < eps) {
        H = (H >= 0.0) ? eps : -eps;
    }

    // 4. Update state error covariance and Kalman Gain
    // S = H * P * H^T + R
    double S = H * kf->P * H + kf->R;
    
    // K = P * H^T * S^-1
    double K = (kf->P * H) / S;

    // 5. Update state estimate and error covariance
    kf->x += K * y;
    kf->P = (1.0 - K * H) * kf->P;
}

double kalman_compensate_voltage(double target_phase, double estimated_drift, double v_pi) {
    // Realized phase = V_set_phase + estimated_drift = target_phase
    // V_set_phase = target_phase - estimated_drift
    double adjusted_phase = target_phase - estimated_drift;

    // Handle 2*Pi periodicity and clamp to safe [0, 2*Pi] micro-heater boundaries
    while (adjusted_phase < 0.0) {
        adjusted_phase += 2.0 * M_PI;
    }
    while (adjusted_phase >= 2.0 * M_PI) {
        adjusted_phase -= 2.0 * M_PI;
    }

    // V = V_pi * sqrt(theta / Pi)
    double voltage = v_pi * sqrt(adjusted_phase / M_PI);
    return voltage;
}
