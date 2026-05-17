// test_kalman_drift.c — Integration test for EKF MZI thermal drift correction

#include "kalman.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Standard Box-Muller Gaussian Noise generator
double rand_gaussian(double mu, double sigma) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    if (u1 < 1e-9) u1 = 1e-9; // Avoid log(0)
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mu + z * sigma;
}

int main(void) {
    srand(42); // Seed for deterministic dynamic tests
    printf("=== Starting Extended Kalman Filter (EKF) Waveguide Drift Test ===\n");

    // Initialize EKF parameters
    KalmanFilter kf;
    double init_x = 0.0;    // Initial drift estimate
    double init_P = 0.5;    // High initial uncertainty
    double Q = 0.0001;      // Process noise covariance (thermal walk drift variance)
    double R = 0.0004;      // Measurement noise covariance (optical shot noise variance)

    kalman_init(&kf, init_x, init_P, Q, R);

    // Simulation configuration
    int timesteps = 100;
    double target_phase = 0.5 * M_PI; // We target a steady 90-degree phase shift (Quadrant 1 boundary)
    double v_pi = 3.3;                // 3.3V V_pi heater constant

    double actual_drift = 0.15; // Setup an initial real drift of 0.15 rad (about 8.6 degrees)
    double tracking_error_sum = 0.0;

    printf("\nParameters: Target Phase = %.4f rad, V_pi = %.1fV\n", target_phase, v_pi);
    printf("Initializing Real Drift: %.4f rad, EKF Estimate: %.4f rad\n", actual_drift, kf.x);
    printf("--------------------------------------------------------------------------------\n");
    printf("%-5s | %-12s | %-12s | %-12s | %-12s | %-12s\n", 
           "Step", "Real Drift", "Est. Drift", "Tracking Err", "Meas. Power", "Comp. Voltage");
    printf("--------------------------------------------------------------------------------\n");

    for (int t = 1; t <= timesteps; t++) {
        // 1. Physical Process Update: Waveguide thermal drift undergoes a random walk
        double step_drift_noise = rand_gaussian(0.0, sqrt(Q));
        actual_drift += step_drift_noise;

        // 2. Kalman Prediction
        kalman_predict(&kf);

        // 3. Simulate Physical MZI Readout (Observation including shot noise)
        // Light intensity transmission T = cos^2(theta_real / 2)
        double real_total_phase = target_phase + actual_drift;
        double optical_intensity = cos(real_total_phase / 2.0) * cos(real_total_phase / 2.0);
        double measurement_noise = rand_gaussian(0.0, sqrt(R));
        double z = optical_intensity + measurement_noise;

        // Clamp simulated intensity to safe physical bounds [0.0, 1.0]
        if (z < 0.0) z = 0.0;
        if (z > 1.0) z = 1.0;

        // 4. Extended Kalman Filter Update
        kalman_update(&kf, target_phase, z);

        // 5. Closed-loop Heater Voltage Compensation Calculation
        double compensated_voltage = kalman_compensate_voltage(target_phase, kf.x, v_pi);

        double current_err = fabs(actual_drift - kf.x);
        tracking_error_sum += current_err;

        if (t <= 10 || t % 10 == 0) {
            printf("%-5d | %-12.6f | %-12.6f | %-12.6f | %-12.6f | %-12.6f\n", 
                   t, actual_drift, kf.x, current_err, z, compensated_voltage);
        }
    }
    printf("--------------------------------------------------------------------------------\n");

    double average_tracking_error = tracking_error_sum / timesteps;
    printf("\n[Summary] Average Phase Tracking Error over %d steps: %.6f rad\n", timesteps, average_tracking_error);

    // Target tracking threshold is strictly less than 0.02 radians
    if (average_tracking_error < 0.02) {
        printf("🎉 SUCCESS! EKF tracking error %.6f rad is strictly below the 0.02 rad target! 🎉\n", average_tracking_error);
        return 0;
    } else {
        printf("✗ FAILURE! EKF tracking error %.6f rad exceeded the 0.02 rad target limit.\n", average_tracking_error);
        return 1;
    }
}
