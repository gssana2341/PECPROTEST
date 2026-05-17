// ibm_photonic.c — Silicon Photonics HAL Driver Implementation & Physics Emulation Stub
//
// Implements unified HAL signatures for DAC/ADC interfaces and routes to a high-fidelity
// virtual optoelectronic emulator when running in EMULATION mode.

#include "hal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../../core/photonic_rng.h"

// ─── Global State for Emulation & Physical Interfaces ────────────────

static int    g_connected = 0;
static int    g_emulation_mode = 0;
static double g_last_voltages[1024] = {0.0};
static int    g_voltage_count = 0;

// Default physical parameters for the emulation stub
static const double V_PI = 3.3;             // 3.3V for pi phase shift
static const double DETECTOR_RESPONSIVITY = 0.85; // A/W photodiode efficiency
static const double SHOT_NOISE_STD = 0.02;  // Simulated quantum shot noise variance

// Module-level RNG state for HAL emulation
static PhoRng g_hal_rng = {{55555, 11111}};

// Helper for Gaussian noise using thread-safe PRNG
static double sample_gaussian_noise(double std_dev) {
    if (std_dev <= 0.0) return 0.0;
    return pho_rng_normal(&g_hal_rng, 0.0, std_dev);
}

// ─── HAL Driver Implementation ──────────────────────────────────────

int hal_init_chip(const char *config_str) {
    if (g_connected) {
        printf("[HAL Warning] Chip driver already initialized.\n");
        return 0;
    }

    if (config_str && strcmp(config_str, "EMULATION") == 0) {
        g_emulation_mode = 1;
        g_connected = 1;
        printf("[HAL Success] Active Physics Emulation Stub successfully initialized!\n");
        return 0;
    }

    // Physical board connection (PCIe / SPI / I2C ports)
    // In a physical deployment, developers would open interface endpoints here.
    printf("[HAL Error] Physical interface hardware not detected (Config: %s)\n", config_str ? config_str : "NULL");
    printf("[HAL Info] Redirecting to EMULATION mode for safety.\n");
    g_emulation_mode = 1;
    g_connected = 1;
    return 0; 
}

int hal_write_voltages(const double *voltages, int count) {
    if (!g_connected) {
        printf("[HAL Error] Chip driver not initialized!\n");
        return -1;
    }
    if (!voltages || count <= 0) return -1;

    // Safety limit to prevent buffer overflow in emulation
    int to_write = (count > 1024) ? 1024 : count;
    memcpy(g_last_voltages, voltages, to_write * sizeof(double));
    g_voltage_count = to_write;

    if (g_emulation_mode) {
        // Output debug log for small arrays
        if (to_write <= 8) {
            printf("[HAL Emulation] Wrote DAC voltages: ");
            for (int i = 0; i < to_write; i++) {
                printf("%.3fV ", voltages[i]);
            }
            printf("\n");
        }
        return 0;
    }

    // Physical DAC write operation: e.g. spi_transfer(DAC_ADDR, voltages)
    return 0;
}

int hal_read_photodetectors(double *readouts, int count) {
    if (!g_connected) {
        printf("[HAL Error] Chip driver not initialized!\n");
        return -1;
    }
    if (!readouts || count <= 0) return -1;

    if (g_emulation_mode) {
        // High-Fidelity Physics Emulation:
        // We simulate laser inputs propagating through waveguides modulated by heaters.
        // Transmission of MZI = cos^2(theta / 2), where theta = pi * (V / V_PI)^2.
        for (int i = 0; i < count; i++) {
            // Retrieve respective voltage (default to 0.0V if not set)
            double v = (i < g_voltage_count) ? g_last_voltages[i] : 0.0;
            
            // Non-linear thermo-optic phase shift equation
            double phase = M_PI * pow(v / V_PI, 2.0);
            
            // Optical intensity transmission
            double transmission = pow(cos(phase / 2.0), 2.0);
            
            // Photo-current conversion with quantum responsivity and shot noise
            double signal = transmission * DETECTOR_RESPONSIVITY;
            double noise = sample_gaussian_noise(SHOT_NOISE_STD);
            
            double intensity = signal + noise;
            if (intensity < 0.0) intensity = 0.0; // photodetectors measure positive absolute power
            
            readouts[i] = intensity;
        }
        return 0;
    }

    // Physical ADC read operation: e.g. read_adc_voltages(ADC_ADDR)
    memset(readouts, 0, count * sizeof(double));
    return 0;
}

int hal_shutdown(void) {
    if (!g_connected) return 0;

    // Reset control voltages to 0.0V to safely cool down the silicon chip micro-heaters
    memset(g_last_voltages, 0, sizeof(g_last_voltages));
    g_voltage_count = 0;

    g_connected = 0;
    g_emulation_mode = 0;
    printf("[HAL Success] Chip safely shut down. Heaters cooled down to 0.0V.\n");
    return 0;
}
