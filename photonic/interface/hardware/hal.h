// hal.h — Hardware Abstraction Layer for Silicon Photonics Chip
//
// Unified driver specifications to communicate with physical DAC/ADC control boards.
// Integrates an active simulation stub for safe off-chip code validation.

#ifndef HAL_H
#define HAL_H

#ifdef __cplusplus
extern "C" {
#endif

// ─── Core Hardware Abstraction Driver API ────────────────────────────

// Initialize chip interface communication (e.g. SPI/I2C/PCIe address ports)
// If config_str equals "EMULATION", routes all operations to the virtual simulation stub.
int hal_init_chip(const char *config_str);

// Write thermo-optic heater control voltages to physical DAC boards
int hal_write_voltages(const double *voltages, int count);

// Read activated optical intensities from photodetectors via ADC boards
int hal_read_photodetectors(double *readouts, int count);

// Safely close connection ports and reset control board DAC values to 0.0V
int hal_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif // HAL_H
