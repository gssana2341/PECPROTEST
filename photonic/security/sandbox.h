// sandbox.h — execution sandbox for photonic simulations
//
// จำกัด resource ที่ simulation ใช้ได้ เพื่อความปลอดภัย
// ใช้ใน embedded / edge / production environments
#ifndef SANDBOX_H
#define SANDBOX_H

#include <stddef.h>

typedef struct {
    size_t max_memory_mb;     // จำกัด RAM (MB)
    int    max_qubits;        // จำกัดขนาด simulation
    int    max_iterations;    // ป้องกัน infinite loop
    int    allow_file_io;     // ควบคุม file access (0 = disabled)
} SandboxConfig;

// Create a default sandbox config (conservative limits)
SandboxConfig sandbox_default(void);

// Initialize sandbox enforcement
int sandbox_init(const SandboxConfig *cfg);

// Check if an allocation is within sandbox limits
int sandbox_check_alloc(size_t bytes);

// Check if iteration count is within limits
int sandbox_check_iteration(int current_iteration);

// Shutdown sandbox
void sandbox_shutdown(void);

#endif // SANDBOX_H
