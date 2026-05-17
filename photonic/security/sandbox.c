// sandbox.c — Execution sandbox implementation for photonic simulations
//
// Enforces resource limits (memory, iterations) to prevent runaway
// computations in embedded, edge, and production environments.

#include "sandbox.h"
#include <stdio.h>

// ─── Sandbox State ──────────────────────────────────────────────────

static int    g_sandbox_active = 0;
static size_t g_max_memory_bytes = 0;
static size_t g_current_memory_bytes = 0;
static int    g_max_iterations = 0;
static int    g_allow_file_io = 1;

// Create a default sandbox config (conservative limits)
SandboxConfig sandbox_default(void) {
    SandboxConfig cfg;
    cfg.max_memory_mb = 512;       // 512 MB
    cfg.max_qubits = 256;          // 256 optical ports max
    cfg.max_iterations = 100000;   // 100k iteration ceiling
    cfg.allow_file_io = 1;         // allow file I/O by default
    return cfg;
}

// Initialize sandbox enforcement
int sandbox_init(const SandboxConfig *cfg) {
    if (!cfg) return -1;

    g_max_memory_bytes = cfg->max_memory_mb * 1024ULL * 1024ULL;
    g_max_iterations = cfg->max_iterations;
    g_allow_file_io = cfg->allow_file_io;
    g_current_memory_bytes = 0;
    g_sandbox_active = 1;

    fprintf(stderr, "[Sandbox] Initialized: max_mem=%zuMB, max_iter=%d, file_io=%s\n",
            cfg->max_memory_mb, cfg->max_iterations,
            cfg->allow_file_io ? "allowed" : "blocked");
    return 0;
}

// Check if an allocation is within sandbox limits
int sandbox_check_alloc(size_t bytes) {
    if (!g_sandbox_active) return 1; // no sandbox = always allow

    if (g_current_memory_bytes + bytes > g_max_memory_bytes) {
        fprintf(stderr, "[Sandbox] DENIED: allocation of %zu bytes would exceed limit (%zu / %zu)\n",
                bytes, g_current_memory_bytes, g_max_memory_bytes);
        return 0; // deny
    }

    g_current_memory_bytes += bytes;
    return 1; // allow
}

// Check if iteration count is within limits
int sandbox_check_iteration(int current_iteration) {
    if (!g_sandbox_active) return 1; // no sandbox = always allow

    if (g_max_iterations > 0 && current_iteration >= g_max_iterations) {
        fprintf(stderr, "[Sandbox] DENIED: iteration %d reached limit %d\n",
                current_iteration, g_max_iterations);
        return 0; // deny
    }

    return 1; // allow
}

// Shutdown sandbox
void sandbox_shutdown(void) {
    if (g_sandbox_active) {
        fprintf(stderr, "[Sandbox] Shutdown. Peak memory tracked: %zu bytes\n",
                g_current_memory_bytes);
    }
    g_sandbox_active = 0;
    g_current_memory_bytes = 0;
}
