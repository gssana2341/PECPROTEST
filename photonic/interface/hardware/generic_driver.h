// generic_driver.h — generic hardware driver interface
//
// Implement interface นี้ แล้ว plug เข้าระบบได้เลย
// รองรับ hardware backend หลายตัวพร้อมกัน
#ifndef GENERIC_DRIVER_H
#define GENERIC_DRIVER_H

// Forward declarations
typedef struct Circuit Circuit;
typedef struct Result  Result;

// ─── Hardware Backend Interface ─────────────────────────────────────

typedef struct {
    const char *name;
    int  (*init)(void *config);
    int  (*run_circuit)(Circuit *c, Result *out);
    void (*shutdown)(void);
} HardwareBackend;

// ─── Backend Registry ───────────────────────────────────────────────

#define MAX_BACKENDS 16

// Register a hardware backend
int driver_register(const HardwareBackend *backend);

// Get a backend by name
const HardwareBackend *driver_get(const char *name);

// List all registered backends
int driver_list(const char **names, int max_count);

#endif // GENERIC_DRIVER_H
