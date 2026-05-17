// ibm_photonic.c — IBM Quantum Photonic API integration
//
// Stub for interfacing with IBM's photonic quantum hardware.
// จะใช้งานได้จริงเมื่อได้รับ API access จาก IBM

#include <stdio.h>
#include "../c_api.h"

// ─── IBM Backend ────────────────────────────────────────────────────

typedef struct {
    const char *api_key;
    const char *endpoint;
    int         connected;
} IBMPhotonicConfig;

int ibm_photonic_init(const char *api_key, const char *endpoint) {
    (void)api_key;
    (void)endpoint;
    printf("[IBM Photonic] Backend not yet available — using simulation mode\n");
    return -1; // Not implemented
}

int ibm_photonic_run(const void *circuit, void *result) {
    (void)circuit;
    (void)result;
    return -1; // Not implemented
}

void ibm_photonic_shutdown(void) {
    // No-op
}
