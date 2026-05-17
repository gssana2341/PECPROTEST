// energy_model.c — energy consumption estimation
//
// นับจำนวน operation (multiply, add, interference) เพื่อประมาณ
// energy consumption ของ photonic circuit เทียบกับ electronic circuit

#include <stdio.h>
#include "../core/complex.h"
#include "../core/matrix.h"

// ─── Energy Constants (picojoules per operation) ────────────────────

#define ENERGY_PHOTONIC_MUL     0.01   // pJ — photonic interference
#define ENERGY_PHOTONIC_ADD     0.005  // pJ — beam combining
#define ENERGY_ELECTRONIC_MUL   4.6    // pJ — 45nm CMOS FP multiply
#define ENERGY_ELECTRONIC_ADD   0.9    // pJ — 45nm CMOS FP add

// ─── Energy Tracker ─────────────────────────────────────────────────

typedef struct {
    long long mul_count;
    long long add_count;
    long long interference_count;
} EnergyTracker;

EnergyTracker energy_tracker_new(void) {
    EnergyTracker t = {0, 0, 0};
    return t;
}

void energy_tracker_reset(EnergyTracker *t) {
    t->mul_count = 0;
    t->add_count = 0;
    t->interference_count = 0;
}

// Estimate total photonic energy (pJ)
double energy_estimate_photonic(const EnergyTracker *t) {
    return (double)t->mul_count * ENERGY_PHOTONIC_MUL
         + (double)t->add_count * ENERGY_PHOTONIC_ADD;
}

// Estimate equivalent electronic energy (pJ)
double energy_estimate_electronic(const EnergyTracker *t) {
    return (double)t->mul_count * ENERGY_ELECTRONIC_MUL
         + (double)t->add_count * ENERGY_ELECTRONIC_ADD;
}

// Print comparison report
void energy_print_report(const EnergyTracker *t) {
    double pho = energy_estimate_photonic(t);
    double elc = energy_estimate_electronic(t);
    printf("=== Energy Report ===\n");
    printf("Operations:  %lld mul, %lld add\n", t->mul_count, t->add_count);
    printf("Photonic:    %.4f pJ\n", pho);
    printf("Electronic:  %.4f pJ\n", elc);
    printf("Ratio:       %.2fx savings\n", elc / (pho > 0 ? pho : 1.0));
}
