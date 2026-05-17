// photonic_rng.h — Thread-safe PRNG and shared math constants
//
// Centralizes all random number generation and removes duplicate M_PI/Box-Muller
// definitions scattered across the codebase. Uses a simple but fast xorshift-based
// PRNG that is thread-safe (each caller manages its own seed state).
#ifndef PHOTONIC_RNG_H
#define PHOTONIC_RNG_H

#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─── Thread-Safe PRNG (splitmix64-seeded xorshift128+) ──────────────

typedef struct {
    uint64_t s[2];
} PhoRng;

// Initialize RNG from a seed value
static inline PhoRng pho_rng_init(uint64_t seed) {
    PhoRng rng;
    // splitmix64 to generate initial state from single seed
    seed += 0x9e3779b97f4a7c15ULL;
    seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9ULL;
    seed = (seed ^ (seed >> 27)) * 0x94d049bb133111ebULL;
    rng.s[0] = seed ^ (seed >> 31);
    seed += 0x9e3779b97f4a7c15ULL;
    seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9ULL;
    seed = (seed ^ (seed >> 27)) * 0x94d049bb133111ebULL;
    rng.s[1] = seed ^ (seed >> 31);
    if (rng.s[0] == 0 && rng.s[1] == 0) rng.s[0] = 1;
    return rng;
}

// Generate next random uint64
static inline uint64_t pho_rng_next(PhoRng *rng) {
    uint64_t s0 = rng->s[0];
    uint64_t s1 = rng->s[1];
    uint64_t result = s0 + s1;
    s1 ^= s0;
    rng->s[0] = ((s0 << 24) | (s0 >> 40)) ^ s1 ^ (s1 << 16);
    rng->s[1] = (s1 << 37) | (s1 >> 27);
    return result;
}

// Generate uniform double in [0, 1)
static inline double pho_rng_uniform(PhoRng *rng) {
    return (double)(pho_rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

// Generate Gaussian random variable using Box-Muller transform
static inline double pho_rng_normal(PhoRng *rng, double mean, double std) {
    double u1 = pho_rng_uniform(rng);
    double u2 = pho_rng_uniform(rng);
    if (u1 < 1e-15) u1 = 1e-15;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + z * std;
}

#endif // PHOTONIC_RNG_H
