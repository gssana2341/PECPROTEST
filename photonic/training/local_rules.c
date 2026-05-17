// local_rules.c — Hebbian / local learning rules
//
// Learning rules ที่ใช้แค่ local information (pre/post-synaptic activity)
// ไม่ต้องการ global error signal → เหมาะกับ photonic hardware มาก
//
// Rules implemented:
// - Oja's rule (normalized Hebbian)
// - BCM rule (sliding threshold)

#include <stdio.h>
#include <stdlib.h>
#include "../core/complex.h"
#include "../core/matrix.h"

// ─── Local Learning Config ──────────────────────────────────────────

typedef enum {
    LOCAL_RULE_OJA,
    LOCAL_RULE_BCM
} LocalRuleType;

typedef struct {
    LocalRuleType rule;
    double        learning_rate;
    double        threshold;        // สำหรับ BCM rule
    int           max_iterations;
} LocalConfig;

LocalConfig local_default_config(void) {
    LocalConfig cfg = {
        .rule           = LOCAL_RULE_OJA,
        .learning_rate  = 0.001,
        .threshold      = 0.5,
        .max_iterations = 500
    };
    return cfg;
}

// Apply one step of local learning rule
// TODO: implement
int local_train_step(const LocalConfig *cfg, Matrix *weights,
                     const double *pre, const double *post, size_t dim) {
    (void)cfg;
    (void)weights;
    (void)pre;
    (void)post;
    (void)dim;
    return 0;
}
