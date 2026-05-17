// feedback_align.c — Feedback Alignment
//
// ใช้ random fixed feedback matrix แทน backpropagation
// ไม่ต้อง transpose weight matrix → เหมาะกับ photonic hardware
// ที่ทำ transpose ได้ยาก

#include <stdio.h>
#include <stdlib.h>
#include "../core/complex.h"
#include "../core/matrix.h"

// ─── Feedback Alignment Config ──────────────────────────────────────

typedef struct {
    double learning_rate;
    int    max_epochs;
    int    feedback_seed;   // seed สำหรับสร้าง random feedback matrix
} FAConfig;

FAConfig fa_default_config(void) {
    FAConfig cfg = {
        .learning_rate = 0.01,
        .max_epochs    = 100,
        .feedback_seed = 12345
    };
    return cfg;
}

// Initialize random feedback matrix (fixed throughout training)
// TODO: implement
int fa_init_feedback(const FAConfig *cfg, Matrix *feedback, size_t rows, size_t cols) {
    (void)cfg;
    (void)feedback;
    (void)rows;
    (void)cols;
    return 0;
}

// Run one feedback alignment training step
// TODO: implement
int fa_train_step(const FAConfig *cfg, Matrix *weights, const Matrix *feedback,
                  const double *error) {
    (void)cfg;
    (void)weights;
    (void)feedback;
    (void)error;
    return 0;
}
