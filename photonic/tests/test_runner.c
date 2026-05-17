// test_runner.c — main test harness
//
// Simple test framework for the Photonic project.
// Run all unit/integration tests and report results.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../core/complex.h"
#include "../core/matrix.h"
#include "../core/activation.h"
#include "../core/loss.h"
#include "../core/memory.h"

// ─── Test Framework ─────────────────────────────────────────────────

static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(condition, msg) do {            \
    g_tests_run++;                                  \
    if (condition) {                                \
        g_tests_passed++;                           \
        printf("  [PASS] %s\n", msg);               \
    } else {                                        \
        g_tests_failed++;                           \
        printf("  [FAIL] %s (line %d)\n", msg, __LINE__); \
    }                                               \
} while(0)

#define TEST_ASSERT_NEAR(a, b, eps, msg) \
    TEST_ASSERT(fabs((a) - (b)) < (eps), msg)

// ─── Test Suites (forward declarations) ─────────────────────────────

void test_complex(void);
void test_matrix(void);
void test_activation(void);
void test_loss(void);
void test_memory(void);

// ─── Main ───────────────────────────────────────────────────────────

int main(void) {
    printf("=== Photonic Test Suite ===\n\n");

    printf("[complex]\n");
    test_complex();

    printf("\n[matrix]\n");
    test_matrix();

    printf("\n[activation]\n");
    test_activation();

    printf("\n[loss]\n");
    test_loss();

    printf("\n[memory]\n");
    test_memory();

    // Summary
    printf("\n=== Results ===\n");
    printf("Run: %d | Passed: %d | Failed: %d\n",
           g_tests_run, g_tests_passed, g_tests_failed);

    return g_tests_failed > 0 ? 1 : 0;
}

// ─── Placeholder Test Implementations ───────────────────────────────

void test_complex(void) {
    Complex a = complex_new(1.0, 2.0);
    Complex b = complex_new(3.0, -1.0);
    
    Complex c = complex_add(a, b);
    TEST_ASSERT_NEAR(c.real, 4.0, 1e-6, "complex_add real");
    TEST_ASSERT_NEAR(c.imag, 1.0, 1e-6, "complex_add imag");
    
    Complex d = complex_mul(a, b); // (1+2i)(3-i) = 3 - i + 6i + 2 = 5 + 5i
    TEST_ASSERT_NEAR(d.real, 5.0, 1e-6, "complex_mul real");
    TEST_ASSERT_NEAR(d.imag, 5.0, 1e-6, "complex_mul imag");
    
    double n2 = complex_norm_sq(a);
    TEST_ASSERT_NEAR(n2, 5.0, 1e-6, "complex_norm_sq");
}

void test_matrix(void) {
    Matrix m = matrix_new(2, 2);
    TEST_ASSERT(m.data != NULL, "matrix_new allocates data");
    
    // Set identity
    m.data[0] = complex_new(1.0, 0.0); m.data[1] = complex_new(0.0, 0.0);
    m.data[2] = complex_new(0.0, 0.0); m.data[3] = complex_new(1.0, 0.0);
    
    TEST_ASSERT(matrix_is_unitary(&m, 1e-6) == 1, "identity is unitary");
    
    matrix_scale(&m, 2.0);
    TEST_ASSERT_NEAR(m.data[0].real, 2.0, 1e-6, "matrix_scale");
    
    matrix_free(&m);
    TEST_ASSERT(m.data == NULL, "matrix_free sets data to NULL");
    
    // Inverse tests
    Matrix I = matrix_identity(2);
    Matrix I_inv;
    int res = matrix_inverse(&I, &I_inv);
    TEST_ASSERT(res == 1, "Identity is invertible");
    TEST_ASSERT_NEAR(I_inv.data[0].real, 1.0, 1e-6, "I_inv[0,0] == 1");
    TEST_ASSERT_NEAR(I_inv.data[3].real, 1.0, 1e-6, "I_inv[1,1] == 1");
    matrix_free(&I);
    matrix_free(&I_inv);
    
    Matrix S = matrix_new(2, 2); // Zero matrix
    Matrix S_inv;
    res = matrix_inverse(&S, &S_inv);
    TEST_ASSERT(res == 0, "Zero matrix is singular");
    matrix_free(&S);
    
    Matrix U = matrix_new(2, 2);
    // Swap matrix (unitary)
    U.data[0] = complex_new(0.0, 0.0); U.data[1] = complex_new(1.0, 0.0);
    U.data[2] = complex_new(1.0, 0.0); U.data[3] = complex_new(0.0, 0.0);
    TEST_ASSERT(matrix_is_unitary(&U, 1e-6) == 1, "Swap matrix is unitary");
    
    Matrix U_inv;
    res = matrix_inverse(&U, &U_inv);
    TEST_ASSERT(res == 1, "Unitary is invertible");
    Matrix U_adj = matrix_adjoint(&U);
    
    TEST_ASSERT_NEAR(U_inv.data[1].real, U_adj.data[1].real, 1e-6, "U_inv == U_adj");
    TEST_ASSERT_NEAR(U_inv.data[2].real, U_adj.data[2].real, 1e-6, "U_inv == U_adj");
    
    matrix_free(&U);
    matrix_free(&U_inv);
    matrix_free(&U_adj);
}

void test_activation(void) {
    Complex z = complex_new(0.0, 0.0);
    Complex s = activation_sigmoid_optical(z);
    TEST_ASSERT_NEAR(s.real, 0.5, 1e-6, "sigmoid(0) is 0.5");
    TEST_ASSERT_NEAR(s.imag, 0.0, 1e-6, "sigmoid(0) imag is 0.0");
    
    Complex k = activation_kerr(complex_new(1.0, 0.0), 3.14159265359);
    TEST_ASSERT_NEAR(k.real, -1.0, 1e-5, "kerr phase shift pi");
}

void test_loss(void) {
    double pred[2] = {0.9, 0.1};
    double targ[2] = {1.0, 0.0};
    double mse = loss_mse(pred, targ, 2);
    TEST_ASSERT_NEAR(mse, 0.01, 1e-6, "mse calculation");
    
    double ce = loss_cross_entropy_softmax(pred, targ, 2);
    TEST_ASSERT_NEAR(ce, -log(0.9) / 2.0, 1e-6, "cross_entropy calculation");
}

void test_memory(void) {
    int initial = pho_check_leaks();
    void *p = pho_alloc(100, "test");
    TEST_ASSERT(pho_check_leaks() == initial + 1, "pho_alloc tracks memory");
    pho_free(p, "test");
    TEST_ASSERT(pho_check_leaks() == initial, "pho_free untracks memory");
}
