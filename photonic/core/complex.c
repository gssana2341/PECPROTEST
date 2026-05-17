#include "complex.h"
#include <math.h>

Complex complex_new(double real, double imag) {
    Complex c = {real, imag};
    return c;
}

Complex complex_add(Complex a, Complex b) {
    Complex c = {a.real + b.real, a.imag + b.imag};
    return c;
}

Complex complex_mul(Complex a, Complex b) {
    return complex_new(
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    );
}

Complex complex_div(Complex a, Complex b) {
    double n_sq = complex_norm_sq(b);
    if (n_sq < 1e-15) return complex_new(0.0, 0.0);
    Complex b_conj = complex_conj(b);
    Complex num = complex_mul(a, b_conj);
    return complex_new(num.real / n_sq, num.imag / n_sq);
}

Complex complex_conj(Complex a) {
    Complex c = {a.real, -a.imag};
    return c;
}

double complex_norm(Complex a) {
    return sqrt(a.real * a.real + a.imag * a.imag);
}

Complex complex_sub(Complex a, Complex b) {
    return (Complex){ a.real - b.real, a.imag - b.imag };
}

double complex_norm_sq(Complex a) {
    return a.real * a.real + a.imag * a.imag;
}

double complex_phase(Complex a) {
    return atan2(a.imag, a.real);
}
