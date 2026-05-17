// complex.h – basic complex number operations
#ifndef COMPLEX_H
#define COMPLEX_H

typedef struct {
    double real;
    double imag;
} Complex;

// Create a complex number
Complex complex_new(double real, double imag);

// Basic arithmetic
Complex complex_add(Complex a, Complex b);
Complex complex_sub(Complex a, Complex b);
Complex complex_mul(Complex a, Complex b);
Complex complex_div(Complex a, Complex b);
Complex complex_conj(Complex a);
double complex_norm(Complex a);
double complex_norm_sq(Complex a);
double complex_phase(Complex a);

#endif // COMPLEX_H
