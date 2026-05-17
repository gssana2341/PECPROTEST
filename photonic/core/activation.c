#include "activation.h"
#include <math.h>

Complex activation_sigmoid_optical(Complex z) {
    // Complex sigmoid: 1 / (1 + exp(-z))
    // exp(-z) = exp(-x) * (cos(-y) + i*sin(-y))
    double exp_nx = exp(-z.real);
    double cos_ny = cos(-z.imag);
    double sin_ny = sin(-z.imag);
    
    // denom = 1 + exp(-z)
    double denom_real = 1.0 + exp_nx * cos_ny;
    double denom_imag = exp_nx * sin_ny;
    
    // 1 / denom = conj(denom) / norm(denom)^2
    double norm_sq = denom_real * denom_real + denom_imag * denom_imag;
    
    if (norm_sq < 1e-15)
        return complex_new(0.0, 0.0);
        
    return complex_new(denom_real / norm_sq, -denom_imag / norm_sq);
}

Complex activation_sigmoid_optical_grad(Complex z) {
    Complex s = activation_sigmoid_optical(z);
    Complex one_minus_s = complex_new(1.0 - s.real, -s.imag);
    return complex_mul(s, one_minus_s);
}

Complex activation_kerr(Complex z, double gamma) {
    double intensity = complex_norm_sq(z);
    double phase = gamma * intensity;
    Complex phase_factor = complex_new(cos(phase), sin(phase));
    return complex_mul(z, phase_factor);
}

void activation_softmax_optical(const Complex *input, Complex *output, size_t length) {
    if (!input || !output || length == 0) return;
    
    // Optical softmax usually acts on the intensity (amplitude squared)
    double max_intensity = -1.0;
    for (size_t i = 0; i < length; i++) {
        double intensity = complex_norm(input[i]);
        intensity = intensity * intensity;
        if (intensity > max_intensity) {
            max_intensity = intensity;
        }
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < length; i++) {
        double intensity = complex_norm(input[i]);
        intensity = intensity * intensity;
        sum += exp(intensity - max_intensity);
    }
    
    for (size_t i = 0; i < length; i++) {
        double intensity = complex_norm(input[i]);
        intensity = intensity * intensity;
        double prob = exp(intensity - max_intensity) / sum;
        
        // Keep the phase, but scale amplitude to sqrt(prob)
        // so that intensity equals the probability
        double norm = complex_norm(input[i]);
        if (norm > 1e-12) {
            double scale = sqrt(prob) / norm;
            output[i] = complex_new(input[i].real * scale, input[i].imag * scale);
        } else {
            output[i] = complex_new(sqrt(prob), 0.0);
        }
    }
}
