#include "loss.h"
#include "memory.h"
#include <math.h>

double loss_mse(const double *predicted, const double *target, size_t length) {
    if (!predicted || !target || length == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < length; i++) {
        double diff = predicted[i] - target[i];
        sum += diff * diff;
    }
    return sum / length;
}

void loss_mse_grad(const double *predicted, const double *target, double *grad_out, size_t length) {
    if (!predicted || !target || !grad_out || length == 0) return;
    for (size_t i = 0; i < length; i++) {
        grad_out[i] = 2.0 * (predicted[i] - target[i]) / length;
    }
}

double loss_cross_entropy(const double *predicted, const double *target, size_t length) {
    if (!predicted || !target || length == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < length; i++) {
        double p = predicted[i];
        // Clip to avoid log(0)
        if (p < 1e-15) p = 1e-15;
        if (p > 1.0 - 1e-15) p = 1.0 - 1e-15;
        sum += -target[i] * log(p);
    }
    return sum / length;
}

void loss_cross_entropy_grad(const double *predicted, const double *target, double *grad_out, size_t length) {
    if (!predicted || !target || !grad_out || length == 0) return;
    for (size_t i = 0; i < length; i++) {
        double p = predicted[i];
        if (p < 1e-15) p = 1e-15;
        if (p > 1.0 - 1e-15) p = 1.0 - 1e-15;
        grad_out[i] = -target[i] / p;
    }
}

double loss_cross_entropy_softmax(const double *predicted,
                                   const double *target,
                                   size_t length) {
    if (!predicted || !target || length == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < length; i++) {
        double p = predicted[i];
        if (p < 1e-15) p = 1e-15;
        sum += -target[i] * log(p);
    }
    return sum / length;
}

void loss_cross_entropy_softmax_grad(const double *predicted,
                                      const double *target,
                                      double *grad_out,
                                      size_t length) {
    if (!predicted || !target || !grad_out || length == 0) return;
    for (size_t i = 0; i < length; i++) {
        grad_out[i] = predicted[i] - target[i]; // หลัง softmax
    }
}

double loss_cross_entropy_softmax_optical(const Complex *output, 
                                          double detector_gain, 
                                          const double *target, 
                                          size_t length, 
                                          double *probabilities_out) {
    if (!output || !target || length == 0) return 0.0;
    
    double max_val = -1e9;
    double *vals = (double*)pho_alloc(length * sizeof(double), "softmax.vals");
    if (!vals) return 0.0;
    
    for (size_t i = 0; i < length; i++) {
        double intensity = complex_norm_sq(output[i]);
        vals[i] = intensity * detector_gain;
        if (vals[i] > max_val) {
            max_val = vals[i];
        }
    }
    
    double sum_exp = 0.0;
    for (size_t i = 0; i < length; i++) {
        double e = exp(vals[i] - max_val);
        vals[i] = e;
        sum_exp += e;
    }
    
    double loss = 0.0;
    for (size_t i = 0; i < length; i++) {
        double prob = vals[i] / sum_exp;
        if (probabilities_out) {
            probabilities_out[i] = prob;
        }
        
        if (target[i] > 0.5) { // Assuming target is one-hot or probability
            double p_clip = prob < 1e-15 ? 1e-15 : prob;
            loss += -log(p_clip);
        }
    }
    
    pho_free(vals, "softmax.vals");
    return loss;
}
