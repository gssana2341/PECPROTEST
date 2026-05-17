#include "pooling.h"
#include <math.h>

void optical_lens_pool_28_to_8(const double *pixels, double *pooled) {
    // 28x28 -> 8x8 average pooling with weighted overlap calculations
    int out_size = 8;
    double cell = 28.0 / out_size;  // = 3.5
    
    for (int oi = 0; oi < out_size; oi++) {
        for (int oj = 0; oj < out_size; oj++) {
            double sum = 0.0, weight_total = 0.0;
            
            double r_start = oi * cell;
            double r_end   = r_start + cell;
            double c_start = oj * cell;
            double c_end   = c_start + cell;
            
            for (int pi = (int)r_start; pi < (int)r_end + 1; pi++) {
                for (int pj = (int)c_start; pj < (int)c_end + 1; pj++) {
                    if (pi >= 28 || pj >= 28) continue;
                    
                    // overlap weight
                    double wr = fmin(pi+1, r_end) - fmax(pi, r_start);
                    double wc = fmin(pj+1, c_end) - fmax(pj, c_start);
                    double w  = wr * wc;
                    
                    sum          += pixels[pi*28+pj] * w;
                    weight_total += w;
                }
            }
            pooled[oi*out_size+oj] = weight_total > 0 ? sum/weight_total : 0.0;
        }
    }
}

void optical_lens_pool_28_to_4(const double *input_784, double *output_16) {
    for (int i = 0; i < 16; i++) {
        output_16[i] = 0.0;
    }
    
    int counts[16] = {0};
    
    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            int out_r = r * 4 / 28;
            int out_c = c * 4 / 28;
            int out_idx = out_r * 4 + out_c;
            
            output_16[out_idx] += input_784[r * 28 + c];
            counts[out_idx]++;
        }
    }
    
    for (int i = 0; i < 16; i++) {
        if (counts[i] > 0) {
            output_16[i] /= counts[i];
        }
    }
}
