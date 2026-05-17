#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <stddef.h>

typedef struct {
    double pixels[784];
    int label;
} ImageSample;

typedef struct {
    ImageSample *samples;
    size_t num_samples;
    size_t capacity;
} ImageDataset;

// Load MNIST from CSV format
// Returns 1 on success, 0 on failure
int load_mnist_csv(const char *filepath, int max_samples, ImageDataset *dataset);

// Free dataset memory
void free_dataset(ImageDataset *dataset);

// Shuffle dataset in-place
void shuffle_dataset(ImageDataset *dataset);

#endif // DATA_LOADER_H
