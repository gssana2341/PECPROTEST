#include "data_loader.h"
#include "../core/memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int load_mnist_csv(const char *filepath, int max_samples, ImageDataset *dataset) {
    if (!dataset) return 0;
    
    FILE *fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open dataset file: %s\n", filepath);
        return 0;
    }
    
    // Allocate initial memory
    size_t initial_cap = max_samples > 0 ? max_samples : 1000;
    dataset->samples = (ImageSample *)pho_alloc(initial_cap * sizeof(ImageSample), "dataset.samples");
    if (!dataset->samples) {
        fclose(fp);
        return 0;
    }
    dataset->capacity = initial_cap;
    dataset->num_samples = 0;
    
    char line[10000]; // Ensure buffer is large enough for 785 comma separated values
    
    // Read header
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return 0;
    }
    
    while (fgets(line, sizeof(line), fp)) {
        if (max_samples > 0 && dataset->num_samples >= (size_t)max_samples) {
            break;
        }
        
        // Parse line
        char *ptr = line;
        int col = 0;
        
        ImageSample *sample = &dataset->samples[dataset->num_samples];
        
        while (ptr) {
            char *next = strchr(ptr, ',');
            if (next) {
                *next = '\0';
                next++;
            } else if (strchr(ptr, '\n')) {
                *strchr(ptr, '\n') = '\0';
            }
            
            if (col == 0) {
                sample->label = atoi(ptr);
            } else if (col <= 784) {
                // Normalize 0-255 to 0.0-1.0
                sample->pixels[col - 1] = atof(ptr) / 255.0;
            }
            
            ptr = next;
            col++;
        }
        
        dataset->num_samples++;
        
        // Realloc not supported by basic pho_alloc easily, so we assume max_samples is enough
        if (dataset->num_samples >= dataset->capacity && max_samples <= 0) {
            size_t new_cap = dataset->capacity * 2;
            ImageSample *new_arr = (ImageSample *)pho_alloc(new_cap * sizeof(ImageSample), "dataset.samples.new");
            if (new_arr) {
                memcpy(new_arr, dataset->samples, dataset->num_samples * sizeof(ImageSample));
                pho_free(dataset->samples, "dataset.samples");
                dataset->samples = new_arr;
                dataset->capacity = new_cap;
            } else {
                break;
            }
        }
    }
    
    fclose(fp);
    return 1;
}

void free_dataset(ImageDataset *dataset) {
    if (dataset && dataset->samples) {
        pho_free(dataset->samples, "dataset.samples");
        dataset->samples = NULL;
    }
    dataset->num_samples = 0;
    dataset->capacity = 0;
}

void shuffle_dataset(ImageDataset *dataset) {
    if (!dataset || !dataset->samples || dataset->num_samples <= 1) return;
    
    for (size_t i = dataset->num_samples - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        ImageSample temp = dataset->samples[i];
        dataset->samples[i] = dataset->samples[j];
        dataset->samples[j] = temp;
    }
}
