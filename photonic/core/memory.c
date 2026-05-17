#include "memory.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdatomic.h>

// Simple memory tracker
static atomic_int g_alloc_count = 0;

void *pho_alloc(size_t size, const char *tag) {
    void *ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "[pho_alloc] failed: %s (%zu bytes)\n",
                tag ? tag : "unknown", size);
        return NULL;
    }
    g_alloc_count++;
    return ptr;
}

void pho_free(void *ptr, const char *tag) {
    (void)tag; // Tracked via logging in a real system
    if (ptr) {
        free(ptr);
        g_alloc_count--;
    }
}

int pho_check_leaks(void) {
    return g_alloc_count;
}

void pho_dump_allocations(void) {
    printf("Outstanding allocations: %d\n", g_alloc_count);
}
