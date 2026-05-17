// memory.h — safe memory management + bounds checking
// ห้าม raw malloc/free ใน codebase — ทุกอย่างต้องผ่าน safe allocator
#ifndef MEMORY_H
#define MEMORY_H

#include <stddef.h>

// Safe allocator — wraps malloc with tag tracking and bounds checking
void *pho_alloc(size_t size, const char *tag);

// Safe free — wraps free with tag verification
void pho_free(void *ptr, const char *tag);

// Leak checker — เรียกตอน shutdown เพื่อตรวจสอบว่า memory leak หรือไม่
// Returns 0 if clean, >0 = number of leaked allocations
int pho_check_leaks(void);

// Report all outstanding allocations (for debugging)
void pho_dump_allocations(void);

#endif // MEMORY_H
