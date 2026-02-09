#include <stdlib.h>
#include <stdio.h>

#define MICRO_malloc(s) malloc(s)
#define MICRO_free(p) free((p))

// Minimal test case: Use After Free
void test_uaf() {
    int *ptr = (int*)MICRO_malloc(sizeof(int));
    *ptr = 42;
    
    MICRO_free(ptr);  // Free the memory
    
    // BUG: Use after free - CodeQL should detect this
    printf("Value: %d\n", *ptr);
}

// Minimal test case: Double Free
void test_double_free() {
    int *ptr = (int*)MICRO_malloc(sizeof(int));
    *ptr = 42;
    
    MICRO_free(ptr);
    MICRO_free(ptr);  // BUG: Double free - CodeQL should detect this
}

// Minimal test case: Memory Leak
void test_memory_leak() {
    int *ptr = (int*)MICRO_malloc(sizeof(int));
    *ptr = 42;
    
    // BUG: Memory leak - ptr is never freed
    // Function returns without freeing memory
}

int main() {
    test_uaf();
    test_double_free();
    test_memory_leak();
    return 0;
}

