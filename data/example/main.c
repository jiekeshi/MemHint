/**
 * Test file for MUST_CHECK_NULL / NULLABLE_RETURN annotations
 *
 * This file contains various scenarios where functions may return NULL
 * and the return values should be checked before use.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// Custom nullable functions (should be annotated as NULLABLE_RETURN)
// ============================================================================

/**
 * Custom function that may return NULL if the key is not found.
 * Annotation: NULLABLE_RETURN
 */
char* find_config_value(const char* key) {
    // Simulated lookup - returns NULL if not found
    if (strcmp(key, "debug") == 0) {
        return "true";
    }
    return NULL;  // Key not found
}

/**
 * Custom function that may return NULL on error.
 * Annotation: NULLABLE_RETURN
 */
void* create_resource(int type) {
    if (type < 0 || type > 10) {
        return NULL;  // Invalid type
    }
    return malloc(sizeof(int) * 100);
}

/**
 * Custom function that wraps file opening - may return NULL.
 * Annotation: NULLABLE_RETURN
 */
FILE* open_log_file(const char* filename) {
    if (filename == NULL || strlen(filename) == 0) {
        return NULL;
    }
    return fopen(filename, "a");
}

/**
 * Custom allocator - may return NULL on failure.
 * Annotation: ALLOC_SOURCE (which implies NULLABLE_RETURN)
 */
void* my_alloc(size_t size) {
    if (size == 0) {
        return NULL;
    }
    return malloc(size);
}

/**
 * Custom deallocator.
 * Annotation: FREE_SINK
 */
void my_free(void* ptr) {
    free(ptr);
}

// ============================================================================
// Test cases - Missing NULL checks (bugs)
// ============================================================================

/**
 * BUG: Missing NULL check after find_config_value
 */
void test_missing_null_check_1() {
    char* value = find_config_value("unknown_key");
    // BUG: value could be NULL, but we use it directly
    printf("Config value: %s\n", value);  // Potential NULL dereference
}

/**
 * BUG: Missing NULL check after create_resource
 */
void test_missing_null_check_2() {
    void* resource = create_resource(-1);  // Will return NULL
    // BUG: resource could be NULL
    int* data = (int*)resource;
    data[0] = 42;  // Potential NULL dereference
}

/**
 * BUG: Missing NULL check after open_log_file
 */
void test_missing_null_check_3() {
    FILE* log = open_log_file("");  // Will return NULL for empty filename
    // BUG: log could be NULL
    fprintf(log, "Log message\n");  // Potential NULL dereference
    fclose(log);
}

/**
 * BUG: Missing NULL check after my_alloc
 */
void test_missing_null_check_4() {
    char* buffer = (char*)my_alloc(1024);
    // BUG: buffer could be NULL
    strcpy(buffer, "Hello, World!");  // Potential NULL dereference
    my_free(buffer);
}

/**
 * BUG: Missing NULL check after standard malloc
 */
void test_missing_null_check_5() {
    int* arr = (int*)malloc(sizeof(int) * 100);
    // BUG: arr could be NULL
    arr[0] = 1;  // Potential NULL dereference
    free(arr);
}

/**
 * BUG: Missing NULL check after fopen
 */
void test_missing_null_check_6() {
    FILE* f = fopen("/nonexistent/path/file.txt", "r");
    // BUG: f could be NULL
    char buf[100];
    fgets(buf, 100, f);  // Potential NULL dereference
    fclose(f);
}

// ============================================================================
// Test cases - Correct NULL checks (no bugs)
// ============================================================================

/**
 * CORRECT: Proper NULL check after find_config_value
 */
void test_correct_null_check_1() {
    char* value = find_config_value("unknown_key");
    if (value != NULL) {
        printf("Config value: %s\n", value);
    } else {
        printf("Config key not found\n");
    }
}

/**
 * CORRECT: Proper NULL check after create_resource
 */
void test_correct_null_check_2() {
    void* resource = create_resource(5);
    if (resource == NULL) {
        fprintf(stderr, "Failed to create resource\n");
        return;
    }
    int* data = (int*)resource;
    data[0] = 42;
    my_free(resource);
}

/**
 * CORRECT: Proper NULL check after my_alloc
 */
void test_correct_null_check_3() {
    char* buffer = (char*)my_alloc(1024);
    if (buffer == NULL) {
        fprintf(stderr, "Allocation failed\n");
        return;
    }
    strcpy(buffer, "Hello, World!");
    printf("%s\n", buffer);
    my_free(buffer);
}

/**
 * CORRECT: Early return pattern
 */
void test_correct_null_check_4() {
    FILE* log = open_log_file("app.log");
    if (!log) {
        return;  // Early return if NULL
    }
    fprintf(log, "Application started\n");
    fclose(log);
}

// ============================================================================
// Main function
// ============================================================================

int main() {
    printf("Running NULL check tests...\n\n");

    // These will have bugs (missing NULL checks)
    printf("=== Tests with missing NULL checks (bugs) ===\n");
    // Commented out to prevent crashes during actual execution
    // test_missing_null_check_1();
    // test_missing_null_check_2();
    // test_missing_null_check_3();
    // test_missing_null_check_4();
    // test_missing_null_check_5();
    // test_missing_null_check_6();

    // These are correct
    printf("=== Tests with correct NULL checks ===\n");
    test_correct_null_check_1();
    test_correct_null_check_2();
    test_correct_null_check_3();
    test_correct_null_check_4();

    printf("\nAll tests completed.\n");
    return 0;
}