#include <stdlib.h>

int foo(int n) {
    int *p = NULL;

    if (n > 0) {
        p = (int *)malloc(sizeof(int));
        *p = 42;
    }

    if (n < 0) {
        free(p);
    }

    if (n > 10) {
        return *p;   // ❌ use-after-free
    }

    return 0;
}

int main() {
    foo(15);
    return 0;
}