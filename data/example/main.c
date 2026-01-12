// #include <stdlib.h>

// int foo(int n) {
//     int *p = NULL;

//     if (n > 0) {
//         p = (int *)malloc(sizeof(int));
//         *p = 42;
//     }

//     if (n < 0) {
//         free(p);
//     }

//     if (n > 10) {
//         return *p;   // ❌ use-after-free
//     }

//     return 0;
// }

// int main() {
//     foo(15);
//     return 0;
// }

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* allocate_memory() {
    char *str = (char*)malloc(100 * sizeof(char));
    if (str != NULL) {
        strcpy(str, "Hello, World!");
    }
    return str;
}

int main() {
    char *str = allocate_memory();
    printf("%s\n", str);
    return 0;
}
