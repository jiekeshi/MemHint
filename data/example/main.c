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