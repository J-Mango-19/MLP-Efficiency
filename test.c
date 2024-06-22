#include <stdio.h>

void add(int *x, int*y, int *z) {
    *x = 1;
    *y = 2;
    *z = 3;
    printf("%d\n", *x + *y + *z);
}

int main() {
    int x;
    int y; 
    int z;
    printf("%p\n", &x);
    add(&x, &y, &z);
    return 0;
}
