#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_LINE_LENGTH 100000

typedef struct {
    int nrows; 
    int ncols; 
    float **mat;
} Matrix;

Matrix read_csv(const char *filename);
void free_arr(Matrix arr);
void remove_first_row(Matrix *data);




