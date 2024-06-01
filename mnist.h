#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int nrows; 
    int ncols; 
    float **mat;
} Matrix;

Matrix read_csv(const char *filename);
void free_arr(Matrix arr);
Matrix transpose_matrix(Matrix *arr);
Matrix process_data(Matrix *data);
void normalize_data(Matrix *data);




