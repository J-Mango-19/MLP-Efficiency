#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int nrows; 
    int ncols; 
    float **mat;
} Matrix;

Matrix read_csv(const char *filename);
void free_matrix(Matrix arr);
Matrix transpose_matrix(Matrix *arr);
void normalize_data(Matrix *data);
void train_test_split(Matrix *data, Matrix *test_data, Matrix *train_data);
void normalize(Matrix *X_train, Matrix *X_test);
void softmax(Matrix *Z);
void relu(Matrix *Z);



