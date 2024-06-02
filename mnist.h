#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
    int nrows; 
    int ncols; 
    float **mat;
} Matrix;

typedef struct {
    Matrix *Z1;
    Matrix *Z2;
    Matrix *Z3;
    Matrix *A1;
    Matrix *A2;
    Matrix *A3;
} Layers;

Matrix read_csv(const char *filename);
void free_matrix_arr(Matrix arr);
void free_matrix_struct(Matrix *arr);
Matrix transpose_matrix(Matrix *arr);
void normalize_data(Matrix *data);
void train_test_split(Matrix *data, Matrix *test_data, Matrix *train_data);
void normalize(Matrix *X_train, Matrix *X_test);
void softmax(Matrix *Z);
void relu(Matrix *Z);
Matrix *multiply_matrices(Matrix *A, Matrix *B, bool extra_bias_row);
Layers forward_pass(Matrix *X, Matrix *W1, Matrix *W2, Matrix *W3);
Matrix *copy_matrix(Matrix *original);



