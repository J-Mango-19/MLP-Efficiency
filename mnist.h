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
float random_float();
void init_weights(Matrix *W);
Matrix transpose_matrix(Matrix *arr);
void XY_split(Matrix *data, Matrix *X, float **Y); 
void train_test_split(Matrix *data, Matrix *test_data, Matrix *train_data);
void normalize(Matrix *X_train, Matrix *X_test);
void softmax(Matrix *Z);
void relu(Matrix *Z);
void multiply_matrices(Matrix *A, Matrix *B, Matrix *C); 
void forward_pass(Layers *layers, Matrix *X, Matrix *W1, Matrix *W2, Matrix *W3);
Layers *init_layers(Matrix *X, Matrix *W1, Matrix *W2, Matrix *W3);
void append_bias_factor(Matrix *A);
void append_bias_input(Matrix *X_train, Matrix *X_test);
void copy_matrix_values(Matrix *original, Matrix *New);
void copy_some_matrix_values(Matrix *original, Matrix *New);
void set_matrix_to_zeros(Matrix *Z);


void get_matrix_stats(Matrix *problem);
