#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

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

typedef struct {
    Matrix *dZ3;
    Matrix *dW3;
    Matrix *dA2;
    Matrix *dZ2;
    Matrix *dA2_dZ2;
    Matrix *dW2;
    Matrix *dA1;
    Matrix *dZ1;
    Matrix *dA1_dZ1;
    Matrix *dW1;
} Deltas;

typedef struct {
    Matrix *one_hot_Y;
    Matrix *A2T;
    Matrix *W3T;
    Matrix *A1T;
    Matrix *W2T;
    Matrix *XT;
} Transpose;


Matrix read_csv(const char *filename);
void free_matrix_arr(Matrix arr);
void free_matrix_struct(Matrix *arr);
float random_float();
void init_weights(Matrix *W);
void transpose_matrix(Matrix *arr, Matrix *transposed);
void XY_split(Matrix *data, Matrix *X, Matrix *Y); 
void train_test_split(Matrix *data, Matrix *test_data, Matrix *train_data);
void normalize(Matrix *X_train, Matrix *X_test);
void softmax(Matrix *Z);
void relu(Matrix *Z);
void multiply_matrices(Matrix *A, Matrix *B, Matrix *C); 
void forward_pass(Layers *layers, Matrix *X, Matrix *W1, Matrix *W2, Matrix *W3);
Layers *init_layers(Matrix *X, Matrix *W1, Matrix *W2, Matrix *W3);
Layers *init_avg_layers(Matrix *X, Matrix *W1, Matrix *W2, Matrix *W3);
void append_bias_factor(Matrix *A);
void append_bias_input(Matrix *X_train, Matrix *X_test);
void copy_matrix_values(Matrix *original, Matrix *New);
void copy_some_matrix_values(Matrix *original, Matrix *New, int start_index, int end_index);
void set_matrix_to_zeros(Matrix *Z);
void subtract_matrices(Matrix *A, Matrix *B, Matrix *C);
Matrix *allocate_matrix(int nrows, int ncols);
void deriv_relu(Matrix *Z, Matrix *derivative);
void multiply_matrices_elementwise(Matrix *A, Matrix *B, Matrix *C, bool omit_last_row);
void init_deltas(Deltas *deltas, Layers *layers, Matrix *W1, Matrix *W2, Matrix *W3, Matrix *X);
void update_weights(Deltas *deltas, Matrix *W1, Matrix *W2, Matrix *W3, float lr);
void one_hot(Matrix *Y, Matrix *one_hot_Y);
float get_accuracy(Matrix *yhat, Matrix *Y);
void divide_matrix_elementwise(Matrix *matrix, int divisor);
void init_transpose(Transpose *transpose, Layers *layers, int batch_size, Matrix *Y, Matrix *W2, Matrix *W3, Matrix *X);
void backward_pass(Matrix *X, Layers *layers, Matrix *W1, Matrix *W2, Matrix *W3, Matrix *Y, Deltas *deltas, Transpose *transpose);
void get_matrix_stats(Matrix *problem);
