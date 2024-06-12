#include <stdbool.h>

// holds a pointer to a 2d array of floats and its dimensions
typedef struct {
    int nrows; 
    int ncols; 
    float **mat;
} Matrix;

// holds the unactivated (Z) and activated (A) layers of the net
typedef struct {
    Matrix *Z1;
    Matrix *Z2;
    Matrix *Z3;
    Matrix *A1;
    Matrix *A2;
    Matrix *A3;
} Layers;

// holds the gradient matrices
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

// holds miscellaneous allocated matrices in order to avoid repetitive allocation
typedef struct {
    Matrix *one_hot_Y;
    Matrix *A2T;
    Matrix *W3T;
    Matrix *A1T;
    Matrix *W2T;
    Matrix *XT;
} Transpose;

// dynamic memory allocation
Matrix *allocate_matrix(int nrows, int ncols);
Layers *init_layers(Matrix *X, Matrix *W1, Matrix *W2, Matrix *W3);
void init_transpose(Transpose *transpose, Layers *layers, int batch_size, Matrix *Y, Matrix *W2, Matrix *W3, Matrix *X);
void init_deltas(Deltas *deltas, Layers *layers, Matrix *W1, Matrix *W2, Matrix *W3, Matrix *X);
void init_weights(Matrix *W);


// input data processing
Matrix read_csv(const char *filename);
void transpose_matrix(Matrix *arr, Matrix *transposed);
void XY_split(Matrix *data, Matrix *X, Matrix *Y); 
void train_test_split(Matrix *data, Matrix *test_data, Matrix *train_data);
void normalize(Matrix *X_train, Matrix *X_test);
void append_bias_input(Matrix *X_train, Matrix *X_test);


// activation functions
void softmax(Matrix *Z);
void relu(Matrix *Z);

// matrix operations 
void multiply_matrices(Matrix *A, Matrix *B, Matrix *C); 
void multiply_matrices_elementwise(Matrix *A, Matrix *B, Matrix *C, bool omit_last_row);
void copy_matrix_values(Matrix *original, Matrix *New);
void copy_some_matrix_values(Matrix *original, Matrix *New, int start_index, int end_index);
void display_matrix(Matrix *X);
void divide_matrix_elementwise(Matrix *matrix, int divisor);
void get_matrix_stats(Matrix *problem);
void subtract_matrices(Matrix *A, Matrix *B, Matrix *C);
void one_hot(Matrix *Y, Matrix *one_hot_Y);

// neural net commands
void forward_pass(Layers *layers, Matrix *X, Matrix *W1, Matrix *W2, Matrix *W3);
void inference_one_example(Matrix *X_test, Matrix *Y_test, Matrix *W1, Matrix *W2, Matrix *W3, int index);
void backward_pass(Matrix *X, Layers *layers, Matrix *W1, Matrix *W2, Matrix *W3, Matrix *Y, Deltas *deltas, Transpose *transpose);
void update_weights(Deltas *deltas, Matrix *W1, Matrix *W2, Matrix *W3, float lr);
float get_accuracy(Matrix *yhat, Matrix *Y);

// dynamic memory freeing functions 
void free_layers(Layers *layers);
void free_transpose(Transpose *transpose);
void free_deltas(Deltas *deltas);
void free_matrix_arr(Matrix arr);
void free_matrix_struct(Matrix *arr);

// unclassified
float random_float();
void append_bias_factor(Matrix *A);
void deriv_relu(Matrix *Z, Matrix *derivative);
