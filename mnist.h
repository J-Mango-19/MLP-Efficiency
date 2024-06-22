#include <stdbool.h>

// holds a pointer to a 2d array of floats and its dimensions
typedef struct {
    int nrows; 
    int ncols; 
    float **mat;
} Matrix;

// holds all model weights as Matrix structs
typedef struct {
    Matrix *W1;
    Matrix *W2;
    Matrix *W3;
} Weights;

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

typedef struct {
    float lr;
    int batch_size;
    int num_iterations;
    int display_start;
    int display_end;
    int status_interval;
} Preferences;


// dynamic memory allocation
Matrix *allocate_matrix(int nrows, int ncols);
Layers *init_layers(Matrix *X, Weights *weights);
void init_transpose(Transpose *transpose, Layers *layers, int batch_size, Weights *weights, Matrix *X);
void init_deltas(Deltas *deltas, Layers *layers, Weights *weights, Matrix *X);
void init_weights(Weights *weights, int num_input, int num_hidden_1, int num_hidden_2, int num_output);



// input data processing
Matrix read_csv(const char *filename);
void transpose_matrix(Matrix *arr, Matrix *transposed);
void normalize(Matrix *X_train, Matrix *X_test);
float random_float();
void randomize_weights(Matrix *W);


// activation functions
void softmax(Matrix *Z);
void relu(Matrix *Z);

// matrix operations 
void multiply_matrices(Matrix *A, Matrix *B, Matrix *C); 
void multiply_matrices_elementwise(Matrix *A, Matrix *B, Matrix *C, bool omit_last_row);
void copy_matrix_values(Matrix *original, Matrix *New);
void copy_some_matrix_values(Matrix *original, Matrix *New, int start_index, int end_index, bool allow_wrap);
void display_matrix(Matrix *X);
void scale_matrix(Matrix *matrix, float factor);
void subtract_matrices(Matrix *A, Matrix *B, Matrix *C);
void one_hot(Matrix *Y, Matrix *one_hot_Y);
void argmax_into_yhat(Matrix *A, Matrix *yhat);

// neural net commands
void forward_pass(Layers *layers, Matrix *X, Weights *weights);
void inference_one_example(Matrix *X_test, Matrix *Y_test, Weights *weights, int index);
void backward_pass(Matrix *X, Layers *layers, Weights *weights, Matrix *Y, Deltas *deltas, Transpose *transpose);
void update_weights(Deltas *deltas, Weights *weights, float lr);
float get_accuracy(Matrix *yhat, Matrix *Y);

// dynamic memory freeing functions 
void free_layers(Layers *layers);
void free_transpose(Transpose *transpose);
void free_deltas(Deltas *deltas);
void free_matrix_arr(Matrix arr);
void free_matrix_struct(Matrix *arr);

// unclassified
void append_bias_factor(Matrix *A);
void append_bias_input(Matrix *X_train, Matrix *X_test);
void deriv_relu(Matrix *Z, Matrix *derivative);
Preferences *get_input(int argc, char *argv[]);
void usage(int code);
float **initialize_array(int nrows, int ncols);
void split_data(Matrix *data, Matrix* X_train, Matrix *Y_train, Matrix *X_test, Matrix *Y_test);
