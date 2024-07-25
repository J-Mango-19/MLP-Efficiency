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

// holds the unactivated (Z) and activated (A) nodes of the net
typedef struct {
    Matrix *Z1;
    Matrix *Z2;
    Matrix *Z3;
    Matrix *A1;
    Matrix *A2;
    Matrix *A3;
} Nodes;

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
} Misc;

typedef struct {
    float lr;
    int batch_size;
    int num_iterations;
    int display_start;
    int display_end;
    int status_interval;
    int num_hidden_1;
    int num_hidden_2;
    int mode;
} Preferences;

// utils.c: Nested struct dynamic memory allocation
Nodes *init_nodes(Matrix *X, Weights *weights);
void init_misc(Misc *misc, Nodes *nodes, int batch_size, Weights *weights, Matrix *X);
void init_deltas(Deltas *deltas, Nodes *nodes, Weights *weights, Matrix *X);
void init_weights(Weights *weights, int num_input, int num_hidden_1, int num_hidden_2, int num_output);

// utils.c: input data processing
Matrix read_csv(const char *filename);
void split_data(Matrix *data, Matrix* X_train, Matrix *Y_train, Matrix *X_test, Matrix *Y_test);

// neural_network.c: all functions
void softmax(Matrix *Z);
void relu(Matrix *Z);
void forward_pass(Nodes *nodes, Matrix *X, Weights *weights);
void deriv_relu(Matrix *Z, Matrix *derivative);
void backward_pass(Matrix *X, Nodes *nodes, Weights *weights, Matrix *Y, Deltas *deltas, Misc *misc);
void update_weights(Deltas *deltas, Weights *weights, float lr);

// matrix_operations.c: all functions
float **initialize_array(int nrows, int ncols);
Matrix *allocate_matrix(int nrows, int ncols);
void transpose_matrix(Matrix *arr, Matrix *transposed);
void multiply_matrices(Matrix *A, Matrix *B, Matrix *C); 
void multiply_matrices_elementwise(Matrix *A, Matrix *B, Matrix *C, bool omit_last_row);
void scale_matrix(Matrix *matrix, float factor);
void subtract_matrices(Matrix *A, Matrix *B, Matrix *C);
void copy_all_matrix_values(Matrix *original, Matrix *New);
void copy_some_matrix_values(Matrix *original, Matrix *New, int start_index, int end_index, bool allow_wrap);
void one_hot(Matrix *Y, Matrix *one_hot_Y);
void argmax_into_yhat(Matrix *A, Matrix *yhat);
void display_matrix(Matrix *X);

// utils.c: dynamic memory freeing functions 
void free_matrix_structs(Matrix *test_yhat, Matrix *train_yhat, Matrix *X_batch, Matrix *Y_batch, Matrix *W1, Matrix *W2, Matrix *W3);
void free_matrix_arrays(Matrix *X_test, Matrix *X_train, Matrix *Y_test, Matrix *Y_train);
void free_all_nodes(Nodes *nodes_train, Nodes *nodes_test, Nodes *nodes_batch);
void free_nodes(Nodes *nodes);
void free_misc(Misc *misc);
void free_deltas(Deltas *deltas);
void free_matrix_arr(Matrix *arr);
void free_matrix_struct(Matrix *arr);

// utils.c: user interaction
Preferences *get_input(int argc, char *argv[]);
void usage(int code);
void print_accuracy(int i, Nodes *nodes_train, Nodes *nodes_test, Matrix *X_train, Matrix *X_test, Matrix *train_yhat, Matrix *test_yhat, Matrix *Y_train, Matrix *Y_test, Weights *weights);
void inference_one_example(Matrix *X_test, Matrix *Y_test, Weights *weights, int index);
void display_examples(int display_start, int display_end, Matrix *X_test, Matrix *Y_test, Weights *weights);
void display_times(float alloc_time, float train_time, float inference_time, int mode);

// utils.c: unclassified 
void append_bias_factor(Matrix *A);
void get_next_batch(int i, int batch_size, Matrix *X_train, Matrix *Y_train, Matrix *X_batch, Matrix *Y_batch); 
float random_float();
void randomize_weights_He(Matrix *W, int fan_in);
float get_accuracy(Matrix *yhat, Matrix *Y);

