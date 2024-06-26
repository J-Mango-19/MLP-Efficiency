#include <stdbool.h>

// holds a pointer to an array of floats and its dimensions
typedef struct {
    int nrows;
    int ncols;
    float *mat;
} Fmatrix;

// holds all model weights as Fmatrix structs
typedef struct {
    Fmatrix *W1;
    Fmatrix *W2;
    Fmatrix *W3;
} Weights;

// holds the unactivated (Z) and activated (A) nodes of the net
typedef struct {
    Fmatrix *Z1;
    Fmatrix *Z2;
    Fmatrix *Z3;
    Fmatrix *A1;
    Fmatrix *A2;
    Fmatrix *A3;
} Nodes;

// holds the gradient matrices
typedef struct {
    Fmatrix *dZ3;
    Fmatrix *dW3;
    Fmatrix *dA2;
    Fmatrix *dZ2;
    Fmatrix *dA2_dZ2;
    Fmatrix *dW2;
    Fmatrix *dA1;
    Fmatrix *dZ1;
    Fmatrix *dA1_dZ1;
    Fmatrix *dW1;
} Deltas;

// holds miscellaneous allocated matrices in order to avoid repetitive allocation
typedef struct {
    Fmatrix *one_hot_Y;
    Fmatrix *A2T;
    Fmatrix *W3T;
    Fmatrix *A1T;
    Fmatrix *W2T;
    Fmatrix *XT;
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
} Preferences;

// utils.c: Nested struct dynamic memory allocation
Nodes *init_nodes(Fmatrix *X, Weights *weights);
void init_misc(Misc *misc, Nodes *nodes, int batch_size, Weights *weights, Fmatrix *X);
void init_deltas(Deltas *deltas, Nodes *nodes, Weights *weights, Fmatrix *X);
void init_weights(Weights *weights, int num_input, int num_hidden_1, int num_hidden_2, int num_output);

// utils.c: input data processing
Fmatrix read_csv(const char *filename);
void split_data(Fmatrix *data, Fmatrix* X_train, Fmatrix *Y_train, Fmatrix *X_test, Fmatrix *Y_test);

// neural_network.c: all functions
void softmax(Fmatrix *Z);
void relu(Fmatrix *Z);
void forward_pass(Nodes *nodes, Fmatrix *X, Weights *weights);
void deriv_relu(Fmatrix *Z, Fmatrix *derivative);
void backward_pass(Fmatrix *X, Nodes *nodes, Weights *weights, Fmatrix *Y, Deltas *deltas, Misc *misc);
void update_weights(Deltas *deltas, Weights *weights, float lr);

// Fmatrix_operations.c: all functions
float *initialize_array(int nrows, int ncols);
Fmatrix *allocate_matrix(int nrows, int ncols);
void transpose_matrix(Fmatrix *arr, Fmatrix *transposed);
void multiply_matrices(Fmatrix *A, Fmatrix *B, Fmatrix *C); 
void multiply_matrices_elementwise(Fmatrix *A, Fmatrix *B, Fmatrix *C, bool omit_last_row);
void scale_matrix(Fmatrix *Fmatrix, float factor);
void subtract_matrices(Fmatrix *A, Fmatrix *B, Fmatrix *C);
void copy_all_matrix_values(Fmatrix *original, Fmatrix *New);
void copy_some_matrix_values(Fmatrix *original, Fmatrix *New, int start_index, int end_index, bool allow_wrap);
void one_hot(Fmatrix *Y, Fmatrix *one_hot_Y);
void argmax_into_yhat(Fmatrix *A, Fmatrix *yhat);
void display_matrix(Fmatrix *X);

// utils.c: dynamic memory freeing functions 
void free_matrix_structs(Fmatrix *test_yhat, Fmatrix *train_yhat, Fmatrix *X_batch, Fmatrix *Y_batch, Fmatrix *W1, Fmatrix *W2, Fmatrix *W3);
void free_matrix_arrays(Fmatrix *X_test, Fmatrix *X_train, Fmatrix *Y_test, Fmatrix *Y_train);
void free_all_nodes(Nodes *nodes_train, Nodes *nodes_test, Nodes *nodes_batch);
void free_nodes(Nodes *nodes);
void free_misc(Misc *misc);
void free_deltas(Deltas *deltas);
void free_matrix_arr(Fmatrix *arr);
void free_matrix_struct(Fmatrix *arr);

// utils.c: user interaction
Preferences *get_input(int argc, char *argv[]);
void usage(int code);
void print_accuracy(int i, Nodes *nodes_train, Nodes *nodes_test, Fmatrix *X_train, Fmatrix *X_test, Fmatrix *train_yhat, Fmatrix *test_yhat, Fmatrix *Y_train, Fmatrix *Y_test, Weights *weights);
void inference_one_example(Fmatrix *X_test, Fmatrix *Y_test, Weights *weights, int index);
void display_examples(int display_start, int display_end, Fmatrix *X_test, Fmatrix *Y_test, Weights *weights);

// utils.c: unclassified 
void append_bias_factor(Fmatrix *A);
void get_next_batch(int i, int batch_size, Fmatrix *X_train, Fmatrix *Y_train, Fmatrix *X_batch, Fmatrix *Y_batch); 
float random_float();
void randomize_weights(Fmatrix *W);
float get_accuracy(Fmatrix *yhat, Fmatrix *Y);

