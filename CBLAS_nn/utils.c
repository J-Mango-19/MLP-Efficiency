#include "mnist.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

Fmatrix read_csv(const char* filename) {
    Fmatrix data_matrix;
    data_matrix.nrows = 60000;
    data_matrix.ncols = 785;
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Could not open file %s for reading\n", filename);
        return data_matrix;
    }

    // Allocate memory for the values
    float *data = (float *)malloc(data_matrix.nrows * data_matrix.ncols * sizeof(float));

    char line[16000]; // Large enough buffer to hold one line of the CSV file

    int size = data_matrix.nrows * data_matrix.ncols;
    float *Dp = &data[0];
    char *token;
    for (int i = 0; i < size; i++) {
        if (i % data_matrix.ncols == 0) {
            fgets(line, sizeof(line), file);
            token = strtok(line, ",");
        }
        *Dp++ = strtof(token, NULL);
        token = strtok(NULL, ",");
    }
    data_matrix.mat = data;
    fclose(file);
    return data_matrix;
}

void usage(int code) {
    printf("usage: \n");
    printf("./mnist -<flag1> -<flag2>\n");
    printf("flags: \n");
    printf("    -h for help\n");
    printf("    -lr sets learning rate\n");
    printf("    -batch_size sets batch size\n");
    printf("    -iterations sets number of training iterations\n");
    printf("    -display <num1> <num2> displays digits and their predictions from the test dataset\n");
    printf("ensure 0 < num1 < num2 < 1000 as there are 1000 test digits\n");
    printf("    -nodisplay turns automatic display examples off\n");
    printf("    -status_interval sets the interval that training accuracy will be displayed\n");
    printf("    -num_hidden_1 sets the number of nodes in the first hidden layer (default 30)\n");
    printf("    -num_hidden_2 sets the number of nodes in the second hidden layer (default 20)\n");
    printf("Example usage: ./mnist -lr 0.05 -batch_size 20 -status_interval 200 -display 200 220\n");
    exit(code);
}

Preferences *get_input(int argc, char *argv[]) {
    Preferences *preferences = malloc(sizeof(Preferences)); 
    preferences->lr = 0.1;
    preferences->batch_size = 100;
    preferences->num_iterations = 10000;
    preferences->display_start = 100;
    preferences->display_end = 105;
    preferences->status_interval = 100;
    preferences->num_hidden_1 = 30;
    preferences->num_hidden_2 = 20;

    char arg[256];
    for (int i = 1; i < argc; i++) {
        strcpy(arg, argv[i]);
        if (strcmp("-h", arg) == 0) {
            usage(0);
        }
        else if (strcmp("-lr", arg) == 0) {
            i++;
            preferences->lr = atof(argv[i]);
        }
        else if (strcmp("-batch_size", arg) == 0) {
            i++; 
            preferences->batch_size = atoi(argv[i]);
        }
        else if (strcmp("-iterations", arg) == 0) {
            i++;
            preferences->num_iterations = atoi(argv[i]);
        }
        else if (strcmp("-display", arg) == 0) {
            i++;
            preferences->display_start = atoi(argv[i]);
            i++;
            preferences->display_end = atoi(argv[i]);
        }
        else if (strcmp("-nodisplay", arg) == 0) {
            preferences->display_start = 0;
            preferences->display_end = 0;
        }
        else if (strcmp("-status_interval", arg) == 0) {
            i++;
            preferences->status_interval = atoi(argv[i]);
        }
        else if (strcmp("-num_hidden_1", arg) == 0) {
            i++;
            preferences->num_hidden_1 = atoi(argv[i]);
        }
        else if (strcmp("-num_hidden_2", arg) == 0) {
            i++;
            preferences->num_hidden_2 = atoi(argv[i]);
        }
        else usage(1);
    }

    if (!(preferences->display_start <= preferences->display_end && preferences->display_start >= 0 && preferences->display_end < 1000)) {
        usage(1);
    }
    return preferences;
}
void free_nodes(Nodes *nodes) {
    free_matrix_struct(nodes->Z1);
    free_matrix_struct(nodes->Z2);
    free_matrix_struct(nodes->Z3);
    free_matrix_struct(nodes->A1);
    free_matrix_struct(nodes->A2);
    free_matrix_struct(nodes->A3);
    free(nodes);
}

void free_misc(Misc *misc) {
    free_matrix_struct(misc->one_hot_Y);
    free_matrix_struct(misc->A2T);
    free_matrix_struct(misc->W3T);
    free_matrix_struct(misc->A1T);
    free_matrix_struct(misc->W2T);
    free_matrix_struct(misc->XT);
}

void free_deltas(Deltas *deltas) {
    free_matrix_struct(deltas->dZ3);
    free_matrix_struct(deltas->dW3);
    free_matrix_struct(deltas->dA2);
    free_matrix_struct(deltas->dZ2);
    free_matrix_struct(deltas->dA2_dZ2);
    free_matrix_struct(deltas->dW2);
    free_matrix_struct(deltas->dA1);
    free_matrix_struct(deltas->dZ1);
    free_matrix_struct(deltas->dA1_dZ1);
    free_matrix_struct(deltas->dW1);
}    

void init_misc(Misc *misc, Nodes *nodes, int batch_size, Weights *weights, Fmatrix *X) {
    misc->one_hot_Y = allocate_matrix(10, batch_size);
    misc->A2T = allocate_matrix(nodes->A2->ncols, nodes->A2->nrows);
    misc->W3T = allocate_matrix(weights->W3->ncols, weights->W3->nrows);
    misc->A1T = allocate_matrix(nodes->A1->ncols, nodes->A1->nrows);
    misc->W2T = allocate_matrix(weights->W2->ncols, weights->W2->nrows);
    misc->XT = allocate_matrix(X->ncols, X->nrows);
}

Nodes *init_nodes(Fmatrix *X, Weights *weights) {
    Nodes *nodes = malloc(sizeof(Nodes));

    // nodes in layers A1 & A2 have an extra row added to them (which will be set to 1's later) as a factor to the bias terms in the next layer's weights
    int ncols = X->ncols;
    nodes->Z1 = allocate_matrix(weights->W1->nrows, ncols);
    nodes->A1 = allocate_matrix(weights->W1->nrows + 1, ncols);
    nodes->Z2 = allocate_matrix(weights->W2->nrows, ncols);
    nodes->A2 = allocate_matrix(weights->W2->nrows + 1, ncols);
    nodes->Z3 = allocate_matrix(weights->W3->nrows, ncols);
    nodes->A3 = allocate_matrix(weights->W3->nrows, ncols);

    return nodes;
}

void free_matrix_arr(Fmatrix *arr) {
    free(arr->mat);
}

void free_matrix_struct(Fmatrix *arr) {
    free(arr->mat);
    free(arr);
}

void init_deltas(Deltas *deltas, Nodes *nodes, Weights *weights, Fmatrix* X) {
    deltas->dZ3 = allocate_matrix(nodes->A3->nrows, nodes->A3->ncols);
    deltas->dW3 = allocate_matrix(deltas->dZ3->nrows, nodes->A2->nrows); // ie A2T->ncols
    deltas->dA2 = allocate_matrix(weights->W3->ncols, deltas->dZ3->ncols);
    deltas->dZ2 = allocate_matrix(deltas->dA2->nrows - 1, deltas->dA2->ncols);
    deltas->dA2_dZ2 = allocate_matrix(nodes->Z2->nrows, nodes->Z2->ncols);
    deltas->dW2 = allocate_matrix(deltas->dZ2->nrows, nodes->A1->nrows); // ie A1T->ncols
    deltas->dA1 = allocate_matrix(weights->W2->ncols, deltas->dZ2->ncols);
    deltas->dZ1 = allocate_matrix(deltas->dA1->nrows - 1, deltas->dA1->ncols);
    deltas->dA1_dZ1 = allocate_matrix(nodes->Z1->nrows, nodes->Z1->ncols);
    deltas->dW1 = allocate_matrix(deltas->dZ1->nrows, X->nrows); // ie XT.ncols
}

void split_data(Fmatrix *data, Fmatrix* X_train, Fmatrix *Y_train, Fmatrix *X_test, Fmatrix *Y_test) {
    /*
      Each row of data matrix (60000 x 785) is a class label concatenated with an image vector
      Each column of the X_train (785 x 59000) and X_test (785 x 1000) matrices will be an image vector, with one extra element allocated on the end for a bias factor
      The corresponding columns of Y_train (1 x 59000) and Y_test (1 x 1000) matrices hold the images' labels
    */
    X_test->nrows = X_train->nrows = data->ncols;
    X_test->ncols = 1000;
    X_train->ncols = 59000; 

    Y_train->nrows = Y_test->nrows = 1;
    Y_test->ncols = 1000;
    Y_train->ncols = 59000;

    // dynamically allocate 2d arrays for each Matrix struct
    X_test->mat = initialize_array(X_test->nrows, X_test->ncols);
    X_train->mat = initialize_array(X_train->nrows, X_train->ncols);
    Y_train->mat = initialize_array(Y_train->nrows, Y_train->ncols);
    Y_test->mat = initialize_array(Y_test->nrows, Y_test->ncols);

    // assign values from original data matrix into splits
    for (int j = 0; j < Y_test->ncols; j++) {
        Y_test->mat[j] = data->mat[j * data->ncols];
    }

    for (int j = Y_test->ncols; j < Y_test->ncols + Y_train->ncols; j++) {
        Y_train->mat[j - Y_test->ncols] = data->mat[j * data->ncols];
    }

    for (int i = 1; i < X_test->nrows; i++) {
        for (int j = 0; j < X_test->ncols; j++) {
            X_test->mat[ (i - 1) * X_test->ncols + j] = data->mat[j * data->ncols + i];
        }
    }

    for (int i = 1; i < X_train->nrows; i++) {
        for (int j = X_test->ncols; j < X_test->ncols + X_train->ncols; j++) {
            X_train->mat[ (i - 1) * X_train->ncols + j - X_test->ncols] = data->mat[j * data->ncols + i];
        }
    }

    // free the original data matrix
    free_matrix_arr(data);
}

void free_matrix_structs(Fmatrix *test_yhat, Fmatrix *train_yhat, Fmatrix *X_batch, Fmatrix *Y_batch, Fmatrix *W1, Fmatrix *W2, Fmatrix *W3) {
    free_matrix_struct(test_yhat);
    free_matrix_struct(train_yhat);
    free_matrix_struct(X_batch);
    free_matrix_struct(Y_batch);
    free_matrix_struct(W1);
    free_matrix_struct(W2);
    free_matrix_struct(W3);
}

void free_matrix_arrays(Fmatrix *X_test, Fmatrix *X_train, Fmatrix *Y_test, Fmatrix *Y_train) {
    free_matrix_arr(X_test);
    free_matrix_arr(X_train);
    free_matrix_arr(Y_test);
    free_matrix_arr(Y_train);
}

void free_all_nodes(Nodes *nodes_train, Nodes *nodes_test, Nodes *nodes_batch) {
    free_nodes(nodes_train);
    free_nodes(nodes_test);
    free_nodes(nodes_batch);
}

void get_next_batch(int i, int batch_size, Fmatrix *X_train, Fmatrix *Y_train, Fmatrix *X_batch, Fmatrix *Y_batch) {
    // calculate next batch index
    int start_idx = ((i + 1) * batch_size) % X_train->ncols;
    int end_idx = ((i + 2) * batch_size) % X_train->ncols;
    if (end_idx == 0) end_idx = X_train->ncols;

    // copy the calculated indices into the next training batch
    copy_some_matrix_values(X_train, X_batch, start_idx, end_idx, true);
    copy_some_matrix_values(Y_train, Y_batch, start_idx, end_idx, true);
}

inline float random_float() {
    return ((float)rand() / (float)RAND_MAX);
}

void randomize_weights_He(Fmatrix *W, int fan_in) {
    // Initialize weight values according to He initialization (normal distribution with 0 mean and small variance determined by the number of incoming weights)
    float std_dev = sqrt(2.0 / fan_in);
    int size = W->nrows * W->ncols;
    for (int i = 0; i < size; i++) {
        float u1 = random_float(); 
        float u2 = random_float();
        float z1 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        W->mat[i] = z1 * std_dev;
    }
}

void append_bias_factor(Fmatrix *A) {
    float *Ap = &A->mat[(A->nrows - 1) * A->ncols];
    for (int j = 0; j < A->ncols; j++) {
        *Ap++ = 1;
    }
}

void print_accuracy(int i, Nodes *nodes_train, Nodes *nodes_test, Fmatrix *X_train, Fmatrix *X_test, Fmatrix *train_yhat, Fmatrix *test_yhat, Fmatrix *Y_train, Fmatrix *Y_test, Weights *weights) {
    forward_pass(nodes_train, X_train, weights);
    forward_pass(nodes_test, X_test, weights);
    argmax_into_yhat(nodes_train->A3, train_yhat);
    argmax_into_yhat(nodes_test->A3, test_yhat);
    printf("Iteration: %.4d | Train Accuracy: %.3f, Test Accuracy: %.3f\n", i, get_accuracy(train_yhat, Y_train), get_accuracy(test_yhat, Y_test));
}

float get_accuracy(Fmatrix *yhat, Fmatrix *Y) {
    float correct_sum = 0;
    float *YhatP = &yhat->mat[0];
    float *Yp = &Y->mat[0];

    for (int i = 0; i < Y->ncols; i++) {
        if (*YhatP++ == *Yp++) {
            correct_sum += 1;
        }
    }

    return correct_sum / Y->ncols;
}

void inference_one_example(Fmatrix *X_test, Fmatrix *Y_test, Weights *weights, int index) {
    Fmatrix *X_example = allocate_matrix(X_test->nrows, 1);
    copy_some_matrix_values(X_test, X_example, index, index + 1, false);

    Fmatrix *yhat = allocate_matrix(1, 1);
    Nodes *nodes = init_nodes(X_example, weights);
    Deltas deltas;
    init_deltas(&deltas, nodes, weights, X_example);
    Misc misc;
    init_misc(&misc, nodes, 1, weights, X_example);
    forward_pass(nodes, X_example, weights);
    argmax_into_yhat(nodes->A3, yhat);
    display_matrix(X_example);
    printf("Actual: %d, Predicted: %d at index %d\n", (int)Y_test->mat[index], (int) yhat->mat[0], index);

    free_matrix_struct(X_example);
    free_matrix_struct(yhat);
    free_deltas(&deltas);
    free_nodes(nodes);
    free_misc(&misc);
}

void display_examples(int display_start, int display_end, Fmatrix *X_test, Fmatrix *Y_test, Weights *weights) {
    for (int index = display_start; index < display_end; index++) {
        inference_one_example(X_test, Y_test, weights, index);
    }
}

void init_weights(Weights *weights, int num_input, int num_hidden_1, int num_hidden_2, int num_output) {
    srand(time(NULL)); // ensures random weight values

    // allocate and initiliaze weights to random values
    weights->W1 = allocate_matrix(num_hidden_1, num_input);
    weights->W2 = allocate_matrix(num_hidden_2, num_hidden_1 + 1);
    weights->W3 = allocate_matrix(num_output, num_hidden_2 + 1);

    randomize_weights_He(weights->W1, num_input);
    randomize_weights_He(weights->W2, num_hidden_1);
    randomize_weights_He(weights->W3, num_hidden_2);
}

void display_times(float alloc_time, float train_time, float inference_time) {
    printf("Allocation took %.4f seconds to execute\n", alloc_time);
    printf("Training took %.4f seconds to execute\n", train_time);
    printf("Inference time for entire training set (784 pixels x 59000 examples): %.4lf seconds\n", inference_time);

    FILE *file = fopen("stats.txt", "w");

    fprintf(file, "%.4f\n", alloc_time);
    fprintf(file, "%.4f\n", train_time);
    fprintf(file, "%.4f\n", inference_time);

    fclose(file);
}
