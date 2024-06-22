#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mnist.h"

float random_float() {
    return ((float)rand() / (float)RAND_MAX);
}

void randomize_weights(Matrix *W) {
    // Initialize every value of each matrix according to a uniform distribution on (-0.5, 0.5)
    for (int i = 0; i < W->nrows; i++) {
        for (int j = 0; j < W->ncols; j++) {
            W->mat[i][j] = random_float() - 0.5;
        }
    }
}

void relu(Matrix *Z) {
    for (int i = 0; i < Z->nrows; i++) {
        for (int j = 0; j < Z->ncols; j++) {
            if (Z->mat[i][j] < 0) Z->mat[i][j] = 0;
        }
    }
}

void softmax(Matrix *Z) {
    // define a matrix of exponentials of the Z matrix
    float exp_matrix[Z->nrows][Z->ncols]; 
    for (int i = 0; i < Z->nrows; i++) {
       for (int j = 0; j < Z->ncols; j++) {
            exp_matrix[i][j] = exp(Z->mat[i][j]);
        }
    }

    // for each column, sum the exponential matrix values
    // then divide each element of the col by the sum and store it
    for (int j = 0; j < Z->ncols; j++) {
        float col_sum = 0; 
        for (int i = 0; i < Z->nrows; i++) {
           col_sum += exp_matrix[i][j];
        }
        for (int i = 0; i < Z->nrows; i++) {
           Z->mat[i][j] = exp_matrix[i][j] / col_sum;
        }
   }
}

void append_bias_factor(Matrix *A) {
    for (int j = 0; j < A->ncols; j++) {
        A->mat[A->nrows - 1][j] = 1; 
    }
}

void forward_pass(Nodes *nodes, Matrix *X, Weights *weights) { 
    
    // layer 1
    multiply_matrices(weights->W1, X, nodes->Z1);
    copy_matrix_values(nodes->Z1, nodes->A1); 
    append_bias_factor(nodes->A1);
    relu(nodes->A1);

    // layer 2
    multiply_matrices(weights->W2, nodes->A1, nodes->Z2);
    copy_matrix_values(nodes->Z2, nodes->A2);
    append_bias_factor(nodes->A2);
    relu(nodes->A2);

    // layer 3 (output)
    multiply_matrices(weights->W3, nodes->A2, nodes->Z3);
    copy_matrix_values(nodes->Z3, nodes->A3);
    softmax(nodes->A3);
}

void deriv_relu(Matrix *Z, Matrix *derivative) {
    for (int i = 0; i < Z->nrows; i++) {
        for (int j = 0; j < Z->ncols; j++) {
            if (Z->mat[i][j] > 0) 
                derivative->mat[i][j] = 1;
            else
                derivative->mat[i][j] = 0;
            }
    }
}

void backward_pass(Matrix *X, Nodes *nodes, Weights *weights, Matrix *Y, Deltas *deltas, Transpose *transpose) {
    one_hot(Y, transpose->one_hot_Y);
    subtract_matrices(nodes->A3, transpose->one_hot_Y, deltas->dZ3); // dL/dZ3 = A3 - one_hot_Y
    transpose_matrix(nodes->A2, transpose->A2T);               
    // divide matrix below by number of patterns ie batch size here
    multiply_matrices(deltas->dZ3, transpose->A2T, deltas->dW3);                      // dL/dW3 = dL/dZ3 · A2T
    scale_matrix(deltas->dW3, (float) 1 / Y->ncols );
    transpose_matrix(weights->W3, transpose->W3T); 
    multiply_matrices(transpose->W3T, deltas->dZ3, deltas->dA2);                       // dL/dA2 = W3T · dL/dZ3
    deriv_relu(nodes->Z2, deltas->dA2_dZ2);      
    multiply_matrices_elementwise(deltas->dA2_dZ2, deltas->dA2, deltas->dZ2, true);         // dL/dZ2 = dA2/dZ2 * dL/dA2
    transpose_matrix(nodes->A1, transpose->A1T);
    // divide matrix below by number of patterns in batch size here
    multiply_matrices(deltas->dZ2, transpose->A1T, deltas->dW2);                      // dL/dW2 = dZ2/dW2 · dL/dZ2
    scale_matrix(deltas->dW2, (float) 1 / Y->ncols);
    transpose_matrix(weights->W2, transpose->W2T);
    multiply_matrices(transpose->W2T, deltas->dZ2, deltas->dA1);                       // dL/dA1 = dZ2/dA1 · dL/dZ2 = W2T · dL/dZ2
    deriv_relu(nodes->Z1, deltas->dA1_dZ1); 
    multiply_matrices_elementwise(deltas->dA1_dZ1, deltas->dA1, deltas->dZ1, true); // dL/dZ1 = dA1/dZ1 * dL/dA1
    transpose_matrix(X, transpose->XT);
    // divide matrix below by number of patterns in bach size here
    multiply_matrices(deltas->dZ1, transpose->XT, deltas->dW1);
    scale_matrix(deltas->dW1, (float) 1 / Y->ncols);
}

void update_weights(Deltas *deltas, Weights *weights, float lr) {
    Matrix *W1 = weights->W1;
    Matrix *W2 = weights->W2;
    Matrix *W3 = weights->W3;
    for (int i = 0; i < W3->nrows; i++) {
        for (int j = 0; j < W3->ncols; j++) {
            W3->mat[i][j] -= lr * deltas->dW3->mat[i][j];
            deltas->dW3->mat[i][j] = 0;
        }
    }

    for (int i = 0; i < W2->nrows; i++) {
        for (int j = 0; j < W2->ncols; j++) {
            W2->mat[i][j] -= lr * deltas->dW2->mat[i][j];
            deltas->dW2->mat[i][j] = 0;
        }
    }

    for (int i = 0; i < W1->nrows; i++) {
        for (int j = 0; j < W1->ncols; j++) {
            W1->mat[i][j] -= lr * deltas->dW1->mat[i][j];
            deltas->dW1->mat[i][j] = 0;
        }
    }
}

float get_accuracy(Matrix *yhat, Matrix *Y) {
    float correct_sum = 0;
    for (int i = 0; i < Y->ncols; i++) {
        if (yhat->mat[0][i] == Y->mat[0][i]) {
            correct_sum += 1;
        }
    }
    return correct_sum / Y->ncols;
}

void inference_one_example(Matrix *X_test, Matrix *Y_test, Weights *weights, int index) {
    Matrix *X_example = allocate_matrix(X_test->nrows, 1);
    copy_some_matrix_values(X_test, X_example, index, index + 1, false);

    Matrix *yhat = allocate_matrix(1, 1);
    Nodes *nodes = init_nodes(X_example, weights);
    Deltas deltas; 
    init_deltas(&deltas, nodes, weights, X_example);
    Transpose transpose;
    init_transpose(&transpose, nodes, 1, weights, X_example);
    forward_pass(nodes, X_example, weights);
    argmax_into_yhat(nodes->A3, yhat);
    display_matrix(X_example);
    printf("Actual: %d, Predicted: %d at index %d\n", (int)Y_test->mat[0][index], (int) yhat->mat[0][0], index);
    
    free_matrix_struct(X_example);
    free_matrix_struct(yhat);
    free_deltas(&deltas);
    free_nodes(nodes);
    free_transpose(&transpose);
}

void init_weights(Weights *weights, int num_input, int num_hidden_1, int num_hidden_2, int num_output) {
    srand(time(NULL)); // ensures random weight values

    // allocate and initiliaze weights to random values
    /*
    Weights.W1 = {.nrows = 30, .ncols = X_train.nrows, .mat = malloc(30 * sizeof(float*))};
    Weights.W2 = {.nrows = 20, .ncols = 30 + 1, .mat = malloc(20 * sizeof(float*))};
    Weights.W3 = {.nrows = 10, .ncols = 20 + 1, .mat = malloc(10 * sizeof(float*))};
    */
    weights->W1 = allocate_matrix(num_hidden_1, num_input);
    weights->W2 = allocate_matrix(num_hidden_2, num_hidden_1 + 1);
    weights->W3 = allocate_matrix(num_output, num_hidden_2 + 1);

    randomize_weights(weights->W1);
    randomize_weights(weights->W2);
    randomize_weights(weights->W3);
}


int main(int argc, char *argv[]) {
    Preferences *preferences = get_input(argc, argv);
    clock_t start, end;
    start = clock();

    // read in & prepare data (transpose, train/test split, x/y split, normalize x values, append bias factor)
    Matrix data = read_csv("MNIST_data.csv");
    Matrix X_train, X_test, Y_train, Y_test;
    split_data(&data, &X_train, &Y_train, &X_test, &Y_test);
    normalize(&X_train, &X_test);
    append_bias_input(&X_train, &X_test);

    // initialize weights struct to hold each layer's weights
    Weights weights;
    init_weights(&weights, X_train.nrows, 30, 20, 10);

    // gets a batch of the data to begin training on 
    Matrix *X_batch = allocate_matrix(X_train.nrows, preferences->batch_size);
    Matrix *Y_batch = allocate_matrix(1, preferences->batch_size);
    copy_some_matrix_values(&X_train, X_batch, 0, preferences->batch_size, false);
    copy_some_matrix_values(&Y_train, Y_batch, 0, preferences->batch_size, false);

    // initialize nodes (Number of nodes in each layer will vary with the batch size) and deltas (derivatives) 
    Nodes *nodes_batch = init_nodes(X_batch, &weights);
    Deltas deltas; 
    init_deltas(&deltas, nodes_batch, &weights, X_batch); // only need to allocate deltas corresponding to batch dimensoins bc train & test are only used for inferencing
    Transpose transpose;
    init_transpose(&transpose, nodes_batch, preferences->batch_size, &weights, X_batch); // only need to allocate transpose in batch dims for same reason as above


    // nodes_train and nodes_test will be used for training and testing accuracy
    Nodes *nodes_train = init_nodes(&X_train, &weights);
    Nodes *nodes_test= init_nodes(&X_test, &weights);
    Matrix *train_yhat = allocate_matrix(10, Y_train.ncols);
    Matrix *test_yhat = allocate_matrix(10, Y_test.ncols);
    end = clock();
    printf("Allocation took %lf seconds to execute\n", ((double) (end - start)) / CLOCKS_PER_SEC);

    // main training loop
    start = clock();
    int start_idx, end_idx;
    for (int i = 0; i < preferences->num_iterations; i++) {
        forward_pass(nodes_batch, X_batch, &weights);
        backward_pass(X_batch, nodes_batch, &weights, Y_batch, &deltas, &transpose);  
        update_weights(&deltas, &weights, preferences->lr);
        if (i % preferences->status_interval == 0) {
            forward_pass(nodes_train, &X_train, &weights);
            forward_pass(nodes_test, &X_test, &weights);
            argmax_into_yhat(nodes_train->A3, train_yhat);
            argmax_into_yhat(nodes_test->A3, test_yhat);
            printf("Iteration: %d | Train Accuracy: %f, Test Accuracy: %f\n", i, get_accuracy(train_yhat, &Y_train), get_accuracy(test_yhat, &Y_test));
        }

        start_idx = ((i + 1) * preferences->batch_size) % X_train.ncols;
        end_idx = ((i + 2) * preferences->batch_size) % X_train.ncols;
        if (end_idx == 0) {
            end_idx = X_train.ncols;
        }
        copy_some_matrix_values(&X_train, X_batch, start_idx, end_idx, true);   
        copy_some_matrix_values(&Y_train, Y_batch, start_idx, end_idx, true);   
    }
    end = clock();
    printf("Non-allocation operations of program (training) took %f seconds to execute\n", ((double) (end - start)) / CLOCKS_PER_SEC);

    // recording forward pass time for comparison - not essential to the program
    start = clock();
    forward_pass(nodes_train, &X_train, &weights);
    end = clock();
    printf("Inference time for entire training set (784 pixels x 41000 examples): %lf seconds\n", ((double) (end - start)) / CLOCKS_PER_SEC);
    
    // displaying some examples
    for (int index = preferences->display_start; index < preferences->display_end; index++) {
        inference_one_example(&X_test, &Y_test, &weights, index);
    }

    // cleanup
    free_matrix_struct(test_yhat);
    free_matrix_struct(train_yhat);
    free_nodes(nodes_train);
    free_nodes(nodes_test);
    free_matrix_struct(X_batch);
    free_matrix_struct(Y_batch);
    free_nodes(nodes_batch);
    free_transpose(&transpose);
    free_matrix_arr(X_test);
    free_matrix_arr(X_train);
    free_matrix_arr(Y_test);
    free_matrix_arr(Y_train);
    free_matrix_struct(weights.W1);
    free_matrix_struct(weights.W2);
    free_matrix_struct(weights.W3);
    free_deltas(&deltas);
    free(preferences);
    printf("All memory frees successful\n");
    return 0;
}

