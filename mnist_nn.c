#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mnist.h"


float random_float() {
    return ((float)rand() / (float)RAND_MAX);
}

void init_weights(Matrix *W) {
    // Initialize every value of each matrix according to a uniform distribution on (-0.5, 0.5)
    for (int i = 0; i < W->nrows; i++) {
        W->mat[i] = malloc(W->ncols * sizeof(float));
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

void forward_pass(Layers *layers, Matrix *X, Matrix *W1, Matrix *W2, Matrix *W3) { 
    
    // layer 1
    multiply_matrices(W1, X, layers->Z1);
    copy_matrix_values(layers->Z1, layers->A1); 
    append_bias_factor(layers->A1);
    relu(layers->A1);

    // layer 2

    multiply_matrices(W2, layers->A1, layers->Z2);
    copy_matrix_values(layers->Z2, layers->A2);
    append_bias_factor(layers->A2);
    relu(layers->A2);

    // layer 3 (output)
    multiply_matrices(W3, layers->A2, layers->Z3);
    copy_matrix_values(layers->Z3, layers->A3);
    softmax(layers->A3);
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

void backward_pass(Matrix *X, Layers *layers, Matrix *W2, Matrix *W3, Matrix *Y, Deltas *deltas, Transpose *transpose) {
    one_hot(Y, transpose->one_hot_Y);
    subtract_matrices(layers->A3, transpose->one_hot_Y, deltas->dZ3); // dL/dZ3 = A3 - one_hot_Y
    transpose_matrix(layers->A2, transpose->A2T);               
    // divide matrix below by number of patterns ie batch size here
    multiply_matrices(deltas->dZ3, transpose->A2T, deltas->dW3);                      // dL/dW3 = dL/dZ3 · A2T
    scale_matrix(deltas->dW3, (float) 1 / Y->ncols );
    transpose_matrix(W3, transpose->W3T); 
    multiply_matrices(transpose->W3T, deltas->dZ3, deltas->dA2);                       // dL/dA2 = W3T · dL/dZ3
    deriv_relu(layers->Z2, deltas->dA2_dZ2);      
    multiply_matrices_elementwise(deltas->dA2_dZ2, deltas->dA2, deltas->dZ2, true);         // dL/dZ2 = dA2/dZ2 * dL/dA2
    transpose_matrix(layers->A1, transpose->A1T);
    // divide matrix below by number of patterns in batch size here
    multiply_matrices(deltas->dZ2, transpose->A1T, deltas->dW2);                      // dL/dW2 = dZ2/dW2 · dL/dZ2
    scale_matrix(deltas->dW2, (float) 1 / Y->ncols);
    transpose_matrix(W2, transpose->W2T);
    multiply_matrices(transpose->W2T, deltas->dZ2, deltas->dA1);                       // dL/dA1 = dZ2/dA1 · dL/dZ2 = W2T · dL/dZ2
    deriv_relu(layers->Z1, deltas->dA1_dZ1); 
    multiply_matrices_elementwise(deltas->dA1_dZ1, deltas->dA1, deltas->dZ1, true); // dL/dZ1 = dA1/dZ1 * dL/dA1

    transpose_matrix(X, transpose->XT);
    // divide matrix below by number of patterns in bach size here
    multiply_matrices(deltas->dZ1, transpose->XT, deltas->dW1);
    scale_matrix(deltas->dW1, (float) 1 / Y->ncols);
}

void update_weights(Deltas *deltas, Matrix *W1, Matrix *W2, Matrix *W3, float lr) {
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

void inference_one_example(Matrix *X_test, Matrix *Y_test, Matrix *W1, Matrix *W2, Matrix *W3, int index) {
    Matrix *X_example = allocate_matrix(X_test->nrows, 1);
    copy_some_matrix_values(X_test, X_example, index, index + 1, false);

    Matrix *yhat = allocate_matrix(1, 1);
    Layers *layers = init_layers(X_example, W1, W2, W3);
    Deltas deltas; 
    init_deltas(&deltas, layers, W2, W3, X_example);
    Transpose transpose;
    init_transpose(&transpose, layers, 1, W2, W3, X_example);
    forward_pass(layers, X_example, W1, W2, W3);
    argmax_into_yhat(layers->A3, yhat);
    display_matrix(X_example);
    printf("Actual: %d, Predicted: %d at index %d\n", (int)Y_test->mat[0][index], (int) yhat->mat[0][0], index);
    
    free_matrix_struct(X_example);
    free_matrix_struct(yhat);
    free_deltas(&deltas);
    free_layers(layers);
    free_transpose(&transpose);
}

int main(int argc, char *argv[]) {
    srand(time(NULL)); // ensures random weight initialization
    Preferences *preferences = get_input(argc, argv);
    clock_t start, end;

    // read in & prepare data (transpose, train/test split, x/y split, normalize x values) 
    start = clock();
    Matrix data = read_csv("MNIST_data.csv");
    Matrix test_data = { .nrows = 785, .ncols = 1000, .mat = malloc(785 * sizeof(float *)) };
    Matrix train_data = { .nrows = 785, .ncols = 41000, .mat = malloc(785 * sizeof(float *)) }; // 41999
    train_test_split(&data, &test_data, &train_data);
    Matrix X_train, X_test;
    Matrix Y_train, Y_test;
    XY_split(&test_data, &X_test, &Y_test);
    XY_split(&train_data, &X_train, &Y_train);
    normalize(&X_train, &X_test);
    append_bias_input(&X_train, &X_test);

    // initialize weights with an extra term for each node acting as bias 
    Matrix W1 = {.nrows = 30, .ncols = X_train.nrows, .mat = malloc(30 * sizeof(float*))};
    Matrix W2 = {.nrows = 20, .ncols = 30 + 1, .mat = malloc(20 * sizeof(float*))};
    Matrix W3 = {.nrows = 10, .ncols = 20 + 1, .mat = malloc(10 * sizeof(float*))};
    init_weights(&W1);
    init_weights(&W2);
    init_weights(&W3);

    // gets a subset of the data to train on 
    Matrix X_in = { .nrows = 785, .ncols = preferences->batch_size, .mat = malloc(785 * sizeof(float *)) };
    Matrix Y_in = { .nrows = 1, .ncols = preferences->batch_size, .mat = malloc(sizeof(float *)) }; 
    Matrix yhat = { .nrows = 10, .ncols = preferences->batch_size, .mat = malloc(10 * sizeof(float *))};
    for (int i = 0; i < X_in.nrows; i++) {
        X_in.mat[i] = malloc(X_in.ncols * sizeof(float));
    }
    for (int i = 0; i < Y_in.nrows; i++) {
        Y_in.mat[i] = malloc(Y_in.ncols * sizeof(float));
    }
    for (int i = 0; i < yhat.nrows; i++) {
        yhat.mat[i] = malloc(yhat.ncols * sizeof(float));
    }
    copy_some_matrix_values(&X_train, &X_in, 0, preferences->batch_size, false);
    copy_some_matrix_values(&Y_train, &Y_in, 0, preferences->batch_size, false);

    // initialize layers and derivatives
    Layers *layers = init_layers(&X_in, &W1, &W2, &W3);
    Deltas deltas; 
    init_deltas(&deltas, layers, &W2, &W3, &X_in);
    Transpose transpose;
    init_transpose(&transpose, layers, preferences->batch_size, &W2, &W3, &X_in);
    Layers *layers_eval = init_layers(&X_train, &W1, &W2, &W3);
    Layers *layers_test = init_layers(&X_test, &W1, &W2, &W3);
    Matrix *eval_yhat = allocate_matrix(10, Y_train.ncols);
    Matrix *test_yhat = allocate_matrix(10, Y_test.ncols);
    end = clock();
    printf("Allocation took %f seconds to execute\n", ((double) (end - start)) / CLOCKS_PER_SEC);

    // main training loop
    start = clock();
    int start_idx, end_idx;
    for (int i = 0; i < preferences->num_iterations; i++) {
        forward_pass(layers, &X_in, &W1, &W2, &W3);
        backward_pass(&X_in, layers, &W2, &W3, &Y_in, &deltas, &transpose);  
        update_weights(&deltas, &W1, &W2, &W3, preferences->lr);
        if (i % preferences->status_interval == 0) {
            forward_pass(layers_eval, &X_train, &W1, &W2, &W3);
            forward_pass(layers_test, &X_test, &W1, &W2, &W3);
            argmax_into_yhat(layers_eval->A3, eval_yhat);
            argmax_into_yhat(layers_test->A3, test_yhat);
            printf("Iteration: %d | Train Accuracy: %f, Test Accuracy: %f\n", i, get_accuracy(eval_yhat, &Y_train), get_accuracy(test_yhat, &Y_test));
        }

        start_idx = ((i + 1) * preferences->batch_size) % X_train.ncols;
        end_idx = ((i + 2) * preferences->batch_size) % X_train.ncols;
        if (end_idx == 0) {
            end_idx = X_train.ncols;
        }
        copy_some_matrix_values(&X_train, &X_in, start_idx, end_idx, true);   
        copy_some_matrix_values(&Y_train, &Y_in, start_idx, end_idx, true);   
    }
    end = clock();
    printf("Non-allocation operations of program (training) took %f seconds to execute\n", ((double) (end - start)) / CLOCKS_PER_SEC);

    // recording forward pass time for comparison - not essential to the program
    start = clock();
    forward_pass(layers_eval, &X_train, &W1, &W2, &W3);
    end = clock();
    printf("Inference time for entire training set (784 pixels x 41000 examples): %lf seconds\n", ((double) (end - start)) / CLOCKS_PER_SEC);
    
    // displaying some examples
    for (int index = preferences->display_start; index < preferences->display_end; index++) {
        inference_one_example(&X_test, &Y_test, &W1, &W2, &W3, index);
    }

    // cleanup
    free_matrix_struct(test_yhat);
    free_matrix_struct(eval_yhat);
    free_layers(layers_eval);
    free_layers(layers_test);
    free_matrix_arr(X_in);
    free_matrix_arr(Y_in);
    free_layers(layers);
    free_transpose(&transpose);
    free_matrix_arr(X_test);
    free_matrix_arr(X_train);
    free_matrix_arr(Y_test);
    free_matrix_arr(Y_train);
    free_matrix_arr(yhat);
    free_matrix_arr(W1);
    free_matrix_arr(W2);
    free_matrix_arr(W3);
    free_deltas(&deltas);
    free(preferences);
    printf("All memory frees successful\n");
    return 0;
}

