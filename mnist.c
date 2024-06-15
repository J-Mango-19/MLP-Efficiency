#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mnist.h"

void transpose_matrix(Matrix *arr, Matrix *transposed) {
    for (int j = 0; j < arr->ncols; j++) {
        for (int i = 0; i < arr->nrows; i++) {
            transposed->mat[j][i] = arr->mat[i][j];
        }
    }
}

void free_matrix_arr(Matrix arr) {
    for (int i = 0; i < arr.nrows; i++) {
        free(arr.mat[i]);
    }
    free(arr.mat);
}

void free_matrix_struct(Matrix *arr) {
    for (int i = 0; i < arr->nrows; i++) {
        free(arr->mat[i]);
    }
    free(arr->mat);
    free(arr);
}


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

void normalize(Matrix *X_train, Matrix *X_test) {
    for (int i = 0; i < X_train->nrows; i++) {

        // normalize every element of train along dimension 1
        for (int j = 0; j < X_train->ncols; j++) {
            X_train->mat[i][j] /= 255;
        }

        // normalize every element of test along dimension 1
        for (int j = 0; j < X_test->ncols; j++) {
            X_test->mat[i][j] /= 255;
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

Matrix *allocate_matrix(int nrows, int ncols) {
    Matrix *M = malloc(sizeof(Matrix));
    M->nrows = nrows;
    M->ncols = ncols;
    M->mat = malloc(nrows * sizeof(float *));
    for (int i = 0; i < nrows; i++) {
        M->mat[i] = malloc(ncols * sizeof(float));
    }
    return M;
}

void multiply_matrices(Matrix *A, Matrix *B, Matrix *C) {
    // reduce linked list operations by storing pointers
    float **Amat = A->mat;
    float **Bmat = B->mat;
    float **Cmat = C->mat;

    if (A->ncols != B->nrows) {
        fprintf(stderr, "Error! Factor matrix dimensions incompatible\n");
        fprintf(stderr, "A: (%d, %d), B: (%d, %d)\n", A->nrows, A->ncols, B->nrows, B->ncols);
        exit(1);
    }

    /*
    for (int i = 0; i < C->nrows; i++) {
        for (int j = 0; j < C->ncols; j++) {
            C->mat[i][j] = 0;
            for (int k = 0; k < A->ncols; k++) {
                C->mat[i][j] += A->mat[i][k] * B->mat[k][j];
            }
        }
    }
    */
    /*
    // new version testing
    float *irowc, *irowa;
    for (int i = 0; i < C->nrows; i++) {
        for (int j = 0; j < C->ncols; j++) {
            Cmat[i][j] = 0;
            irowc = Cmat[i];
            irowa = Amat[i];
            for (int k = 0; k < A->ncols; k++) {
                irowc[j] += irowa[k] * Bmat[k][j];
            }
        }
    }
    */
    // newest version ntesting
    // set the matrix to zero
    size_t rowsize = C->ncols * sizeof(float);
    for (int i = 0; i < C->nrows; i++) {
        memset(Cmat[i], 0, rowsize);
    }

    // perform matmul loop in different order
            for (int k = 0; k < A->ncols; k++) {
    for (int i = 0; i < C->nrows; i++) {
        for (int j = 0; j < C->ncols; j++) {
                Cmat[i][j] += Amat[i][k] * Bmat[k][j];
            }
        }
    }



}

// use this function to copy a new matrix that is larger than the original (for example, copying Z values into A matrices, which are larger to hold bias factor)
void copy_matrix_values(Matrix *original, Matrix *New) {
    for (int i = 0; i < original->nrows; i++) {
        for (int j = 0; j < original->ncols; j++) {
            New->mat[i][j] = original->mat[i][j];
        }
    }
}

// use this fxn to copy a subset of values from a larger matrix into a smaller one (for example, copying X values into a batch matrix)
void copy_some_matrix_values(Matrix *original, Matrix *New, int start_idx, int end_idx, bool allow_wrap) {
    bool execute_wrap = false;
    if (end_idx > original->ncols && !allow_wrap) {
        fprintf(stderr, "Error! attempted to copy a piece of a matrix out of original matrix dimensions");
        exit(1);
    }
    else if (end_idx - start_idx < 0 && allow_wrap) {
        execute_wrap = true; // the end index has looped around to the beginning of the original matrix, so we can't do a simple copy 
    }
    if (end_idx - start_idx != New->ncols && !allow_wrap) {
        printf("end_idx = %d, start_idx = %d, end-start = %d, != New->ncols = %d\n", end_idx, start_idx, end_idx-start_idx, New->ncols);
        fprintf(stderr, "Error! copy size != new matrix size");
        exit(1);
    }
    if (!execute_wrap) {
        for (int i = 0; i < New->nrows; i++) {
            for (int j = start_idx; j < end_idx; j++) {
                New->mat[i][j - start_idx] = original->mat[i][j];
            }
        }
    }
    else {
        // fill out the values that we can until the end
        int values_till_end = original->ncols - start_idx;
        for (int i = 0; i < New->nrows; i++) {
            for (int j = start_idx; j < New->ncols; j++) {
                New->mat[i][j - start_idx] = original->mat[i][j];
            }
        }

        // fill out the remaining values using the beginning of the original matrix
        for (int i = 0; i < New->nrows; i++) {
            for ( int j = 0; j < New->ncols - values_till_end; j++) {
                New->mat[i][j] = original->mat[i][j];
            }
        }
    }
}


void append_bias_factor(Matrix *A) {
    for (int j = 0; j < A->ncols; j++) {
        A->mat[A->nrows - 1][j] = 1; 
    }
}

// this is just for debugging
void get_matrix_stats(Matrix *problem) {
    float sum, max;
    for (int i = 0; i < problem->nrows; i++) {
        sum = 0;
        max = 0;
        for (int j = 0; j < problem->ncols; j++) {
            sum += problem->mat[i][j];
            if (problem->mat[i][j] > max) max = problem->mat[i][j];
        }
        printf("Average of row %d: %f\n", i, sum / problem->ncols);
        printf("Max of row %d i: %f\n", i, max);
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

    // debugging purposes
    /*
    float sum = 0;
    for (int i = 0; i < layers->A3->nrows; i++) {
        sum = 0;
        for (int j = 0; j < layers->A3->ncols; j++) {
            sum += layers->A3->mat[i][j];
        } 
            printf("avg of prediction %d: %f\n", i, sum / layers->A3->ncols);
    }
    */
}

Layers *init_layers(Matrix *X, Matrix *W1, Matrix *W2, Matrix *W3) {
    // layers A1 & A2 have an extra row added to them (which will be set to 1's later) as a factor to the bias terms in the next layer's weights
    Matrix *Z1 = allocate_matrix(W1->nrows, X->ncols);
    Matrix *A1 = allocate_matrix(W1->nrows + 1, X->ncols);
    Matrix *Z2 = allocate_matrix(W2->nrows, A1->ncols);
    Matrix *A2 = allocate_matrix(W2->nrows + 1, A1->ncols);
    Matrix *Z3 = allocate_matrix(W3->nrows, A2->ncols);
    Matrix *A3 = allocate_matrix(W3->nrows, A2->ncols);

    Layers *initialized_layers = malloc(sizeof(Layers));
    initialized_layers->Z1 = Z1;
    initialized_layers->A1 = A1;
    initialized_layers->Z2 = Z2;
    initialized_layers->A2 = A2;
    initialized_layers->Z3 = Z3;
    initialized_layers->A3 = A3;

    return initialized_layers;
}

void one_hot(Matrix *Y, Matrix* one_hot_Y) {
    for (int i = 0; i < one_hot_Y->nrows; i++) {
        for (int j = 0; j < one_hot_Y->ncols; j++) {
            if (i == Y->mat[0][j]) 
                one_hot_Y->mat[i][j] = 1;
            else
                one_hot_Y->mat[i][j] = 0;
        }
    }
}

void subtract_matrices(Matrix *A, Matrix *B, Matrix *C) {
    if (A->ncols != B->ncols || A->nrows != B->nrows) {
        fprintf(stderr, "Error! Subtraction matrix dimensions incompatible\n");
        fprintf(stderr, "A: (%d, %d), B: (%d, %d)\n", A->nrows, A->ncols, B->nrows, B->ncols);
        exit(1);
    }

    for (int i = 0; i < A->nrows; i++) {
        for (int j = 0; j < A->ncols; j++) {
            C->mat[i][j] = A->mat[i][j] - B->mat[i][j];
        }
    }
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

void multiply_matrices_elementwise(Matrix *A, Matrix *B, Matrix *C, bool omit_last_row) {
    // B is one bigger
    if (A->nrows != B->nrows - (int)omit_last_row || A->ncols != B->ncols) {
        fprintf(stderr, "Error! Elementwise multiplication matrix dimensions incompatible\n");
        exit(1);
    }

    for (int i = 0; i < C->nrows; i++) {
        for (int j = 0; j < C->ncols; j++) {
            C->mat[i][j] = A->mat[i][j] * B->mat[i][j];
        }
    }
}

void divide_matrix_elementwise(Matrix *matrix, int divisor) {
    for (int i = 0; i < matrix->nrows; i++) {
        for (int j = 0 ; j < matrix->ncols; j++) {
            matrix->mat[i][j] /= divisor;
        }
    }
}

void backward_pass(Matrix *X, Layers *layers, Matrix *W1, Matrix *W2, Matrix *W3, Matrix *Y, Deltas *deltas, Transpose *transpose) {
    one_hot(Y, transpose->one_hot_Y);
    subtract_matrices(layers->A3, transpose->one_hot_Y, deltas->dZ3); // dL/dZ3 = A3 - one_hot_Y
    transpose_matrix(layers->A2, transpose->A2T);               
    // divide matrix below by number of patterns ie batch size here
    multiply_matrices(deltas->dZ3, transpose->A2T, deltas->dW3);                      // dL/dW3 = dL/dZ3 · A2T
    divide_matrix_elementwise(deltas->dW3, Y->ncols);
    transpose_matrix(W3, transpose->W3T); 
    multiply_matrices(transpose->W3T, deltas->dZ3, deltas->dA2);                       // dL/dA2 = W3T · dL/dZ3
    deriv_relu(layers->Z2, deltas->dA2_dZ2);      
    multiply_matrices_elementwise(deltas->dA2_dZ2, deltas->dA2, deltas->dZ2, true);         // dL/dZ2 = dA2/dZ2 * dL/dA2
    transpose_matrix(layers->A1, transpose->A1T);
    // divide matrix below by number of patterns in batch size here
    multiply_matrices(deltas->dZ2, transpose->A1T, deltas->dW2);                      // dL/dW2 = dZ2/dW2 · dL/dZ2
    divide_matrix_elementwise(deltas->dW2, Y->ncols);
    transpose_matrix(W2, transpose->W2T);
    multiply_matrices(transpose->W2T, deltas->dZ2, deltas->dA1);                       // dL/dA1 = dZ2/dA1 · dL/dZ2 = W2T · dL/dZ2
    deriv_relu(layers->Z1, deltas->dA1_dZ1); 
    multiply_matrices_elementwise(deltas->dA1_dZ1, deltas->dA1, deltas->dZ1, true); // dL/dZ1 = dA1/dZ1 * dL/dA1

    transpose_matrix(X, transpose->XT);
    // divide matrix below by number of patterns in bach size here
    multiply_matrices(deltas->dZ1, transpose->XT, deltas->dW1);
    divide_matrix_elementwise(deltas->dW1, Y->ncols);
}

void init_deltas(Deltas *deltas, Layers *layers, Matrix *W1, Matrix *W2, Matrix *W3, Matrix* X) {
    deltas->dZ3 = allocate_matrix(layers->A3->nrows, layers->A3->ncols);
    deltas->dW3 = allocate_matrix(deltas->dZ3->nrows, layers->A2->nrows); // ie A2T->ncols
    deltas->dA2 = allocate_matrix(W3->ncols, deltas->dZ3->ncols);
    deltas->dZ2 = allocate_matrix(deltas->dA2->nrows - 1, deltas->dA2->ncols);
    deltas->dA2_dZ2 = allocate_matrix(layers->Z2->nrows, layers->Z2->ncols);
    deltas->dW2 = allocate_matrix(deltas->dZ2->nrows, layers->A1->nrows); // ie A1T->ncols
    deltas->dA1 = allocate_matrix(W2->ncols, deltas->dZ2->ncols);
    deltas->dZ1 = allocate_matrix(deltas->dA1->nrows - 1, deltas->dA1->ncols);
    deltas->dA1_dZ1 = allocate_matrix(layers->Z1->nrows, layers->Z1->ncols);
    deltas->dW1 = allocate_matrix(deltas->dZ1->nrows, X->nrows); // ie XT.ncols
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

void get_yhat(Matrix *A, Matrix *yhat) {
    float max;
    int max_idx = 0;
    for (int j = 0; j < A->ncols; j++) {
        max = 0;
        for (int i = 0; i < A->nrows; i++) {
            if (A->mat[i][j] > max) {
                max = A->mat[i][j];
                max_idx = i;
            }
        }
        yhat->mat[0][j] = max_idx;
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

void init_transpose(Transpose *transpose, Layers *layers, int batch_size, Matrix *Y, Matrix *W2, Matrix *W3, Matrix *X) {
    transpose->one_hot_Y = allocate_matrix(10, batch_size);
    transpose->A2T = allocate_matrix(layers->A2->ncols, layers->A2->nrows);
    transpose->W3T = allocate_matrix(W3->ncols, W3->nrows);
    transpose->A1T = allocate_matrix(layers->A1->ncols, layers->A1->nrows);
    transpose->W2T = allocate_matrix(W2->ncols, W2->nrows);
    transpose->XT = allocate_matrix(X->ncols, X->nrows);
}

void inference_one_example(Matrix *X_test, Matrix *Y_test, Matrix *W1, Matrix *W2, Matrix *W3, int index) {
    Matrix *X_example = allocate_matrix(X_test->nrows, 1);
    copy_some_matrix_values(X_test, X_example, index, index + 1, false);
    if (index == 0)
        get_matrix_stats(X_example);

    Matrix *yhat = allocate_matrix(1, 1);
    Layers *layers = init_layers(X_example, W1, W2, W3);
    Deltas deltas; 
    init_deltas(&deltas, layers, W1, W2, W3, X_example);
    Transpose transpose;
    init_transpose(&transpose, layers, 1, yhat, W2, W3, X_example);
    forward_pass(layers, X_example, W1, W2, W3);
    get_yhat(layers->A3, yhat);
    display_matrix(X_example);
    printf("Actual: %d, Predicted: %d at index %d\n", (int)Y_test->mat[0][index], (int) yhat->mat[0][0], index);
    
    free_matrix_struct(X_example);
    free_matrix_struct(yhat);
    free_deltas(&deltas);
    free_layers(layers);
    free_transpose(&transpose);
}

void display_matrix(Matrix *X) {
   printf("\n\n"); 
    for (int i = 0; i < X->nrows - 1; i++) {
        if (i % 28 == 0) printf("\n");
        if (X->mat[i][0] * 255 > 200) printf("*");
        else if (X->mat[i][0] * 255 > 150) printf("+");
        else if (X->mat[i][0] * 255 > 100) printf("-");
        else printf(" ");
    }
}

void free_layers(Layers *layers) {
    free_matrix_struct(layers->Z1);
    free_matrix_struct(layers->Z2);
    free_matrix_struct(layers->Z3);
    free_matrix_struct(layers->A1);
    free_matrix_struct(layers->A2);
    free_matrix_struct(layers->A3);
    free(layers);
}

void free_transpose(Transpose *transpose) {
    free_matrix_struct(transpose->one_hot_Y);
    free_matrix_struct(transpose->A2T);
    free_matrix_struct(transpose->W3T);
    free_matrix_struct(transpose->A1T);
    free_matrix_struct(transpose->W2T);
    free_matrix_struct(transpose->XT);
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

int main(int argc, char *argv[]) {
    srand(time(NULL)); // ensures random weight initialization
    Preferences *preferences = get_input(argc, argv);
    clock_t start_operations, end_operations, start_inference, end_inference;
    double main_cpu_time;
    clock_t start, end;
    double cpu_time_used; 



    // read in & prepare data (transpose, train/test split, x/y split, normalize x values) 
    start = clock();
    Matrix data = read_csv("MNIST_train.csv");
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
    init_deltas(&deltas, layers, &W1, &W2, &W3, &X_in);
    Transpose transpose;
    init_transpose(&transpose, layers, preferences->batch_size, &Y_in, &W2, &W3, &X_in);
    Layers *layers_eval = init_layers(&X_train, &W1, &W2, &W3);
    Layers *layers_test = init_layers(&X_test, &W1, &W2, &W3);
    Matrix *eval_yhat = allocate_matrix(10, Y_train.ncols);
    Matrix *test_yhat = allocate_matrix(10, Y_test.ncols);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Allocation took %f seconds to execute\n", cpu_time_used);

    start_operations = clock();
    int start_idx, end_idx;
    for (int i = 0; i < preferences->num_iterations; i++) {
        forward_pass(layers, &X_in, &W1, &W2, &W3);
        backward_pass(&X_in, layers, &W1, &W2, &W3, &Y_in, &deltas, &transpose);  
        update_weights(&deltas, &W1, &W2, &W3, preferences->lr);
        if (i % preferences->status_interval == 0) {
            forward_pass(layers_eval, &X_train, &W1, &W2, &W3);
            forward_pass(layers_test, &X_test, &W1, &W2, &W3);
            get_yhat(layers_eval->A3, eval_yhat);
            get_yhat(layers_test->A3, test_yhat);
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

    end_operations = clock();
    main_cpu_time = ((double) (end_operations - start_operations)) / CLOCKS_PER_SEC;
    printf("Non-allocation operations of program (training) took %f seconds to execute\n", main_cpu_time);

    start_inference = clock();
    forward_pass(layers_eval, &X_train, &W1, &W2, &W3);
    end_inference = clock();
    printf("Inference time for entire training set (784 pixels x 41000 examples): %lf seconds\n", ((double) (end_inference - start_inference)) / CLOCKS_PER_SEC);
    
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

