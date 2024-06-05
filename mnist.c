#include "mnist.h"

Matrix transpose_matrix(Matrix *arr) {
    float **new_matrix = calloc(arr->ncols, sizeof(float*));
    for (int j = 0; j < arr->ncols; j++) {
        new_matrix[j] = calloc(arr->nrows, sizeof(float));
        for (int i = 0; i < arr->nrows; i++) {
            new_matrix[j][i] = arr->mat[i][j];
        }
    }
    Matrix transposed = { .nrows = arr->ncols, .ncols = arr->nrows, .mat = new_matrix };
    return transposed;
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
        W->mat[i] = calloc(W->ncols, sizeof(float));
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
    M->mat = calloc(nrows, sizeof(float *));
    for (int i = 0; i < nrows; i++) {
        M->mat[i] = calloc(ncols, sizeof(float));
    }
    return M;
}

void multiply_matrices(Matrix *A, Matrix *B, Matrix *C) {
    if (A->ncols != B->nrows) {
        fprintf(stderr, "Error! Factor matrix dimensions incompatible\n");
        fprintf(stderr, "A: (%d, %d), B: (%d, %d)\n", A->nrows, A->ncols, B->nrows, B->ncols);
        exit(1);
    }

    for (int i = 0; i < C->nrows; i++) {
        for (int j = 0; j < C->ncols; j++) {
            for (int k = 0; k < A->ncols; k++) {
                C->mat[i][j] += A->mat[i][k] * B->mat[k][j];
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
void copy_some_matrix_values(Matrix *original, Matrix *New) {
    for (int i = 0; i < New->nrows; i++) {
        for (int j = 0; j < New->ncols; j++) {
            New->mat[i][j] = original->mat[i][j];
        }
    }
}

void append_bias_factor(Matrix *A) {
    for (int j = 0; j < A->ncols; j++) {
        A->mat[A->nrows -1][j] = 1; 
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

void set_matrix_to_zeros(Matrix *Z) {
    for (int i = 0; i < Z->nrows; i++) {
        for (int j= 0; j < Z->ncols; j++) {
            Z->mat[i][j] = 0;
        }
    }
}


void forward_pass(Layers *layers, Matrix *X, Matrix *W1, Matrix *W2, Matrix *W3) { 
    
    // layer 1
    set_matrix_to_zeros(layers->Z1);
    set_matrix_to_zeros(layers->A1);
    multiply_matrices(W1, X, layers->Z1);
    copy_matrix_values(layers->Z1, layers->A1); 
    append_bias_factor(layers->A1);
    relu(layers->A1);

    // layer 2
    set_matrix_to_zeros(layers->Z2);
    set_matrix_to_zeros(layers->A2);
    multiply_matrices(W2, layers->A1, layers->Z2);
    copy_matrix_values(layers->Z2, layers->A2);
    append_bias_factor(layers->A2);
    relu(layers->A2);

    // layer 3 (output)
    set_matrix_to_zeros(layers->Z3);
    set_matrix_to_zeros(layers->A3);
    multiply_matrices(W3, layers->A2, layers->Z3);
    copy_matrix_values(layers->Z3, layers->A3);
    softmax(layers->A3);

    // debugging purposes
    float sum = 0;
    for (int i = 0; i < layers->A3->nrows; i++) {
        sum = 0;
        for (int j = 0; j < layers->A3->ncols; j++) {
            sum += layers->A3->mat[i][j];
        } 
            printf("avg of row %d: %f\n", i, sum / layers->A3->ncols);
    }
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


int main(int argc, char *argv[]) {
    // read in & prepare data (transpose, train/test split, x/y split, normalize x values) 
    Matrix data = read_csv("MNIST_train.csv");
    Matrix test_data = { .nrows = 785, .ncols = 1000, .mat = calloc(785, sizeof(float *)) };
    Matrix train_data = { .nrows = 785, .ncols = 40999, .mat = calloc(785, sizeof(float *)) };
    train_test_split(&data, &test_data, &train_data);
    Matrix X_train, X_test;
    float *Y_train, *Y_test;
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
    Matrix X_in = { .nrows = 785, .ncols = 10, .mat = calloc(785, sizeof(float *)) };
    for (int i = 0; i < X_in.nrows; i++) {
        X_in.mat[i] = malloc(X_in.ncols * sizeof(float));
    }
    copy_some_matrix_values(&X_test, &X_in);

    // forward pass, prints average for all input examples
    Layers *layers = init_layers(&X_in, &W1, &W2, &W3);
    for (int i = 0; i < 5; i++) {
        forward_pass(layers, &X_test, &W1, &W2, &W3);
    }


    // cleanup
    free_matrix_arr(X_in);
    free_matrix_struct(layers->Z1);
    free_matrix_struct(layers->Z2);
    free_matrix_struct(layers->Z3);
    free_matrix_struct(layers->A1);
    free_matrix_struct(layers->A2);
    free_matrix_struct(layers->A3);
    free(layers);
    free_matrix_arr(X_test);
    free_matrix_arr(X_train);
    free(Y_test);
    free(Y_train);
    free_matrix_arr(W1);
    free_matrix_arr(W2);
    free_matrix_arr(W3);
    return 0;
}

