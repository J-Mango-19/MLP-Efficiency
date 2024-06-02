#include "mnist.h"

// Function to read a CSV file into a 2D array of floats
Matrix read_csv(const char* filename) {
    Matrix data_matrix;
    data_matrix.nrows = 41999;
    data_matrix.ncols = 785;
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Could not open file %s for reading\n", filename);
        return data_matrix;
    }

    // Allocate memory for the 2D array
    float **data = (float **)malloc(data_matrix.nrows* sizeof(float *));
    if (!data) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return data_matrix;
    }

    for (int i = 0; i < data_matrix.nrows; i++) {
        data[i] = (float *)malloc(data_matrix.ncols * sizeof(float));
        if (!data[i]) {
            fprintf(stderr, "Memory allocation failed\n");
            // Free previously allocated memory before returning
            for (int j = 0; j < i; j++) {
                free(data[j]);
            }
            free(data);
            fclose(file);
            return data_matrix;
        }
    }

    char line[16000]; // Large enough buffer to hold one line of the CSV file

    for (int i = 1; i < data_matrix.nrows + 1; i++) {
        if (!fgets(line, sizeof(line), file)) {
            fprintf(stderr, "Error reading line %d\n", i);
            // Free allocated memory before returning
            for (int j = 0; j <= i; j++) {
                free(data[j]);
            }
            free(data);
            fclose(file);
            return data_matrix;
        }
        char *token = strtok(line, ",");
        for (int j = 0; j < data_matrix.ncols; j++) {
            if (token) {
                data[i-1][j] = strtof(token, NULL);
                token = strtok(NULL, ",");
            } else {
                fprintf(stderr, "Error parsing line %d\n", i);
                // Free allocated memory before returning
                for (int k = 0; k <= i; k++) {
                    free(data[k]);
                }
                free(data);
                fclose(file);
                return data_matrix;
            }
        }
    }
    data_matrix.mat = data;
    fclose(file);
    return data_matrix;
}

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

void normalize_data(Matrix *data) {
    for (int i = 0; i < data->nrows; i++) {
        for (int j = 0; j < data->ncols; j++) {
            data->mat[i][j] /= 255;
        }
    }
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

void train_test_split(Matrix *data, Matrix *test_data, Matrix *train_data) {
    size_t test_rowsize = test_data->ncols * sizeof(float);
    size_t train_rowsize = train_data->ncols * sizeof(float);
    for (int i = 0; i < 785; i++) {

        // allocate and store data for one row of train data
        train_data->mat[i] = malloc(train_rowsize);
        for (int j = 0; j < 41999; j++) {
            train_data->mat[i][j-1000] = data->mat[j][i];
        }

        // allocate and store data for one row of test data
        test_data->mat[i] = malloc(test_rowsize);
        for (int j = 0; j < 1000; j++) {
            test_data->mat[i][j] = data->mat[j][i];
        }
    }
    // free the original data matrix
    free_matrix_arr(*data);

}

void XY_split(Matrix *data, Matrix *X, float **Y) {
    // copy first row into Y array
    *Y = (float*)malloc(data->ncols * sizeof(float));
    memcpy(*Y, data->mat[0], data->ncols * sizeof(float));


    // copy the rest of the rows of data into X matrix
    X->ncols = data->ncols;
    X->nrows = data->nrows - 1;
    X->mat = malloc(X->nrows * sizeof(float(*)));
    size_t row_size = data->ncols * sizeof(float);
    for (int i = 1; i < X->nrows + 1; i++) {
        X->mat[i-1] = malloc(row_size);
        memcpy(X->mat[i-1], data->mat[i], row_size); 
    }

    // free the original data array 
    free_matrix_arr(*data);
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

void append_bias(Matrix *X_train, Matrix *X_test) {
    // define & allocate replacement arrays that will contain the same values and also a bias multpliying 1 for each input pattern
    float **new_arr_train = malloc((X_train->nrows + 1) * sizeof(float *));
    float **new_arr_test = malloc((X_test->nrows + 1) * sizeof(float *));
    for (int i = 0; i < X_train->nrows; i++) {

        // allocate & reassign values for X_train
        new_arr_train[i] = malloc(X_train->ncols * sizeof(float));
        for (int j = 0; j < X_train->ncols; j++) {
            new_arr_train[i][j] = X_train->mat[i][j];
        }

        //allocate & reassign values for X_test
        new_arr_test[i] = malloc(X_test->ncols * sizeof(float));
        for (int j = 0; j < X_test->ncols; j++) {
            new_arr_test[i][j] = X_test->mat[i][j];
        }
    }

    // allocate & assign a 1 for the bias term of each input pattern
    new_arr_train[X_train->nrows] = malloc(X_train->ncols * sizeof(float));
    new_arr_test[X_test->nrows] = malloc(X_test->ncols * sizeof(float));
    for (int j = 0; j < X_train->ncols; j++) {
        new_arr_train[X_train->nrows][j] = 1;
    }
    for (int j = 0; j < X_test->ncols; j++) {
        new_arr_test[X_test->nrows][j] = 1;
    }

    // free the old matrices & replace with new ones
    free_matrix_arr(*X_train);
    free_matrix_arr(*X_test);
    X_train->nrows += 1;
    X_test->nrows += 1;
    X_train->mat = new_arr_train;
    X_test->mat = new_arr_test;
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

Matrix *multiply_matrices(Matrix *A, Matrix *B, bool extra_bias_row) {
    if (A->ncols != B->nrows) {
        fprintf(stderr, "Error! Factor matrix dimensions incompatible\n");
        fprintf(stderr, "A: (%d, %d), B: (%d, %d)\n", A->nrows, A->ncols, B->nrows, B->ncols);
        exit(1);
    }


    Matrix *C = allocate_matrix(A->nrows + (int)extra_bias_row, B->ncols);
    C->nrows -= (int)extra_bias_row;

    // multiply matrices
    for (int i = 0; i < C->nrows; i++) {
        for (int j = 0; j < C->ncols; j++) {
            for (int k = 0; k < A->ncols; k++) {
                C->mat[i][j] += A->mat[i][k] * B->mat[k][j];
            }
        }
    }
    if (extra_bias_row) C->nrows += 1;
    return C;
}

void add_bias_factor(Matrix *A) {
    for (int j = 0; j < A->ncols; j++) {
        A->mat[A->nrows - 1][j] = 1;
    }
}

Matrix *copy_matrix(Matrix *original) {
    Matrix *new_matrix = malloc(sizeof(Matrix));
    new_matrix->nrows = original->nrows;
    new_matrix->ncols = original->ncols;
    new_matrix->mat = malloc(new_matrix->nrows * sizeof(float *));
    size_t row_size = new_matrix->ncols * sizeof(float);
    for (int i = 0; i < new_matrix->nrows; i++) {
        new_matrix->mat[i] = malloc(row_size);
        memcpy(new_matrix->mat[i], original->mat[i], row_size); 
    }
    return new_matrix;
}
        

Layers forward_pass(Matrix *X, Matrix *W1, Matrix *W2, Matrix *W3) { 
    Matrix *Z1 = multiply_matrices(W1, X, true);
    Matrix *A1 = copy_matrix(Z1);
    relu(A1);
    add_bias_factor(A1); 
    Matrix *Z2 = multiply_matrices(W2, A1, true);
    Matrix *A2 = copy_matrix(Z2);
    relu(A2);
    add_bias_factor(A2);
    Matrix *Z3 = multiply_matrices(W3, A2, false);
    Matrix *A3 = copy_matrix(Z3);
    softmax(A3);
    float sum = 0;
    for (int i = 0; i < A3->nrows; i++) {
        sum = 0;
        for (int j = 0; j < A3->ncols; j++) {
            sum += A3->mat[i][j];
        } 
            printf("avg of row %d: %f\n", i, sum/A3->ncols);
    }
    Layers layers = {.Z1 = Z1, .Z2 = Z2, .Z3 = Z3, .A1 = A1, .A2 = A2, .A3 = A3};
    return layers;
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
    append_bias(&X_train, &X_test);

    // initialize weights (don't forget to include biases!!) I think this would involve adding 1 element to the end of the rows and initializing it to random too, also must
    // add 1 to end of every input
    Matrix W1 = {.nrows = 30, .ncols = X_train.nrows, .mat = malloc(30 * sizeof(float*))};
    Matrix W2 = {.nrows = 20, .ncols = 30 + 1, .mat = malloc(20 * sizeof(float*))};
    Matrix W3 = {.nrows = 10, .ncols = 20 + 1, .mat = malloc(10 * sizeof(float*))};

    init_weights(&W1);
    init_weights(&W2);
    init_weights(&W3);

    // forward pass, prints average for all training examples
    // it will become necessary to write a function that returns a matrix containing x number of training examples
    Layers layers = forward_pass(&X_test, &W1, &W2, &W3);


    // cleanup
    free_matrix_struct(layers.Z1);
    free_matrix_struct(layers.Z2);
    free_matrix_struct(layers.Z3);
    free_matrix_struct(layers.A1);
    free_matrix_struct(layers.A2);
    free_matrix_struct(layers.A3);
    free_matrix_arr(X_test);
    free_matrix_arr(X_train);
    free(Y_test);
    free(Y_train);
    free_matrix_arr(W1);
    free_matrix_arr(W2);
    free_matrix_arr(W3);
    return 0;
}



