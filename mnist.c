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

void free_matrix(Matrix arr) {
    for (int i = 0; i < arr.nrows; i++) {
        free(arr.mat[i]);
    }
    free(arr.mat);
}

void normalize_data(Matrix *data) {
    for (int i = 0; i < data->nrows; i++) {
        for (int j = 0; j < data->ncols; j++) {
            data->mat[i][j] /= 255;
        }
    }
}

float random_float() {
    return ((float)rand() / (float)RAND_MAX) - 0.5;
}

void init_weights(Matrix *W) {
    // Initialize every value of each matrix according to a uniform distribution on (-0.5, 0.5)
    W->mat = (float **)calloc(W->nrows, sizeof(float*)); 
    for (int i = 0; i < W->ncols; i++) {
        W->mat[i] = calloc(W->ncols, sizeof(float));
        for (int j = 0; j < W->ncols; j++) {
            W->mat[i][j] = random_float();
        }
    }
}

void train_test_split(Matrix *data, Matrix *test_data, Matrix *train_data) {
    size_t test_rowsize = test_data->ncols * sizeof(float);
    for (int i = 0; i < 785; i++) {
        test_data->mat[i] = malloc(test_rowsize);
        for (int j = 0; j < 1000; j++) {
            test_data->mat[i][j] = data->mat[j][i];
        }
    }
    // i corresponds to the rows of data matrix, so each i is a pattern
    size_t train_rowsize = train_data->ncols * sizeof(float);
    for (int i = 0; i < 785; i++) {
        train_data->mat[i] = malloc(train_rowsize);
        for (int j = 1000; j < 41999; j++) {
            train_data->mat[i][j-1000] = data->mat[j][i];
        }
    }

    // free the original data matrix
    free_matrix(*data);
    /* POTENTIAL TO DOUBLE UP ON FOR LOOP COVERAGE HERE!!
    for (int i = 0; i < 785, i++) {

        // allocate and store data for one row of 
        train_data->mat[i] = malloc(train_rowsize);
    */
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
    free_matrix(*data);
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

    // testing X 
    /*
    for (int i = 0; i < X_test.nrows; i++){
        printf("------------------------------\n");
        for (int j = 0; j < X_test.ncols; j++) {
            printf("%f\n", X_test.mat[i][j]);
        }
    }
    */

    /*
    // testing Y
    for (int i = 0; i < train_data.ncols; i++) {
        printf("%f\n", Y_train[i]);
    }
    */

    // initialize weights (don't forget to include biases!!) I think this would involve adding 1 element to the end of the rows and initializing it to random too, also must
    // add 1 to end of every input
    Matrix W1 = { .nrows = data.ncols, .ncols = 30};
    Matrix W2 = { .nrows = 30, .ncols = 20};
    Matrix W3 = { .nrows = 20, .ncols = 10};

    init_weights(&W1);
    init_weights(&W2);
    init_weights(&W3);



    // cleanup
    free_matrix(X_test);
    free_matrix(X_train);
    free(Y_test);
    free(Y_train);
    free_matrix(W1);
    free_matrix(W2);
    free_matrix(W3);
    return 0;
}




