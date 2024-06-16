#include "mnist.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

Matrix read_csv(const char* filename) {
    Matrix data_matrix;
    data_matrix.nrows = 42000;
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

    for (int i = 0; i < data_matrix.nrows; i++) {
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
                data[i][j] = strtof(token, NULL);
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

void train_test_split(Matrix *data, Matrix *test_data, Matrix *train_data) {
    size_t test_rowsize = test_data->ncols * sizeof(float);
    size_t train_rowsize = train_data->ncols * sizeof(float);
    for (int i = 0; i < 785; i++) {
        // allocate and store data for one row of train data
        train_data->mat[i] = malloc(train_rowsize);
        for (int j = 1000; j < 42000; j++) {
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

void XY_split(Matrix *data, Matrix *X, Matrix *Y) {
    // copy first row into Y matrix
    Y->nrows = 1;
    Y->ncols = data->ncols;
    Y->mat = malloc(1 * sizeof(float(*)));
    Y->mat[0] = malloc(Y->ncols * sizeof(float));
    memcpy(Y->mat[0], data->mat[0], data->ncols*sizeof(float));


    // copy the rest of the rows of data into X matrix
    X->ncols = data->ncols;
    X->nrows = data->nrows - 1; 
    X->mat = malloc(X->nrows * sizeof(float(*)));
    size_t row_size = data->ncols * sizeof(float);
    int rowsum = 0;
    for (int i = 1; i < X->nrows + 1; i++) { 
        rowsum ++;
        X->mat[i-1] = malloc(row_size);
        memcpy(X->mat[i-1], data->mat[i], row_size);
    }

    // free the original data array
    free_matrix_arr(*data);
}

void append_bias_input(Matrix *X_train, Matrix *X_test) {
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
        else usage(1);
    }

    if (!(preferences->display_start <= preferences->display_end && preferences->display_start >= 0 && preferences->display_end < 1000)) {
        usage(1);
    }
    return preferences;
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

void init_transpose(Transpose *transpose, Layers *layers, int batch_size, Matrix *W2, Matrix *W3, Matrix *X) {
    transpose->one_hot_Y = allocate_matrix(10, batch_size);
    transpose->A2T = allocate_matrix(layers->A2->ncols, layers->A2->nrows);
    transpose->W3T = allocate_matrix(W3->ncols, W3->nrows);
    transpose->A1T = allocate_matrix(layers->A1->ncols, layers->A1->nrows);
    transpose->W2T = allocate_matrix(W2->ncols, W2->nrows);
    transpose->XT = allocate_matrix(X->ncols, X->nrows);
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

void init_deltas(Deltas *deltas, Layers *layers, Matrix *W2, Matrix *W3, Matrix* X) {
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

