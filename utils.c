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
void free_nodes(Nodes *nodes) {
    free_matrix_struct(nodes->Z1);
    free_matrix_struct(nodes->Z2);
    free_matrix_struct(nodes->Z3);
    free_matrix_struct(nodes->A1);
    free_matrix_struct(nodes->A2);
    free_matrix_struct(nodes->A3);
    free(nodes);
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

void init_transpose(Transpose *transpose, Nodes *nodes, int batch_size, Weights *weights, Matrix *X) {
    transpose->one_hot_Y = allocate_matrix(10, batch_size);
    transpose->A2T = allocate_matrix(nodes->A2->ncols, nodes->A2->nrows);
    transpose->W3T = allocate_matrix(weights->W3->ncols, weights->W3->nrows);
    transpose->A1T = allocate_matrix(nodes->A1->ncols, nodes->A1->nrows);
    transpose->W2T = allocate_matrix(weights->W2->ncols, weights->W2->nrows);
    transpose->XT = allocate_matrix(X->ncols, X->nrows);
}

Nodes *init_nodes(Matrix *X, Weights *weights) {
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

void init_deltas(Deltas *deltas, Nodes *nodes, Weights *weights, Matrix* X) {
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

float** initialize_array(int nrows, int ncols) {
    float **arr = malloc(nrows * sizeof(float *));
    size_t row_size = ncols * sizeof(float);
    for (int i = 0; i < nrows; i++) {
        arr[i] = malloc(row_size);
    }
    return arr;
}


void split_data(Matrix *data, Matrix* X_train, Matrix *Y_train, Matrix *X_test, Matrix *Y_test) {
    /*
      Each row of data matrix (42000 x 785) is a class label concatenated with an image vector
      Each column of the X_train (785 x 41000) and X_test (785 x 1000) matrices will be an image vector, with one extra element allocated on the end for a bias factor
      The corresponding columns of Y_train (1 x 41000) and Y_test (1 x 1000) matrices hold the images' labels
    */
    X_test->nrows = X_train->nrows = data->ncols;
    X_test->ncols = 1000;
    X_train->ncols = 41000;

    Y_train->nrows = Y_test->nrows = 1;
    Y_test->ncols = 1000;
    Y_train->ncols = 41000;

    // dynamically allocate 2d arrays for each Matrix struct
    X_test->mat = initialize_array(X_test->nrows, X_test->ncols);
    X_train->mat = initialize_array(X_train->nrows, X_train->ncols);
    Y_train->mat = initialize_array(Y_train->nrows, Y_train->ncols);
    Y_test->mat = initialize_array(Y_test->nrows, Y_test->ncols);

    // assign values from original data matrix into splits
    for (int j = 0; j < Y_test->ncols; j++) {
        Y_test->mat[0][j] = data->mat[j][0];
    }

    for (int j = Y_test->ncols; j < Y_test->ncols + Y_train->ncols; j++) {
        Y_train->mat[0][j - Y_test->ncols] = data->mat[j][0];
    }

    for (int i = 1; i < X_test->nrows; i++) {
        for (int j = 0; j < X_test->ncols; j++) {
            X_test->mat[i - 1][j] = data->mat[j][i];
        }
    }

    for (int i = 1; i < X_train->nrows; i++) {
        for (int j = X_test->ncols; j < X_test->ncols + X_train->ncols; j++) {
            X_train->mat[i - 1][j - X_test->ncols] = data->mat[j][i];
        }
    }

    /*
    // DEBUGGING
    for (int i = 0; i < X_train->nrows; i++) {
        printf("%d ", (int) X_train->mat[i][0]);
    }
    */

    // free the original data matrix
    free_matrix_arr(*data);
}


