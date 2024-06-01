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

    for (int i = 1; i < data_matrix.nrows; i++) {
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

void free_arr(Matrix arr) {
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

Matrix process_data(Matrix *data) {
    normalize_data(data);
    Matrix data_T = transpose_matrix(data);
    free_arr(*data);
    return data_T;
}
    
/*
float random_float() {
    return ((float)rand() / (float)RAND_MAX) - 0.5;
}

void init_weights(Matrix *W) {
    // I need to initialize every value of each matrix according to a uniform distribution on (-0.5, 0.5)
    *W.arr = (float **)calloc(W.rows, sizeof(float*)); 

*/



int main(int argc, char *argv[]) {

  Matrix data = read_csv("MNIST_train.csv");
  printf("data: rows: %d, cols: %d\n", data.nrows, data.ncols);
  data = process_data(&data);
  printf("data(processed): rows: %d, cols: %d\n", data.nrows, data.ncols);

  /* 
  Matrix W1 = { .num_rows = data.ncols, .num_cols = nodes_L2 } 
  Matrix W2 = { .num_rows=  

  init_weights(&W1);
  init_weights(&W2);
  init_weights(&W3);
  */

  free_arr(data);
  return 0;
}




