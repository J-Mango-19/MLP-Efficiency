#include "mnist.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

float** initialize_array(int nrows, int ncols) {
    float **arr = malloc(nrows * sizeof(float *));
    size_t row_size = ncols * sizeof(float);
    for (int i = 0; i < nrows; i++) {
        arr[i] = malloc(row_size);
    }
    return arr;
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

void transpose_matrix(Matrix *arr, Matrix *transposed) {
    for (int j = 0; j < arr->ncols; j++) {
        for (int i = 0; i < arr->nrows; i++) {
            transposed->mat[j][i] = arr->mat[i][j];
        }
    }
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

    //set the matrix to zero
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


void multiply_matrices_elementwise(Matrix *A, Matrix *B, Matrix *C, bool omit_last_row) {
    // omit_last_row will be true when multiplying dL/dZ * dL/dA bc dL/dA is one longer (A holds a bias factor and Z does not)
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

void scale_matrix(Matrix *matrix, float factor) {
    for (int i = 0; i < matrix->nrows; i++) {
        for (int j = 0 ; j < matrix->ncols; j++) {
            matrix->mat[i][j] *= factor;
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

// use this function to copy a new matrix that is sized >= the original (for example, copying Z values into A matrices, which are larger to hold bias factor)
void copy_all_matrix_values(Matrix *original, Matrix *New) {
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


void argmax_into_yhat(Matrix *A, Matrix *yhat) {
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
