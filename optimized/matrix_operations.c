#include "mnist.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

float *initialize_array(int nrows, int ncols) {
    float *arr = malloc(nrows * ncols * sizeof(float));
    return arr;
}

Fmatrix *allocate_matrix(int nrows, int ncols) {
    Fmatrix *M = malloc(sizeof(Fmatrix));
    M->nrows = nrows;
    M->ncols = ncols;
    M->mat = malloc(nrows * ncols * sizeof(float));
    return M;
}

void transpose_matrix(Fmatrix *original, Fmatrix *transposed) {
    int nrows = original->nrows;
    int ncols = original->ncols;
    for (int j = 0; j < ncols; j++) {
        for (int i = 0; i < nrows; i++) {
            transposed->mat[j * nrows + i] = original->mat[i * ncols + j];
        }
    }
}

void multiply_matrices(Fmatrix *A, Fmatrix *B, Fmatrix *C) {
    // reduce linked list operations by storing pointers
    int ncolsA = A->ncols;
    int nrows = C->nrows;
    int ncols = C->ncols;
    float *Amat = A->mat;
    float *Bmat = B->mat;
    float *Cmat = C->mat;
    float *ap;
    float *bp = &Bmat[0];
    float *cp = &Cmat[0];
    int i, j, k;

    if (A->ncols != B->nrows) {
        fprintf(stderr, "Error! Factor matrix dimensions incompatible\n");
        fprintf(stderr, "A: (%d, %d), B: (%d, %d)\n", A->nrows, A->ncols, B->nrows, B->ncols);
        exit(1);
    }

    //set the matrix to zero
    memset(Cmat, 0, C->ncols * C->nrows * sizeof(float));

    for (k = 0; k < ncolsA; k++) {
      ap = &Amat[k];
      for (i = 0; i < nrows; i++) {
        cp = &Cmat[i * ncols];
        bp = &Bmat[k * ncols];
        for (j = 0; j < ncols - 4; j += 4) {
          *cp++ += *ap * *bp++;
          *cp++ += *ap * *bp++;
          *cp++ += *ap * *bp++;
          *cp++ += *ap * *bp++;
        }
        for (; j < ncols; j++) {
          *cp++ += *ap * *bp++;
        }
        ap += ncolsA;
      }
      ap++;
    }
    /*

    //testing non optimized on compiler

    // N: number of rows in A and C
    // M: number of columns in A and rows in B
    // K: number of columns in B and C
    int N = C->nrows;
    int M = A->ncols;
    int K = B->ncols;
    float *Cmat = C->mat;
    float *Amat = A->mat;
    float *Bmat = B->mat;

    // Initialize C to zero

    memset(Cmat, 0, C->ncols * C->nrows * sizeof(float));
    int rowsize, colsize;

    // Perform matrix multiplication
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            rowsize = i * K;
            colsize = i * M;
            for (int k = 0; k < M; k++) {
                Cmat[rowsize + j] += Amat[colsize + k] * Bmat[k * K + j];
            }
        }
    }
    */




}


void multiply_matrices_elementwise(Fmatrix *A, Fmatrix *B, Fmatrix *C, bool omit_last_row) {
    // omit_last_row will be true when multiplying dL/dZ * dL/dA bc dL/dA is one longer (A holds a bias factor and Z does not)
    if (A->nrows != B->nrows - (int)omit_last_row || A->ncols != B->ncols) {
        fprintf(stderr, "Error! Elementwise multiplication matrix dimensions incompatible\n");
        exit(1);
    }

    float *Amat = A->mat;
    float *Bmat = B->mat;
    float *Cmat = C->mat;
    int nrows = C->nrows;
    int ncols = C->ncols;
    int index;

    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            index = i * ncols + j;
            Cmat[index] = Amat[index] * Bmat[index];
        }
    }
}

void scale_matrix(Fmatrix *matrix, float factor) {
    float *mat = matrix->mat;
    int nrows = matrix->nrows;
    int ncols = matrix->ncols;

    for (int i = 0; i < nrows; i++) {
        for (int j = 0 ; j < ncols; j++) {
            mat[i * ncols + j] *= factor;
        }
    }
}

void subtract_matrices(Fmatrix *A, Fmatrix *B, Fmatrix *C) {
    if (A->ncols != B->ncols || A->nrows != B->nrows) {
        fprintf(stderr, "Error! Subtraction matrix dimensions incompatible\n");
        fprintf(stderr, "A: (%d, %d), B: (%d, %d)\n", A->nrows, A->ncols, B->nrows, B->ncols);
        exit(1);
    }
    int index;
    float *Amat = A->mat;
    float *Bmat = B->mat;
    float *Cmat = C->mat;
    int nrows = A->nrows;
    int ncols = A->ncols;

    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            index = i * ncols + j;
            Cmat[index] = Amat[index] - Bmat[index];
        }
    }
}

// use this function to copy a new matrix that is sized >= the original (for example, copying Z values into A matrices, which are larger to hold bias factor)
void copy_all_matrix_values(Fmatrix *original, Fmatrix *New) {
    memcpy(New->mat, original->mat, original->nrows * original->ncols * sizeof(float));
}

// use this fxn to copy a subset of values from a larger matrix into a smaller one (for example, copying X values into a batch matrix)
void copy_some_matrix_values(Fmatrix *original, Fmatrix *New, int start_idx, int end_idx, bool allow_wrap) {
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

    int new_nrows = New->nrows;
    int new_ncols = New->ncols;
    int orig_ncols = original->ncols;
    float *New_mat = New->mat;
    float *orig_mat = original->mat;

    if (!execute_wrap) {
        for (int i = 0; i < new_nrows; i++) {
            for (int j = start_idx; j < end_idx; j++) {
                New_mat[i * new_ncols + j - start_idx] = orig_mat[i * orig_ncols + j];
            }
        }
    }
    else {

        // fill out the values that we can until the end
        int values_till_end = orig_ncols - start_idx;
        for (int i = 0; i < new_nrows; i++) {
            for (int j = start_idx; j < new_ncols; j++) {
                New_mat[i * new_ncols + j - start_idx] = orig_mat[i * orig_ncols + j];
            }
        }

        // fill out the remaining values using the beginning of the original matrix
        for (int i = 0; i < new_nrows; i++) {
            for ( int j = 0; j < new_ncols - values_till_end; j++) {
                New_mat[i * new_ncols + j] = orig_mat[i * orig_ncols + j];
            }
        }
    }
}

void one_hot(Fmatrix *Y, Fmatrix* one_hot_Y) {
    for (int i = 0; i < one_hot_Y->nrows; i++) {
        for (int j = 0; j < one_hot_Y->ncols; j++) {
            if (i == Y->mat[j]) 
                one_hot_Y->mat[i * one_hot_Y->ncols + j] = 1;
            else
                one_hot_Y->mat[i * one_hot_Y->ncols + j] = 0;
        }
    }
}


void argmax_into_yhat(Fmatrix *A, Fmatrix *yhat) {
    float max;
    int max_idx = 0;
    for (int j = 0; j < A->ncols; j++) {
        max = 0;
        for (int i = 0; i < A->nrows; i++) {
            if (A->mat[i * A->ncols + j] > max) {
                max = A->mat[i * A->ncols + j];
                max_idx = i;
            }
        }
        yhat->mat[j] = max_idx;
    }
}

void display_matrix(Fmatrix *X) {
   printf("\n\n"); 
    for (int i = 0; i < X->nrows - 1; i++) {
        if (i % 28 == 0) printf("\n");
        if (X->mat[i] * 255 > 200) printf("*");
        else if (X->mat[i] * 255 > 150) printf("+");
        else if (X->mat[i] * 255 > 100) printf("-");
        else printf(" ");
    }
}
