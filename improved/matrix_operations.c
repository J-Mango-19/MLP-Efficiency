#include "mnist.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>
#include <pthread.h>

#define NUM_THREADS 10

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
    float *Tmat = transposed->mat;
    float *Omat = original->mat;
    float *Tp, *Op;

    for (int j = 0; j < ncols; j++) {
        Tp = &Tmat[j * nrows];
        Op = &Omat[j];
        for (int i = 0; i < nrows; i++) {
            //Tmat[j * nrows + i] = Omat[i * ncols + j];
            *Tp++ = *Op;
            Op += ncols;
        }
    }
}

void multiply_matrices_standard(Fmatrix *A, Fmatrix *B, Fmatrix *C) {
    // reduce linked list operations by storing pointers
    int ncolsA = A->ncols;
    int nrows = C->nrows;
    int ncols = C->ncols;
    float *Amat = A->mat;
    float *Bmat = B->mat;
    float *Cmat = C->mat;
    float *Ap = &Amat[0];
    float *Bp = &Bmat[0];
    float *Cp = &Cmat[0];
    float *end_ptr;
    int i, k;

    if (A->ncols != B->nrows) {
        fprintf(stderr, "Error! Factor matrix dimensions incompatible\n");
        fprintf(stderr, "A: (%d, %d), B: (%d, %d)\n", A->nrows, A->ncols, B->nrows, B->ncols);
        exit(1);
    }

    //set the matrix to zero
    memset(Cmat, 0, ncols * nrows * sizeof(float));

    // multiplication logic
    for (i = 0; i < nrows; i++) {
        Ap = &Amat[i * ncolsA];
        for (k = 0; k < ncolsA; k++) {
            __m256 a = _mm256_set1_ps(*Ap);
            Bp = &Bmat[k * ncols];
            Cp = &Cmat[i * ncols];
            end_ptr = (Cp + ncols) - 8;
            for(; Cp <= end_ptr; Cp += 8) {
                __m256 b = _mm256_loadu_ps(Bp);
                __m256 c = _mm256_loadu_ps(Cp);
                c = _mm256_add_ps(c, _mm256_mul_ps(a, b));
                _mm256_storeu_ps(Cp, c);
                Bp += 8;
            }
            // Handle remaining elements
            end_ptr += 8;
            while (Cp < end_ptr) {
                *Cp++ += *Ap * *Bp++;
            }
            Ap++;
        }
    }
}

void multiply_matrices_elementwise(Fmatrix *A, Fmatrix *B, Fmatrix *C, bool omit_last_row) {
    // omit_last_row will be true when multiplying dL/dZ * dL/dA bc dL/dA is one longer (A holds a bias factor and Z does not)
    if (A->nrows != B->nrows - (int)omit_last_row || A->ncols != B->ncols) {
        fprintf(stderr, "Error! Elementwise multiplication matrix dimensions incompatible\n");
        exit(1);
    }
    int size = A->nrows * A->ncols;
    float *Ap = &A->mat[0];
    float *Bp = &B->mat[0];
    float *Cp = &C->mat[0];

    for (int i = 0; i < size; i++) {
        *Cp++ = *Ap++ * *Bp++;
    }
}

void scale_matrix(Fmatrix *matrix, float factor) {
    int size = matrix->nrows * matrix->ncols;
    float *Mp = &matrix->mat[0];
    for (int i = 0; i < size; i++) {
        *Mp++ *= factor;
    }
}

void subtract_matrices(Fmatrix *A, Fmatrix *B, Fmatrix *C) {
    if (A->ncols != B->ncols || A->nrows != B->nrows) {
        fprintf(stderr, "Error! Subtraction matrix dimensions incompatible\n");
        fprintf(stderr, "A: (%d, %d), B: (%d, %d)\n", A->nrows, A->ncols, B->nrows, B->ncols);
        exit(1);
    }
    int size = A->nrows * A->ncols;
    float *Ap = &A->mat[0];
    float *Bp = &B->mat[0];
    float *Cp = &C->mat[0];
    for (int i = 0; i < size; i++) {
        *Cp++ = *Ap++ - *Bp++;
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
    float *Np, *Op;

    if (!execute_wrap) {
        for (int i = 0; i < new_nrows; i++) {
            Np = &New_mat[i * new_ncols];
            Op = &orig_mat[i * orig_ncols + start_idx];
            for (int j = start_idx; j < end_idx; j++) {
                *Np++ = *Op++;
            }
        }
    }
    else {

        int values_till_end = orig_ncols - start_idx;
        for (int i = 0; i < new_nrows; i++) {
            Np = &New_mat[i * new_ncols - start_idx];
            Op = &orig_mat[i * orig_ncols];
            for (int j = start_idx; j < new_ncols; j++) {
                *Np++ = *Op++;
            }
        }

        // fill out the remaining values using the beginning of the original matrix
        for (int i = 0; i < new_nrows; i++) {
            Np = &New_mat[i * new_ncols];
            Op = &orig_mat[i * orig_ncols];
            for ( int j = 0; j < new_ncols - values_till_end; j++) {
                //New_mat[i * new_ncols + j] = orig_mat[i * orig_ncols + j];
                *Np++ = *Op++;
            }
        }
    }
}

void one_hot(Fmatrix *Y, Fmatrix* one_hot_Y) {
    float *One_hot;
    float *Yp; 
    for (int i = 0; i < one_hot_Y->nrows; i++) {
        One_hot = &one_hot_Y->mat[i * one_hot_Y->ncols];
        Yp = &Y->mat[0];
        for (int j = 0; j < one_hot_Y->ncols; j++) {
            if (i == *Yp++) 
                *One_hot++ = 1;
            else 
                *One_hot++ = 0;
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

void *matmul_threaded_worker(void *arg) {
    ThreadData* data = (ThreadData*)arg;
    float *Amat = data->Amat;
    float *Bmat = data->Bmat;
    float *Cmat = data->Cmat;
    int ncolsA = data->ncolsA;
    int nrows = data->nrows;
    int ncols = data->ncols;
    int start = data->start;
    int end = data->end;
    float *Cp, *Ap, *Bp;
    int i, j, k;
    for (k = 0; k < ncolsA; k++) {
        Ap = Amat + k;
        for (i = 0; i < nrows; i++) {
            __m256 a = _mm256_set1_ps(*Ap);
            Cp = &Cmat[i * ncols + start];
            Bp = &Bmat[k * ncols + start];
            for (j = start; j <= end - 8; j+= 8) {
                __m256 b = _mm256_loadu_ps(Bp);
                __m256 c = _mm256_loadu_ps(Cp);
                c = _mm256_add_ps(c, _mm256_mul_ps(a, b));
                _mm256_storeu_ps(Cp, c);
                Cp += 8;
                Bp += 8;
            }
            while (j < end) {
                *Cp++ += *Ap * *Bp++;
                j++;
            }
            Ap += ncolsA;
        }
    }
    return NULL;
}

void multiply_matrices(Fmatrix *A, Fmatrix *B, Fmatrix *C) {
    // use multitheading for large matrices
    if (B->ncols > 2000 && B->ncols % NUM_THREADS == 0) {
        multiply_matrices_threads(A, B, C);
    }
    // use standard algorithm for small matrices
    else {
        multiply_matrices_standard(A, B, C);
    }
}
void multiply_matrices_threads(Fmatrix *A, Fmatrix *B, Fmatrix *C) {
    // check size compatibililty
    if (A->ncols != B->nrows || C->nrows != A->nrows || C->ncols != B->ncols)  {
        printf("matmul_base: Error: incompatible matrix dimensions\n");
        printf("%d\n", A->ncols == B->nrows);
        printf("%d, %d\n", A->ncols, B->nrows);
        printf("%d\n", C->nrows == A->nrows);
        printf("%d\n", C->ncols == B->ncols);
        exit(1);
    }

    // reduce linked list operations by storing pointers
    int ncolsA = A->ncols;
    int nrows = C->nrows;
    int ncols = C->ncols;
    float *Amat = A->mat;
    float *Bmat = B->mat;
    float *Cmat = C->mat;

    memset(C->mat, 0, C->nrows * C->ncols * sizeof(float));

    /*
     * number of examples: 41000 or 100
     * each thread will work on a different section of each example 
     * so, divide the number of examples by the number of threads to get the area that each one works on
     * 41000 / 5 = 8200
     * so t1 : (0, 8200), t2 : (8200, 16400), t3 : (16400, 24600) etc
     * all of the threads are synchronized after each example
     * No need a mutex for C access bc it's incremented by j, and each thread has its own j 
     */

    // Make the threads
    pthread_t threads[NUM_THREADS];
    ThreadData thread_datas[NUM_THREADS]; 
    int examples_per_thread = B->ncols / NUM_THREADS;
    if (B->ncols % NUM_THREADS != 0) {
        printf("Batch size not divisible by number of threads\n");
        exit(1);
    }

    // assign the thread args based on the portion of examples they will cover
    for (int i = 0; i < NUM_THREADS ; i++) {
            thread_datas[i] = (ThreadData){
            .Amat = Amat,
            .Bmat = Bmat,
            .Cmat = Cmat,
            .ncolsA = ncolsA,
            .ncols = ncols,
            .nrows = nrows,
            .start = i * examples_per_thread,
            .end = (i + 1) * examples_per_thread
            };
    }

    // launch the threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, matmul_threaded_worker, (void*) &thread_datas[i]);
    }

    // collect the threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}
