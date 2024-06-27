#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mnist.h"
#include <math.h>
#include <time.h>

void softmax(Fmatrix *Z) {
    float exp_matrix[Z->nrows * Z->ncols]; 
    float *Ep = &exp_matrix[0];
    float *Zp = &Z->mat[0];
    int nrows = Z->nrows;
    int ncols = Z->ncols;

    // since there are 10 classes, Z will have 10 rows
    for (int i = 0; i < 10; i++) {
        Ep = &exp_matrix[i];
        Zp = &Z->mat[ncols * i];
        for (int j = 0; j < ncols; j++) {
            *Ep = exp(*Zp++);
            Ep += nrows;
        }
    }

    // for each column, sum the exponential matrix values
    // then divide each element of the col by the sum and store it
    for (int j = 0; j < ncols; j++) {
        float col_sum = 0; 
        Ep = &exp_matrix[j * nrows];
        for (int i = 0; i < nrows; i++) {
           col_sum += *Ep++;
        }
        Ep = &exp_matrix[j * nrows];
        Zp = &Z->mat[j];
        for (int i = 0; i < nrows; i++) {
           *Zp = *Ep++ / col_sum;
           Zp += ncols;
        }
    }
    // Z is 10 x 42k ie very long so the best way to access is along the rows 
    // e is 42k x 10 ie very tall so the best way to access is along the columns
    // task is to look at a chunk of 10 at a time, then move onto the next chunk of 10
    // this would mean e is very efficient to access
    // ill start by accessing e better using the transposed values then check training then try out transposing z for better writing acess

}

void relu(Fmatrix *Z) {
    int size = Z->nrows * Z->ncols;
    float *Zp = &Z->mat[0];
    for (int i = 0; i < size; i++) {
        if (*Zp < 0) *Zp = 0;
        Zp++;
    }
}

void forward_pass(Nodes *nodes, Fmatrix *X, Weights *weights) { 
    // layer 1
    multiply_matrices(weights->W1, X, nodes->Z1);
    copy_all_matrix_values(nodes->Z1, nodes->A1); 
    append_bias_factor(nodes->A1);
    relu(nodes->A1);

    // layer 2
    multiply_matrices(weights->W2, nodes->A1, nodes->Z2);
    copy_all_matrix_values(nodes->Z2, nodes->A2);
    append_bias_factor(nodes->A2);
    relu(nodes->A2);

    // layer 3 (output)
    multiply_matrices(weights->W3, nodes->A2, nodes->Z3);
    copy_all_matrix_values(nodes->Z3, nodes->A3);
    softmax(nodes->A3);
}

void deriv_relu(Fmatrix *Z, Fmatrix *derivative) {
    int size = Z->nrows * Z->ncols;
    float *Zp = &Z->mat[0];
    float *Dp = &derivative->mat[0];
    for (int i = 0; i < size; i++) {
        if (*Zp++ > 0) *Dp++ = 1;
        else *Dp++ = 0;
    }

}

void backward_pass(Fmatrix *X, Nodes *nodes, Weights *weights, Fmatrix *Y, Deltas *deltas, Misc *misc) {
    one_hot(Y, misc->one_hot_Y);
    subtract_matrices(nodes->A3, misc->one_hot_Y, deltas->dZ3); // dL/dZ3 = A3 - one_hot_Y
    transpose_matrix(nodes->A2, misc->A2T);               
    // divide matrix below by number of patterns ie batch size here
    multiply_matrices(deltas->dZ3, misc->A2T, deltas->dW3);                      // dL/dW3 = dL/dZ3 · A2T
    scale_matrix(deltas->dW3, (float) 1 / Y->ncols );
    transpose_matrix(weights->W3, misc->W3T); 
    multiply_matrices(misc->W3T, deltas->dZ3, deltas->dA2);                       // dL/dA2 = W3T · dL/dZ3
    deriv_relu(nodes->Z2, deltas->dA2_dZ2);      
    multiply_matrices_elementwise(deltas->dA2_dZ2, deltas->dA2, deltas->dZ2, true);         // dL/dZ2 = dA2/dZ2 * dL/dA2
    transpose_matrix(nodes->A1, misc->A1T);
    // divide matrix below by number of patterns in batch size here
    multiply_matrices(deltas->dZ2, misc->A1T, deltas->dW2);                      // dL/dW2 = dZ2/dW2 · dL/dZ2
    scale_matrix(deltas->dW2, (float) 1 / Y->ncols);
    transpose_matrix(weights->W2, misc->W2T);
    multiply_matrices(misc->W2T, deltas->dZ2, deltas->dA1);                       // dL/dA1 = dZ2/dA1 · dL/dZ2 = W2T · dL/dZ2
    deriv_relu(nodes->Z1, deltas->dA1_dZ1); 
    multiply_matrices_elementwise(deltas->dA1_dZ1, deltas->dA1, deltas->dZ1, true); // dL/dZ1 = dA1/dZ1 * dL/dA1
    transpose_matrix(X, misc->XT);
    // divide matrix below by number of patterns in bach size here
    multiply_matrices(deltas->dZ1, misc->XT, deltas->dW1);
    scale_matrix(deltas->dW1, (float) 1 / Y->ncols);
}

void update_weights(Deltas *deltas, Weights *weights, float lr) {

    float *W1p = &weights->W1->mat[0];
    float *W2p = &weights->W2->mat[0];
    float *W3p = &weights->W3->mat[0];
    int size_W1 = weights->W1->nrows * weights->W1->ncols;
    int size_W2 = weights->W2->nrows * weights->W2->ncols;
    int size_W3 = weights->W3->nrows * weights->W3->ncols;
    float *dW1p = &deltas->dW1->mat[0];
    float *dW2p = &deltas->dW2->mat[0];
    float *dW3p = &deltas->dW3->mat[0];

    for (int i = 0; i < size_W1; i++) {
        *W1p++ -= lr * *dW1p++;
    }

    for (int i = 0; i < size_W2; i++) {
        *W2p++ -= lr * *dW2p++;
    }

    for (int i = 0; i < size_W3; i++) {
        *W3p++ -= lr * *dW3p++;
    }
}

