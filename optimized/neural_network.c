#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mnist.h"
#include <math.h>
#include <time.h>

void softmax(Fmatrix *Z) {
    // define a matrix of exponentials of the Z matrix
    float exp_matrix[Z->nrows * Z->ncols]; 
    for (int i = 0; i < Z->nrows; i++) {
       for (int j = 0; j < Z->ncols; j++) {
            exp_matrix[i * Z->ncols + j] = exp(Z->mat[i * Z->ncols + j]);
        }
    }

    // for each column, sum the exponential matrix values
    // then divide each element of the col by the sum and store it
    for (int j = 0; j < Z->ncols; j++) {
        float col_sum = 0; 
        for (int i = 0; i < Z->nrows; i++) {
           col_sum += exp_matrix[i * Z->ncols + j];
        }
        for (int i = 0; i < Z->nrows; i++) {
           Z->mat[i * Z->ncols + j] = exp_matrix[i * Z->ncols + j] / col_sum;
        }
   }
}

void relu(Fmatrix *Z) {
    for (int i = 0; i < Z->nrows; i++) {
        for (int j = 0; j < Z->ncols; j++) {
            if (Z->mat[i * Z->ncols + j] < 0) Z->mat[i * Z->ncols +j] = 0;
        }
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
    for (int i = 0; i < Z->nrows; i++) {
        for (int j = 0; j < Z->ncols; j++) {
            if (Z->mat[i * Z->ncols + j] > 0) 
                derivative->mat[i * derivative->ncols + j] = 1;
            else
                derivative->mat[i * derivative->ncols + j] = 0;
            }
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
    Fmatrix *W1 = weights->W1;
    Fmatrix *W2 = weights->W2;
    Fmatrix *W3 = weights->W3;
    for (int i = 0; i < W3->nrows; i++) {
        for (int j = 0; j < W3->ncols; j++) {
            W3->mat[i * W3->ncols + j] -= lr * deltas->dW3->mat[i * W3->ncols + j];
            deltas->dW3->mat[i * W3->ncols + j] = 0;
        }
    }

    for (int i = 0; i < W2->nrows; i++) {
        for (int j = 0; j < W2->ncols; j++) {
            W2->mat[i * W2->ncols + j] -= lr * deltas->dW2->mat[i * W2->ncols + j];
            deltas->dW2->mat[i * W2->ncols + j] = 0;
        }
    }

    for (int i = 0; i < W1->nrows; i++) {
        for (int j = 0; j < W1->ncols; j++) {
            W1->mat[i * W1->ncols + j] -= lr * deltas->dW1->mat[i * W1->ncols + j];
            deltas->dW1->mat[i * W1->ncols + j] = 0;
        }
    }
}

