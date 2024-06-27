#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "mnist.h"

int main(int argc, char *argv[]) {
    clock_t start, end; // allocation, training, and inference will be timed in this program
    start = clock();

    // read in & prepare data (misc, train/test split, x/y split, normalize x values, append bias factor)
    Preferences *preferences = get_input(argc, argv);
    Fmatrix data = read_csv("MNIST_data.csv");
    Fmatrix X_train, X_test, Y_train, Y_test;
    split_data(&data, &X_train, &Y_train, &X_test, &Y_test);
    scale_matrix(&X_train, (float) 1 / 255);
    scale_matrix(&X_test, (float) 1 / 255);
    append_bias_factor(&X_train);
    append_bias_factor(&X_test);

    // initialize the weights struct to randomize and hold each layer's weights
    Weights weights;
    init_weights(&weights, X_train.nrows, preferences->num_hidden_1, preferences->num_hidden_2, 10);

    // get a batch of the data to begin training on 
    Fmatrix *X_batch = allocate_matrix(X_train.nrows, preferences->batch_size);
    Fmatrix *Y_batch = allocate_matrix(1, preferences->batch_size);
    copy_some_matrix_values(&X_train, X_batch, 0, preferences->batch_size, false);
    copy_some_matrix_values(&Y_train, Y_batch, 0, preferences->batch_size, false);


    // initialize nodes (Number of nodes in each layer will vary with the batch size) and deltas (derivatives) and miscellaneous matrices
    Nodes *nodes_batch = init_nodes(X_batch, &weights);
    Deltas deltas; 
    init_deltas(&deltas, nodes_batch, &weights, X_batch); // only need to allocate deltas corresponding to batch dimensoins bc train & test are only used for inferencing
    Misc misc_matrices; //Even matrices that don't fit into a category are allocated before training begins to avoid repetitive allocation
    init_misc(&misc_matrices, nodes_batch, preferences->batch_size, &weights, X_batch); // only need to allocate misc_matrices in batch dims for same reason as above

    // nodes_train and nodes_test will be used in the forward pass for training and testing accuracy
    Nodes *nodes_train = init_nodes(&X_train, &weights);
    Nodes *nodes_test = init_nodes(&X_test, &weights);
    Fmatrix *train_yhat = allocate_matrix(10, Y_train.ncols);
    Fmatrix *test_yhat = allocate_matrix(10, Y_test.ncols);
    end = clock();
    printf("Allocation took %lf seconds to execute\n", ((double) (end - start)) / CLOCKS_PER_SEC);

    // main training loop
    start = clock();
    for (int i = 0; i < preferences->num_iterations; i++) {
        // forward pass, backpropagation, & weight update
        forward_pass(nodes_batch, X_batch, &weights);
        backward_pass(X_batch, nodes_batch, &weights, Y_batch, &deltas, &misc_matrices);  
        update_weights(&deltas, &weights, preferences->lr);

        // print accuracy
        if (i % preferences->status_interval == 0) {
            print_accuracy(i, nodes_train, nodes_test, &X_train, &X_test, train_yhat, test_yhat, &Y_train, &Y_test, &weights);
        }

        get_next_batch(i, preferences->batch_size, &X_train, &Y_train, X_batch, Y_batch);

    }
    end = clock();
    printf("Training (non-allocation) operations of program took %f seconds to execute\n", ((double) (end - start)) / CLOCKS_PER_SEC);

    // recording forward pass time for comparison - not essential to the program
    start = clock();
    forward_pass(nodes_train, &X_train, &weights);
    end = clock();
    printf("Inference time for entire training set (784 pixels x 41000 examples): %lf seconds\n", ((double) (end - start)) / CLOCKS_PER_SEC);
    
    // Optionally display some examples to command line
    display_examples(preferences->display_start, preferences->display_end, &X_test, &Y_test, &weights);

    // cleanup
    free_matrix_structs(test_yhat, train_yhat, X_batch, Y_batch, weights.W1, weights.W2, weights.W3);
    free_matrix_arrays(&X_test, &X_train, &Y_test, &Y_train);
    free_all_nodes(nodes_train, nodes_test, nodes_batch);
    free_misc(&misc_matrices);
    free_deltas(&deltas);
    free(preferences);
    printf("All memory frees successful\n");
    return 0;
}

