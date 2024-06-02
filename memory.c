#include <stdio.h>
#include <stdlib.h>

#define ROWS 3
#define COLS 5

int main() {
    int rows = ROWS;
    int cols = COLS;

    // Allocate memory for the original 2D array
    int **original_array = (int **)malloc(rows * sizeof(int *));
    for (int i = 0; i < rows; i++) {
        original_array[i] = (int *)malloc(cols * sizeof(int));
    }

    // Initialize the original 2D array with some values
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            original_array[i][j] = i * cols + j;
        }
    }

    // Allocate memory for the new 2D array
    int **new_array = (int **)malloc(ROWS * sizeof(int *));
    for (int i = 0; i < ROWS; i++) {
        new_array[i] = (int *)malloc(COLS * sizeof(int));
    }

    // Copy the first 1000 rows of original_array into new_array
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            new_array[i][j] = original_array[i][j];
        }
    }

    // Print the first few rows of new_array
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%d ", new_array[i][j]);
        }
        printf("\n");
    }

    // Free memory for the original 2D array
    for (int i = 0; i < rows; i++) {
        free(original_array[i]);
    }
    free(original_array);

    // Free memory for the new 2D array
    for (int i = 0; i < ROWS; i++) {
        free(new_array[i]);
    }
    free(new_array);



    return 0;
}
