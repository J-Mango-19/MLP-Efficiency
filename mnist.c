#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mnist.h"
#define MAX_LINE_LENGTH 100000

float ** read_csv(const char * filename, int *num_rows, int *num_cols) {
  FILE *fp_orig = fopen(filename, "r");
  if (fp_orig == NULL) {
    printf("Error Opening File %s\n", filename);
    return NULL;
  }

  char line[MAX_LINE_LENGTH];
  int num_lines = 0;
  int max_tokens = 0;
  FILE *fp = fp_orig;

  // each line in the MNIST csv file is a row
  while (fgets(line, MAX_LINE_LENGTH, fp)) {
      num_lines ++;
      int num_tokens = 1;
      for (char *p = line; *p != '\0'; p++) {
        if (*p == ',') num_tokens++;
      }
      if (num_tokens > max_tokens) {
        max_tokens = num_tokens;
      }
  }

  // memory for 2d array
  float **data = (float**)calloc(num_lines, sizeof(float*));
  for (int i = 0; i < num_lines; i++) {
    data[i] = (float*)calloc(max_tokens, sizeof(float));
  }

  fp = fp_orig;
  int row = 0;
  while(fgets(line, MAX_LINE_LENGTH, fp)) {  // iterates through every row of the csv file 
    char *token = strtok(line, ",");
    int col = 0;
    while (token != NULL) { // iterates through every entry (col) of the selected row 
      data[row][col] = atof(token);
      token = strtok(NULL, ",");
      col++;
    }
    row++;
  }

  fclose(fp);
  *num_rows = num_lines; 
  *num_cols = max_tokens;
  return data;
}

void remove_first_row(float*** arr, int *nrows, int ncols) {
  float **orig_arr = *arr; 
  int new_rows = *nrows - 1;
  float **new_arr = (float **)calloc(new_rows, sizeof(float *));
  for (int i = 0; i < new_rows; i++) {
      new_arr[i] = (float *)calloc(ncols, sizeof(float)); 
      memcpy(new_arr[i], orig_arr[i + 1], ncols * sizeof(float));
  }
  free_arr(orig_arr, *nrows);
  *arr = new_arr;
  *nrows = new_rows;

}

void free_arr(float **arr, int nrows) {
    for (int i = 0; i < nrows; i++) {
        free(arr[i]);
    }
    free(arr);
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
  int n_rows, n_cols;
  float **data = read_csv("MNIST_train.csv", &n_rows, &n_cols);
  printf("rows: %d, cols: %d\n", n_rows, n_cols);
  remove_first_row(&data, &n_rows, n_cols);
  printf("rows: %d, cols: %d\n", n_rows, n_cols);

  
  /*
  Matrix W1 = { .num_rows = n_rows, .num_cols = nodes_L2 } 

  init_weights(&W1);
  init_weights(&W2);
  init_weights(&W3);
  */

  free_arr(data, n_rows);
  return 0;
}




