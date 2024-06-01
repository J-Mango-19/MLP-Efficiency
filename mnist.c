#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mnist.h"
#define MAX_LINE_LENGTH 100000

void remove_first_row(float ***arr, int *n_rows, int n_cols);
float ** read_csv(const char *filename, int *num_rows, int *num_cols);
int main(int argc, char *argv[]);
void free_arr(float **arr, int rows);

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
/*
void remove_first_row(float*** arr) {
  float **orig_arr = *arr; 
  *arr = &orig_arr[1]; //&orig_arr[1];
  free(orig_arr[0]);
}
*/
// GPT generated:
void remove_first_row(float ***arr, int *num_rows, int num_cols) {
    float **orig_arr = *arr;
    int new_rows = *num_rows - 1;
    float **new_arr = (float **)calloc(new_rows, sizeof(float *));

    for (int i = 0; i < new_rows; i++) {
        new_arr[i] = (float *)calloc(num_cols, sizeof(float));
        memcpy(new_arr[i], orig_arr[i + 1], num_cols * sizeof(float));
    }

    free_arr(orig_arr, *num_rows);
    *arr = new_arr;
    *num_rows = new_rows;
}


int count_rows(float **arr) {
  int rows = 2;
  while(arr[rows] != NULL) {
    rows++;
  }
  return rows;
}

void free_arr(float **arr, int nrows) {
    int i = 0;
    while(i < nrows) {
        free(arr[i]);
        i++;
    }
    free(arr);
}


int main(int argc, char *argv[]) {
  int n_rows, n_cols;
  float **data = read_csv("MNIST_train.csv", &n_rows, &n_cols);
  printf("%p\n", &data);
  //n_rows = count_rows(data);
  printf("%f %f\n", data[0][0], data[0][1]);
  printf("rows: %d, columns: %d\n", n_rows, n_cols);
  remove_first_row(&data, &n_rows, n_cols); // GPT generated
  printf("%f %f\n", data[0][0], data[0][1]);
  printf("rows: %d, columns: %d\n", n_rows, n_cols);
  free_arr(data, n_rows);
  printf("%p\n", &data);
  return 0;
}




