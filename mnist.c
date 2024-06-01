#include "mnist.h"

Matrix read_csv(const char *filename) {
  Matrix data;

  FILE *fp_orig = fopen(filename, "r");
  if (fp_orig == NULL) {
    printf("Error Opening File %s\n", filename);
    return data;
  }

  char line[MAX_LINE_LENGTH];
  int num_lines = 0;
  int max_tokens = 0;
  FILE *fp = fp_orig;

  // each line in the MNIST csv file is a row containing all info for one pattern 
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
  float **arr = (float**)calloc(num_lines, sizeof(float*));
  for (int i = 0; i < num_lines; i++) {
    arr[i] = (float*)calloc(max_tokens, sizeof(float));
  }

  fp = fp_orig;
  int row = 0;
  while(fgets(line, MAX_LINE_LENGTH, fp)) {  // iterates through every row of the csv file 
    char *token = strtok(line, ",");
    int col = 0;
    while (token != NULL) { // iterates through every entry (col) of the selected row 
      arr[row][col] = atof(token);
      token = strtok(NULL, ",");
      col++;
    }
    row++;
  }

  fclose(fp_orig);
  data.mat = arr;
  data.nrows = num_lines; 
  data.ncols = max_tokens;
  return data;
}

void remove_first_row(Matrix *data) {
  int new_rows = data->nrows - 1;
  float **new_arr = (float **)calloc(new_rows, sizeof(float *));
  for (int i = 0; i < new_rows; i++) {
      new_arr[i] = (float *)calloc(data->ncols, sizeof(float)); 
      memcpy(new_arr[i], data->mat[i + 1], data->ncols * sizeof(float));
  }
  free_arr(*data);
  data->mat = new_arr;
  data->nrows = new_rows;
}

void free_arr(Matrix arr) {
    for (int i = 0; i < arr.nrows; i++) {
        free(arr.mat[i]);
    }
    free(arr.mat);
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
  printf("rows: %d, cols: %d\n", data.nrows, data.ncols);
  for (int i = 0; i < 100; i++) {
      remove_first_row(&data);
      printf("rows: %d, cols: %d\n", data.nrows, data.ncols);
  }

  
  //Matrix W1 = { .num_rows = n_rows, .num_cols = nodes_L2 } 

  /*
  init_weights(&W1);
  init_weights(&W2);
  init_weights(&W3);
  */

  free_arr(data);
  return 0;
}




