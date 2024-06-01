#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <string.h>

#define N_ROWS 10
#define N_COLS 10

void fill_array(int **arr, int nrows, int ncols); 
void print_arr(int**arr, int nrows, int ncols); 

void fill_array(int **arr, int nrows, int ncols) {
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j< ncols; j++) {
      arr[i][j] = i * j;
    }
  }
}

void print_arr(int**arr, int nrows, int ncols) {
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      printf("%d ", arr[i][j]);
    }
    printf("\n");
  }
}

int **segment_arr(int**arr, int nrows, int ncols) {
  int **segarr = calloc(5, sizeof(int*)); 
  for (int i = 0; i < 5; i++ ) { // allocate memory for the rest of the new 2d array
    segarr[i] = calloc(N_COLS, sizeof(int));
  }

  memcpy(segarr, arr, 5 * ncols * sizeof(int));

  print_arr(segarr, 5, ncols);
// then get it going in the real program
  
  return segarr;
}

int main() {
  int **arr2d = calloc(N_ROWS, sizeof(int*));
  for (int i = 0; i < N_ROWS; i++) {
    arr2d[i] = calloc(N_COLS, sizeof(int));
  }

  fill_array(arr2d, N_ROWS, N_COLS);
  //print_arr(arr2d, N_ROWS, N_COLS);

  int **segarr = segment_arr(arr2d, 5, N_COLS);
  return 0;
}

