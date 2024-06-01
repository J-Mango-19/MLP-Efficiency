
void remove_first_row(float ***arr, int *n_rows, int n_cols);
float ** read_csv(const char *filename, int *num_rows, int *num_cols);
int main(int argc, char *argv[]);
void free_arr(float **arr, int rows);
void init_weights(float ***W1, float ***W2, float ***W3);


typedef struct {
    int nrows; 
    int ncols; 
    float **arr;
} matrix;


