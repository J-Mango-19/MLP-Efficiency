#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

float random_float(); 
void randomize_weights(float *weights, int length);
void randomize_weights_He(float *weights, int length, int fan_in);

inline float random_float() {
    return ((float)rand() / (float)RAND_MAX); // Generate numbers between -0.5 and 0.5
}

void randomize_weights(float *weights, int length) {
    // Initialize every value of each matrix according to a uniform distribution on (-0.5, 0.5)
    for (int i = 0; i < length; i++) {
        weights[i] = random_float() - 0.5;
    }
}

void randomize_weights_He(float *weights, int length, int fan_in) {
    float stddev = sqrt(2.0 / fan_in);
    for (int i = 0; i < length; i++) {
        float u1 = random_float(); 
        float u2 = random_float();
        float z1 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        weights[i] = z1 * stddev;
    }
}

void visualize_weights(float *weights, int length) {
    int *positions = calloc(20, sizeof(int));
    for (int i = 0; i < length; i++) {
        if ((weights[i] < -0.9) && (weights[i] >= -1.0)) 
            positions[0] += 1;
        else if ((weights[i] < -0.8) && (weights[i] >= -0.9)) 
            positions[1] += 1;
        else if ((weights[i] < -0.7) && (weights[i] >= -0.8)) 
            positions[2] += 1;
        else if ((weights[i] < -0.6) && (weights[i] >= -0.7)) 
            positions[3] += 1;
        else if ((weights[i] < -0.5) && (weights[i] >= -0.6)) 
            positions[4] += 1;
        else if ((weights[i] < -0.4) && (weights[i] >= -0.5)) 
            positions[5] += 1;
        else if ((weights[i] < -0.3) && (weights[i] >= -0.4)) 
            positions[6] += 1;
        else if ((weights[i] < -0.2) && (weights[i] >= -0.3)) 
            positions[7] += 1;
        else if ((weights[i] < -0.1) && (weights[i] >= -0.2)) 
            positions[8] += 1;
        else if ((weights[i] < 0.0) && (weights[i] >= -0.1)) 
            positions[9] += 1;
        else if ((weights[i] < 0.1) && (weights[i] >= 0.0)) 
            positions[10] += 1;
        else if ((weights[i] < 0.2) && (weights[i] >= 0.1))
            positions[11] += 1;
        else if ((weights[i] < 0.3) && (weights[i] >= 0.2))
            positions[12] += 1;
        else if ((weights[i] < 0.4) && (weights[i] >= 0.3))
            positions[13] += 1;
        else if ((weights[i] < 0.5) && (weights[i] >= 0.4))
            positions[14] += 1;
        else if ((weights[i] < 0.6) && (weights[i] >= 0.5))
            positions[15] += 1;
        else if ((weights[i] < 0.7) && (weights[i] >= 0.6))
            positions[16] += 1;
        else if ((weights[i] < 0.8) && (weights[i] >= 0.7))
            positions[17] += 1;
        else if ((weights[i] < 0.9) && (weights[i] >= 0.8))
            positions[18] += 1;
        else if ((weights[i] <= 1.0) && (weights[i] >= 0.9))
            positions[19] += 1;
    }

    for (int i = 0; i < 20; i++) {
        for (int num_hashtags = 0; num_hashtags < positions[i]; num_hashtags++) {
            printf("#");
        }
        printf("\n");
    }
    free(positions);
}

int main() {
    srand(time(NULL));
    float weights[300];
    randomize_weights(weights, 300);
    printf("Weight distribution for uniform distribution on [-0.5, 0.5]: \n");
    visualize_weights(weights, 300);
    printf("\n-----------\n");
    printf("He weight distribution (closer to a normal distribution with 0 mean and very low variance):\n");
    randomize_weights_He(weights, 300, 30);
    visualize_weights(weights, 300);
    return 0;
}

