#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <fstream>
#include <cstdlib>

#define NUM_POINTS 500
#define POPULATION_SIZE 1000
#define DEGREE 5
#define BLOCK_SIZE 1024
#define NUM_THREADS (BLOCK_SIZE * 256)

struct Individual {
    float params[DEGREE];
};

__global__ void init_curand(curandState* states) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(time(NULL), global_id, 0, &states[global_id]);
}

void gen_coeff(int* coeff, int degree, int coef_range) {
    printf("Coefficients:\n");
    for (int i = 0; i < degree; i++) {
        coeff[i] = (int)(rand() % (2 * coef_range + 1) - coef_range);
        printf("c%d = %d\n", i, coeff[i]);
    }
}

void gen_points(float* x_coord, float* y_coord, int num_points, int degree, int* coeffs) {
    float x, y, x_pow;

    for (int i = 0; i < num_points; i++) {
        x = i;
        y = 0.0;
        x_pow = 1.0;

        for (int j = 0; j <= degree; j++) {
            y += coeffs[j] * x_pow;
            x_pow *= x;
        }

        x_coord[i] = x;
        y_coord[i] = y;
    }
}

int main() {
    srand(time(NULL));

    cuda init
    curandState* states;
    cudaMalloc(&states, NUM_THREADS * sizeof(curandState));

    cudaEvent_t init_start_time, init_end_time;
    cudaEventCreate(&init_start_time);
    cudaEventCreate(&init_end_time);

    cudaEventRecord(init_start_time);
    init_curand<<<NUM_THREADS / BLOCK_SIZE, BLOCK_SIZE>>>(states);
    cudaDeviceSynchronize();

    cudaEventRecord(init_end_time);
    cudaEventSynchronize(init_end_time);

    float cuda_init_time = 0.0;
    cudaEventElapsedTime(&cuda_init_time, init_start_time, init_end_time);
    printf("CUDA curand init time: %fms\n", cuda_init_time);

    cudaEventDestroy(init_start_time);
    cudaEventDestroy(init_end_time);


    // generate coefficients and points
    int* coeff = (int*)malloc((size_t)DEGREE * sizeof(int));
    float* x_coord = (float*)malloc((size_t)NUM_POINTS * sizeof(float));
    float* y_coord = (float*)malloc((size_t)NUM_POINTS * sizeof(float));

    gen_coeff(coeff, DEGREE, 10);
    gen_points(x_coord, y_coord, NUM_POINTS, DEGREE, coeff);


    // copy points to GPU
    float* gpu_x, gpu_y;
    cudaMalloc(&gpu_x, NUM_POINTS * sizeof(float));
    cudaMalloc(&gpu_y, NUM_POINTS * sizeof(float));
    cudaMemcpy(gpu_x, x_coord, NUM_POINTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_y, y_coord, NUM_POINTS * sizeof(float), cudaMemcpyHostToDevice);
    
    // initial population
    Individual* population;
    cudaMemset(population, 0, POPULATION_SIZE * sizeof(Individual));

    cudaFree(states);
    return 0;
}