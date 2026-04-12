#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 1024
#define NUM_THREADS (BLOCK_SIZE * 256)

double pi_calc_cpu(long int N) {
    double x, y;
    long int inside_circle = 0;
    for (long int i = 0; i < N; i++) {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        if (x * x + y * y < 1.0)
            ++inside_circle;
    }
    return (double)(4.0 * inside_circle / N);
}

__global__ void init_curand(curandState* states) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(42, global_id, 0, &states[global_id]);
}

__global__ void pi_calc_gpu(curandState* states, double* res, long int N) {
    __shared__ double block_data[BLOCK_SIZE];

    int thread_id = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    double x, y;

    curandState cur_state = states[global_id];

    x = curand_uniform_double(&cur_state);
    y = curand_uniform_double(&cur_state);

    if((global_id < N) && (x * x + y * y < 1.0))
        block_data[thread_id] = 1;
    else
        block_data[thread_id] = 0;

    __syncthreads();

    res[global_id] = block_data[thread_id];
}

__global__ void reduce_gpu(double* res, long int N) {
    __shared__ double block_data[BLOCK_SIZE];

    int thread_id = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(global_id < N)
        block_data[thread_id] = res[global_id];
    else
        block_data[thread_id] = 0;
    __syncthreads();

    int part = blockDim.x / 2;
    while (part > 0) {
        if (thread_id < part)
            block_data[thread_id] += block_data[thread_id + part];

        __syncthreads();
        part = part / 2;
    }

    if (thread_id == 0)
        res[blockIdx.x] = block_data[0];
}

void run_pi_calc_time_test(long int N, curandState* states) {
    printf("\nN = %ld\n", N);


    // CPU calculation
    clock_t cpu_start_time = clock();
    double cpu_pi = pi_calc_cpu(N);
    float cpu_time = (float)(clock() - cpu_start_time) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU pi: %.15f\n", cpu_pi);
    printf("CPU time: %fms\n", cpu_time);


    // GPU calculation
    double* gpu_buf;
    cudaMalloc(&gpu_buf, N * sizeof(double));

    cudaEvent_t gpu_start_time, gpu_end_time;
    cudaEventCreate(&gpu_start_time);
    cudaEventCreate(&gpu_end_time);

    // calc in the kernel gpu
    cudaEventRecord(gpu_start_time);

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pi_calc_gpu<<<num_blocks, BLOCK_SIZE>>>(states, gpu_buf, N);
    cudaDeviceSynchronize();

    long int cur_size = N;
    int grid_size;

    while (cur_size > 1) {
        grid_size = (cur_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduce_gpu<<<grid_size, BLOCK_SIZE>>>(gpu_buf, cur_size);
        cur_size = grid_size;
    }
    cudaDeviceSynchronize();

    cudaEventRecord(gpu_end_time);
    cudaEventSynchronize(gpu_end_time);

    // copy result to the cpu
    double gpu_pi = 0;
    cudaMemcpy(&gpu_pi, gpu_buf, sizeof(double), cudaMemcpyDeviceToHost);
    gpu_pi = (double)(4.0 * gpu_pi / N);

    float gpu_time = 0.0;
    cudaEventElapsedTime(&gpu_time, gpu_start_time, gpu_end_time);
    printf("GPU pi: %.15f\n", gpu_pi);
    printf("GPU time: %fms\n", gpu_time);

    // show acceleration
    printf("Acceleration: %f\n", cpu_time / gpu_time);

    cudaEventDestroy(gpu_start_time);
    cudaEventDestroy(gpu_end_time);
    cudaFree(gpu_buf);
}

int main() {
    srand(42);

    // cuda init
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

    
    long int N[] = {100, 1000000, 10000000, 100000000, 1000000000};

    for (int i = 0; i < 5; i++) {
        run_pi_calc_time_test(N[i], states);
    }

    cudaFree(states);

    return 0;
}