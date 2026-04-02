#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 64

long int vector_sum_cpu(const long int* A, int N) {
    long int sum = 0;
    for (int i = 0; i < N; i++)
        sum += A[i];
    return sum;
}

__global__ void vector_sum_gpu(const long int* A, long int* res, int N) {
    __shared__ long int block_data[BLOCK_SIZE];

    int thread_id = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(global_id < N)
        block_data[thread_id] = A[global_id];
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

void run_vector_sum_time_test(int N) {
    printf("\nN = %d\n", N);

    size_t bytes = (size_t)N * sizeof(long int);

    // CPU calculation
    long int* cpu_vect;
    //long int* cpu_vect = (long int*)malloc(bytes);
    cudaMallocHost(&cpu_vect, bytes);

    for (int i = 0; i < N; i++)
        cpu_vect[i] = 1;

    clock_t cpu_start_time = clock();
    long int cpu_sum = vector_sum_cpu(cpu_vect, N);
    float cpu_time = (float)(clock() - cpu_start_time) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU vector sum: %ld\n", cpu_sum);
    printf("CPU time: %fms\n", cpu_time);

    
    // GPU calculation
    int max_grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    long int* gpu_vect;
    cudaMalloc(&gpu_vect, bytes);

    cudaEvent_t gpu_start_time, gpu_end_time;
    cudaEventCreate(&gpu_start_time);
    cudaEventCreate(&gpu_end_time);


    // copy array to the gpu
    cudaEventRecord(gpu_start_time);

    cudaMemcpy(gpu_vect, cpu_vect, bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(gpu_end_time);
    cudaEventSynchronize(gpu_end_time);

    float gpu_copy_values_time = 0.0;
    cudaEventElapsedTime(&gpu_copy_values_time, gpu_start_time, gpu_end_time);
    printf("GPU copy values time: %fms\n", gpu_copy_values_time);


    // calc in the kernel gpu
    cudaEventRecord(gpu_start_time);

    int cur_size = N;
    int grid_size = (cur_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_sum_gpu<<<grid_size, BLOCK_SIZE>>>(gpu_vect, gpu_vect, cur_size);
    cudaDeviceSynchronize();
    cur_size = grid_size;

    while (cur_size > 1) {
        grid_size = (cur_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vector_sum_gpu<<<grid_size, BLOCK_SIZE>>>(gpu_vect, gpu_vect, cur_size);
        cudaDeviceSynchronize();
        cur_size = grid_size;
    }

    cudaEventRecord(gpu_end_time);
    cudaEventSynchronize(gpu_end_time);

    float gpu_calc_time = 0.0;
    cudaEventElapsedTime(&gpu_calc_time, gpu_start_time, gpu_end_time);
    printf("GPU calc time: %fms\n", gpu_calc_time);


    // copy result to the cpu
    cudaEventRecord(gpu_start_time);

    long int gpu_sum = 0;
    cudaMemcpy(&gpu_sum, gpu_vect, sizeof(long int), cudaMemcpyDeviceToHost);
    printf("GPU vector sum: %ld\n", gpu_sum);

    cudaEventRecord(gpu_end_time);
    cudaEventSynchronize(gpu_end_time);

    float gpu_copy_result_time = 0.0;
    cudaEventElapsedTime(&gpu_copy_result_time, gpu_start_time, gpu_end_time);
    printf("GPU copy result time: %fms\n", gpu_copy_result_time);

    float gpu_full_time = gpu_copy_result_time + gpu_copy_values_time + gpu_calc_time;
    printf("GPU full time: %fms\n", gpu_full_time);


    // show acceleration
    printf("Acceleration calc: %f\n", cpu_time / gpu_calc_time);
    printf("Acceleration in general: %f\n", cpu_time / gpu_full_time);

    cudaEventDestroy(gpu_start_time);
    cudaEventDestroy(gpu_end_time);
    cudaFree(gpu_vect);
    cudaFreeHost(cpu_vect);
}

int main() {
    long int sizes[] = {100000, 1000000, 10000000, 100000000, 1000000000};

    // cuda init
    cudaEvent_t init_start_time, init_end_time;
    cudaEventCreate(&init_start_time);
    cudaEventCreate(&init_end_time);

    cudaEventRecord(init_start_time);
    long int *start_vect;
    cudaMalloc(&start_vect, sizeof(long int));

    dim3 block(1, 1);
    dim3 grid(1, 1);
    vector_sum_gpu<<<grid, block>>>(start_vect, start_vect, 1);
    cudaDeviceSynchronize();

    cudaEventRecord(init_end_time);
    cudaEventSynchronize(init_end_time);

    float cuda_init_time = 0.0;
    cudaEventElapsedTime(&cuda_init_time, init_start_time, init_end_time);
    printf("CUDA init time: %fms\n", cuda_init_time);

    cudaEventDestroy(init_start_time);
    cudaEventDestroy(init_end_time);

    cudaFree(start_vect);


    for (int i = 0; i < 5; i++) {
        run_vector_sum_time_test(sizes[i]);
    }

    return 0;
}