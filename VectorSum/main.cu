#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 16

long int vector_sum_cpu(const long int* A, int N) {
    long int sum = 0;
    for (int i = 0; i < N; i++)
        sum += A[i];
    return sum;
}

__global__ void vector_sum_gpu(const long int* A, int N) {
    
}

void run_vector_sum_time_test(int N) {
    printf("\nN = %d\n", N);

    size_t bytes = (size_t)N * sizeof(long int);

    // CPU calculation
    long int* cpu_vect = (long int*)malloc(bytes);

    for (int i = 0; i < N; i++)
        cpu_vect[i] = 1;

    clock_t cpu_start_time = clock();
    long int cpu_sum = vector_sum_cpu(cpu_vect, N);
    float cpu_time = (float)(clock() - cpu_start_time) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU vector sum: %ld\n", cpu_sum);
    printf("CPU time: %fms\n", cpu_time);

    
    // GPU calculation
    // float* gpu_result_C = (float*)malloc(bytes);

    // float *gpu_A, *gpu_B, *gpu_C;
    // cudaMalloc(&gpu_A, bytes);
    // cudaMalloc(&gpu_B, bytes);
    // cudaMalloc(&gpu_C, bytes);

    // cudaEvent_t gpu_start_time, gpu_end_time;
    // cudaEventCreate(&gpu_start_time);
    // cudaEventCreate(&gpu_end_time);


    // // copy array to the gpu
    // cudaEventRecord(gpu_start_time);

    // cudaMemcpy(gpu_A, cpu_A, bytes, cudaMemcpyHostToDevice);
    // cudaMemcpy(gpu_B, cpu_B, bytes, cudaMemcpyHostToDevice);

    // cudaEventRecord(gpu_end_time);
    // cudaEventSynchronize(gpu_end_time);

    // float gpu_copy_values_time = 0.0;
    // cudaEventElapsedTime(&gpu_copy_values_time, gpu_start_time, gpu_end_time);
    // printf("GPU copy values time: %fms\n", gpu_copy_values_time);


    // // calc in the kernel gpu
    // cudaEventRecord(gpu_start_time);

    // dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    // matmul_gpu<<<grid, block>>>(gpu_A, gpu_B, gpu_C, N);
    // cudaDeviceSynchronize();

    // cudaEventRecord(gpu_end_time);
    // cudaEventSynchronize(gpu_end_time);

    // float gpu_calc_time = 0.0;
    // cudaEventElapsedTime(&gpu_calc_time, gpu_start_time, gpu_end_time);
    // printf("GPU calc time: %fms\n", gpu_calc_time);


    // // copy result to the cpu
    // cudaEventRecord(gpu_start_time);
    // cudaMemcpy(gpu_result_C, gpu_C, bytes, cudaMemcpyDeviceToHost);

    // cudaEventRecord(gpu_end_time);
    // cudaEventSynchronize(gpu_end_time);

    // float gpu_copy_result_time = 0.0;
    // cudaEventElapsedTime(&gpu_copy_result_time, gpu_start_time, gpu_end_time);
    // printf("GPU copy result time: %fms\n", gpu_copy_result_time);

    // float gpu_full_time = gpu_copy_result_time + gpu_copy_values_time + gpu_calc_time;
    // printf("GPU full time: %fms\n", gpu_full_time);


    // // validation of calcuations 
    // bool is_valid_calc = validation_calc(cpu_C, gpu_result_C, N * N);
    // printf("Calculation are right: %s\n", is_valid_calc ? "Yes" : "No");


    // // show acceleration
    // printf("Acceleration calc: %f\n", cpu_time / gpu_calc_time);
    // printf("Acceleration in general: %f\n", cpu_time / gpu_full_time);

    // cudaEventDestroy(gpu_start_time);
    // cudaEventDestroy(gpu_end_time);
    // cudaFree(gpu_A);
    // cudaFree(gpu_B);
    // cudaFree(gpu_C);

    // free(gpu_result_C);
    free(cpu_vect);
}

int main() {
    long int sizes[] = {100000, 1000000, 10000000, 100000000, 1000000000};

    // cuda init
    // cudaEvent_t init_start_time, init_end_time;
    // cudaEventCreate(&init_start_time);
    // cudaEventCreate(&init_end_time);

    // cudaEventRecord(init_start_time);
    // float *start_A, *start_B, *start_C;
    // cudaMalloc(&start_A, sizeof(float));
    // cudaMalloc(&start_B, sizeof(float));
    // cudaMalloc(&start_C, sizeof(float));

    // dim3 block(1, 1);
    // dim3 grid(1, 1);
    // matmul_gpu<<<grid, block>>>(start_A, start_B, start_C, 1);
    // cudaDeviceSynchronize();

    // cudaEventRecord(init_end_time);
    // cudaEventSynchronize(init_end_time);

    // float cuda_init_time = 0.0;
    // cudaEventElapsedTime(&cuda_init_time, init_start_time, init_end_time);
    // printf("CUDA init time: %fms\n", cuda_init_time);

    // cudaEventDestroy(init_start_time);
    // cudaEventDestroy(init_end_time);

    // cudaFree(start_A);
    // cudaFree(start_B);
    // cudaFree(start_C);

    for (int i = 0; i < 5; i++) {
        run_vector_sum_time_test(sizes[i]);
    }

    return 0;
}