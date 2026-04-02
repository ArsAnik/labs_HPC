#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 16

void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

__global__ void matmul_gpu(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

bool validation_calc(const float* cpu, const float* gpu, int size, float eps = 1e-2f) {
    for (int i = 0; i < size; i++){
        if (fabsf(cpu[i] - gpu[i]) > eps) return false;
    }
    return true;
}

void run_matmul_time_test(int N) {
    printf("\nN = %d\n", N);

    size_t bytes = (size_t)N * N * sizeof(float);

    // CPU calculation
    float* cpu_A = (float*)malloc(bytes);
    float* cpu_B = (float*)malloc(bytes);
    float* cpu_C = (float*)malloc(bytes);

    for (int i = 0; i < (N * N); i++)
        cpu_A[i] = (float)rand() / RAND_MAX;

    for (int i = 0; i < (N * N); i++)
        cpu_B[i] = (float)rand() / RAND_MAX;

    clock_t cpu_start_time = clock();
    matmul_cpu(cpu_A, cpu_B, cpu_C, N);
    float cpu_time = (float)(clock() - cpu_start_time) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU time: %fms\n", cpu_time);

    
    // GPU calculation
    float* gpu_result_C = (float*)malloc(bytes);

    float *gpu_A, *gpu_B, *gpu_C;
    cudaMalloc(&gpu_A, bytes);
    cudaMalloc(&gpu_B, bytes);
    cudaMalloc(&gpu_C, bytes);

    cudaEvent_t gpu_start_time, gpu_end_time;
    cudaEventCreate(&gpu_start_time);
    cudaEventCreate(&gpu_end_time);


    // copy array to the gpu
    cudaEventRecord(gpu_start_time);

    cudaMemcpy(gpu_A, cpu_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, cpu_B, bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(gpu_end_time);
    cudaEventSynchronize(gpu_end_time);

    float gpu_copy_values_time = 0.0;
    cudaEventElapsedTime(&gpu_copy_values_time, gpu_start_time, gpu_end_time);
    printf("GPU copy values time: %fms\n", gpu_copy_values_time);


    // calc in the kernel gpu
    cudaEventRecord(gpu_start_time);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matmul_gpu<<<grid, block>>>(gpu_A, gpu_B, gpu_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(gpu_end_time);
    cudaEventSynchronize(gpu_end_time);

    float gpu_calc_time = 0.0;
    cudaEventElapsedTime(&gpu_calc_time, gpu_start_time, gpu_end_time);
    printf("GPU calc time: %fms\n", gpu_calc_time);


    // copy result to the cpu
    cudaEventRecord(gpu_start_time);
    cudaMemcpy(gpu_result_C, gpu_C, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(gpu_end_time);
    cudaEventSynchronize(gpu_end_time);

    float gpu_copy_result_time = 0.0;
    cudaEventElapsedTime(&gpu_copy_result_time, gpu_start_time, gpu_end_time);
    printf("GPU copy result time: %fms\n", gpu_copy_result_time);

    float gpu_full_time = gpu_copy_result_time + gpu_copy_values_time + gpu_calc_time;
    printf("GPU full time: %fms\n", gpu_full_time);


    // validation of calcuations 
    bool is_valid_calc = validation_calc(cpu_C, gpu_result_C, N * N);
    printf("Calculation are right: %s\n", is_valid_calc ? "Yes" : "No");


    // show acceleration
    printf("Acceleration calc: %f\n", cpu_time / gpu_calc_time);
    printf("Acceleration in general: %f\n", cpu_time / gpu_full_time);

    cudaEventDestroy(gpu_start_time);
    cudaEventDestroy(gpu_end_time);
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);

    free(gpu_result_C);
    free(cpu_A);
    free(cpu_B);
    free(cpu_C);
}

int main() {
    srand(42);

    int sizes[] = {100, 500, 1000, 1500, 2000};

    // cuda init
    cudaEvent_t init_start_time, init_end_time;
    cudaEventCreate(&init_start_time);
    cudaEventCreate(&init_end_time);

    cudaEventRecord(init_start_time);
    float *start_A, *start_B, *start_C;
    cudaMalloc(&start_A, sizeof(float));
    cudaMalloc(&start_B, sizeof(float));
    cudaMalloc(&start_C, sizeof(float));

    dim3 block(1, 1);
    dim3 grid(1, 1);
    matmul_gpu<<<grid, block>>>(start_A, start_B, start_C, 1);
    cudaDeviceSynchronize();

    cudaEventRecord(init_end_time);
    cudaEventSynchronize(init_end_time);

    float cuda_init_time = 0.0;
    cudaEventElapsedTime(&cuda_init_time, init_start_time, init_end_time);
    printf("CUDA init time: %fms\n", cuda_init_time);

    cudaEventDestroy(init_start_time);
    cudaEventDestroy(init_end_time);

    cudaFree(start_A);
    cudaFree(start_B);
    cudaFree(start_C);

    for (int i = 0; i < 5; i++) {
        run_matmul_time_test(sizes[i]);
    }

    return 0;
}