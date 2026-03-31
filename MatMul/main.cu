#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

void run_matmul_time_test(int N) {
    printf("\nN = %d\n", N);

    size_t bytes = (size_t)N * N * sizeof(float);

    float* cpu_A = (float*)malloc(bytes);
    float* cpu_B = (float*)malloc(bytes);
    float* cpu_C = (float*)malloc(bytes);

    for (int i = 0; i < (N * N); i++)
        cpu_A[i] = (float)rand() / RAND_MAX;

    for (int i = 0; i < (N * N); i++)
        cpu_B[i] = (float)rand() / RAND_MAX;

    clock_t cpu_start_time = clock();
    matmul_cpu(cpu_A, cpu_B, cpu_C, N);
    float cpu_ms = (float)(clock() - cpu_start_time) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU time: %fms\n", cpu_ms);

    free(cpu_A);
    free(cpu_B);
    free(cpu_C);
}

int main() {
    srand(42);

    int sizes[] = {100, 500, 1000, 1500, 2000};

    for (int i = 0; i < 5; i++) {
        run_matmul_time_test(sizes[i]);
    }

    return 0;
}