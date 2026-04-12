#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 64

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

void run_pi_calc_time_test(long int N) {
    printf("\nN = %ld\n", N);

    // CPU calculation
    clock_t cpu_start_time = clock();
    double cpu_pi = pi_calc_cpu(N);
    float cpu_time = (float)(clock() - cpu_start_time) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU pi: %.15f\n", cpu_pi);
    printf("CPU time: %fms\n", cpu_time);
}

int main() {
    long int N[] = {100000, 1000000, 10000000, 100000000, 1000000000};

    for (int i = 0; i < 5; i++) {
        run_pi_calc_time_test(N[i]);
    }

    return 0;
}