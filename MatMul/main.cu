#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main() {

    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("GPU name: %s \n\n", p.name);

    float val = 0.0, out = 0.0;
    float* c_val;

    cudaError_t err = cudaMalloc(&c_val, sizeof(float));
    if (err != cudaSuccess) {
        printf("err cudaMalloc: %s\n\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(c_val, &val, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&out, c_val, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(c_val);

    if (out == val)
        printf("ОК!\n");

    return 0;
}
