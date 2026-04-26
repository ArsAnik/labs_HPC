#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <fstream>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define NUM_POINTS 1000
#define POPULATION_SIZE 2000
#define DEGREE 5
#define BLOCK_SIZE 256
#define NUM_THREADS (BLOCK_SIZE * 256)
#define MAX_CONST_ITER 1000

#define INIT_RANGE 10
#define MUTATION_SIGMA 0.5
#define MUTATION_PROB 0.15

struct Individual {
    float params[DEGREE];
};

__global__ void init_curand(curandState* states, unsigned long long seed) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, global_id, 0, &states[global_id]);
}

__global__ void init_population(Individual* population, curandState* states, int population_size, int degree, float range) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= population_size) return;

    curandState local_state = states[global_id];
    
    for (int i = 0; i < degree; i++) {
        population[global_id].params[i] = curand_uniform(&local_state) * 2.0 * range - range;
    }
    
    states[global_id] = local_state;
}

__global__ void fitness(const Individual* population, const float* point_x, const float* point_y, 
                        float* fitnesses, int population_size, int num_points, int degree) {

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= population_size) return;

    const Individual& individ = population[global_id];
    double sum_error = 0.0;
    double x, f_approx, x_pow, diff;

    for (int i = 0; i < num_points; i++) {
        x = point_x[i];
        f_approx = 0.0;
        x_pow = 1.0;

        for (int j = 0; j < degree; j++) {
            f_approx += individ.params[j] * x_pow;
            x_pow *= x;
        }

        diff = f_approx - point_y[i];
        sum_error += diff * diff;
    }

    fitnesses[global_id] = sum_error;
}

__global__ void crossover(const Individual* population, Individual* new_population, 
                        curandState* states, int population_size, int degree, int save_range=20) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= population_size) return;

    if (global_id < save_range) {
        new_population[global_id] = population[global_id];
        return;
    }

    curandState local_state = states[global_id];
    // use only first half
    int half = population_size / 2;

    int p1 = (int)(curand_uniform(&local_state) * half) % half;
    int p2 = (int)(curand_uniform(&local_state) * half) % half;

    int crosspoint = curand(&local_state) % (degree - 1) + 1;

    Individual parent1 = population[p1];
    Individual parent2 = population[p2];
    Individual child;

    for (int i = 0; i < degree; i++) {
        if (i < crosspoint)
            child.params[i] = parent1.params[i];
        else
            child.params[i] = parent2.params[i];
    }

    new_population[global_id] = child;
    states[global_id] = local_state;
}

__global__ void mutation(Individual* population, curandState* states, int population_size,
                        float mutation_prob, float mutation_sigma, int save_range=20) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id == 0 || global_id >= population_size) return;

    if (global_id < save_range) return;

    curandState local_state = states[global_id];

    for (int i = 0; i < DEGREE; i++) {
        if (curand_uniform(&local_state) < mutation_prob) {
            float delta = curand_normal(&local_state) * mutation_sigma;
            population[global_id].params[i] += delta;
        }
    }

    states[global_id] = local_state;
}

void selection(Individual* population, float* fitnesses, int population_size) {
    thrust::device_ptr<float> fit_ptr(fitnesses);
    thrust::device_ptr<Individual> pop_ptr(population);
    thrust::sort_by_key(fit_ptr, fit_ptr + population_size, pop_ptr);
}


Individual genetic_algorithm(float* gpu_x, float* gpu_y, int max_iter, curandState* states, float best_fitness=1e30) {

    Individual* population;
    Individual* new_population;
    cudaMalloc(&population, POPULATION_SIZE * sizeof(Individual));
    cudaMalloc(&new_population, POPULATION_SIZE * sizeof(Individual));

    int num_blocks = (POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // initial population
    init_population<<<num_blocks, BLOCK_SIZE>>>(population, states, POPULATION_SIZE, DEGREE, INIT_RANGE);
    cudaDeviceSynchronize();

    float* gpu_fitnesses;
    cudaMalloc(&gpu_fitnesses, POPULATION_SIZE * sizeof(float));

    int const_iter = 0;

    for (int gen = 0; gen < max_iter; gen++) {
        fitness<<<num_blocks, BLOCK_SIZE>>>(population, gpu_x, gpu_y, gpu_fitnesses, POPULATION_SIZE, NUM_POINTS, DEGREE);
        cudaDeviceSynchronize();

        crossover<<<num_blocks, BLOCK_SIZE>>>(population, new_population, states, POPULATION_SIZE, DEGREE);
        cudaDeviceSynchronize();

        mutation<<<num_blocks, BLOCK_SIZE>>>(new_population, states, POPULATION_SIZE, MUTATION_PROB, MUTATION_SIGMA);
        cudaDeviceSynchronize();

        fitness<<<num_blocks, BLOCK_SIZE>>>(new_population, gpu_x, gpu_y, gpu_fitnesses, POPULATION_SIZE, NUM_POINTS, DEGREE);
        cudaDeviceSynchronize();
        
        selection(new_population, gpu_fitnesses, POPULATION_SIZE);
        cudaMemcpy(population, new_population, POPULATION_SIZE * sizeof(Individual), cudaMemcpyDeviceToDevice);

        float cur_best;
        cudaMemcpy(&cur_best, gpu_fitnesses, sizeof(float), cudaMemcpyDeviceToHost);

        if (cur_best < best_fitness) {
            best_fitness = cur_best;
            const_iter = 0;
        } else {
            const_iter++;
        }

        if (gen % 50 == 0 || const_iter >= MAX_CONST_ITER) {
            printf("Generation #%d - fitness: %f\n", gen, best_fitness);
        }

        if (const_iter >= MAX_CONST_ITER){
            printf("Last generation #%d - fitness: %f\n", gen, best_fitness);
            break;
        }
    }

    Individual best_ind;
    cudaMemcpy(&best_ind, population, sizeof(Individual), cudaMemcpyDeviceToHost);

    cudaFree(population);
    cudaFree(new_population);
    cudaFree(gpu_fitnesses);

    return best_ind;
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
        x = (float)i / num_points * 10 - 5;
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

    // cuda init
    curandState* states;
    cudaMalloc(&states, NUM_THREADS * sizeof(curandState));

    cudaEvent_t init_start_time, init_end_time;
    cudaEventCreate(&init_start_time);
    cudaEventCreate(&init_end_time);

    cudaEventRecord(init_start_time);
    init_curand<<<NUM_THREADS / BLOCK_SIZE, BLOCK_SIZE>>>(states, time(NULL));
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
    float* gpu_x;
    float* gpu_y;
    cudaMalloc(&gpu_x, NUM_POINTS * sizeof(float));
    cudaMalloc(&gpu_y, NUM_POINTS * sizeof(float));
    cudaMemcpy(gpu_x, x_coord, NUM_POINTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_y, y_coord, NUM_POINTS * sizeof(float), cudaMemcpyHostToDevice);


    Individual best_individ = genetic_algorithm(gpu_x, gpu_y, 2000, states);
    
    for (int i = 0; i < DEGREE; i++)
        printf("c%d = %f (true: %d)\n", i, best_individ.params[i], coeff[i]);

    cudaFree(states);
    cudaFree(gpu_x);
    cudaFree(gpu_y);
    free(coeff);
    free(x_coord);
    free(y_coord);

    return 0;
}