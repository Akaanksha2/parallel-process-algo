#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DAMPING_FACTOR 0.85f

__global__ void pagerank_kernel(float *d_A, float *d_R, float *d_sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0;
        for (int j = 0; j < N; j++) {
            sum += d_A[idx * N + j] * d_R[j];
        }
        d_sum[idx] = sum;
        d_R[idx] = (1 - DAMPING_FACTOR) / N + DAMPING_FACTOR * sum;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s input_file\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    char *input_file = argv[1];

    // Open file
    FILE *f = fopen(input_file, "r");
    if (f == NULL) {
        fprintf(stderr, "Failed to open file %s\n", input_file);
        exit(EXIT_FAILURE);
    }

    // Read matrix size
    int N;
    fscanf(f, "%d", &N);

    // Allocate host memory
    float *A = (float *) malloc(N * N * sizeof(float));
    float *R = (float *) malloc(N * sizeof(float));
    float *sum = (float *) malloc(N * sizeof(float));

    // Read matrix from file
    for (int i = 0; i < N * N; i++) {
        fscanf(f, "%f", &A[i]);
	//printf("Read A[%d] = %f\n", i, A[i]);	
    }

    // Initialize R to 1 / N
    for (int i = 0; i < N; i++) {
        R[i] = 1.0f / N;
    }

    // Allocate device memory
    float *d_A, *d_R, *d_sum;
    cudaMalloc((void **) &d_A, N * N * sizeof(float));
    cudaMalloc((void **) &d_R, N * sizeof(float));
    cudaMalloc((void **) &d_sum, N * sizeof(float));

    // Copy input to device memory
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, R, N * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel for 100 iterations
    for (int i = 0; i < 100; i++) {
        pagerank_kernel<<<(N + 255) / 256, 256>>>(d_A, d_R, d_sum, N);
        cudaMemcpy(R, d_R, N * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Print result
    for (int i = 0; i < N; i++) {
        printf("R[%d] = %f\n", i, R[i]);
    }

    // Free memory
    free(A);
    free(R);
    free(sum);
    cudaFree(d_A);
    cudaFree(d_R);
    cudaFree(d_sum);

    return 0;
}
