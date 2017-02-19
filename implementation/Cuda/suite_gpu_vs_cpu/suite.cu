/*!
 * \file    suite.cu
 * \brief   GPU (Cuda) and CPU code computing a simple mathematical suite (for 0<i<max_i): Σ(1/n^i)
 * \author  Adrien Python
 * \date    17.02.2017
 */

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <libgen.h>

#ifdef COMPUTE_ON_CPU

// Tweak the code to run on CPU
#define genericMalloc(dst_ptr, type) do { *(dst_ptr) = (type*)malloc(sizeof(type)); } while(0)
#define cudaMemcpy(dst, src, size, mode) memcpy(dst, src, size)
#define cudaMemcpyToSymbol(dst, src, size) memcpy(&dst, src, size)
#define cudaFree(ptr) free(ptr)
#define HANDLE_ERROR(ans) ans
#define HANDLE_KERNEL_ERROR(...) do { __VA_ARGS__; } while(0)
#define RUN_KERNEL(kernel, ...) kernel(__VA_ARGS__)

#else

// Code for GPU usage only
#define genericMalloc(dst_ptr, type) cudaMalloc(dst_ptr, sizeof(type))
#define HANDLE_ERROR(ans) (handleError((ans), __FILE__, __LINE__))
inline void handleError(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(EXIT_FAILURE);
   }
}

#define HANDLE_KERNEL_ERROR(...) \
do {                                         \
    __VA_ARGS__;                             \
    HANDLE_ERROR( cudaPeekAtLastError() );   \
    HANDLE_ERROR( cudaDeviceSynchronize() ); \
} while(0)

#define RUN_KERNEL(kernel, ...) HANDLE_KERNEL_ERROR( kernel<<<1, 1>>>(__VA_ARGS__) )
#endif

#ifndef COMPUTE_ON_CPU
__device__
#endif
double powd(double x, double y)
{
    double result = 1;
    for (int i = 0; i < y; ++i) {
        result *= x;
    }
    return result;
}

#ifdef COMPUTE_ON_CPU
void suite(double* d_result, long* d_n, size_t* d_i)
#else
__global__ void suite(double* d_result, long* d_n, size_t* d_i)
#endif
{
    *d_result += 1.0 / powd(*d_n, *d_i);
}

int main(int argc, char * const argv[])
{
    // Read arguments
    if (argc != 3) {
        fprintf(stderr, "usage: %s <n> <max_i>\n", basename(argv[0]));
        return EXIT_FAILURE;
    }

    long* d_n, n = strtol(argv[1], NULL, 10);
    int max_i = strtol(argv[2], NULL, 10);
    double *d_result, result = 0;

    HANDLE_ERROR(genericMalloc(&d_n, long));
    HANDLE_ERROR(cudaMemcpy(d_n, &n, sizeof(long), cudaMemcpyHostToDevice));
    HANDLE_ERROR(genericMalloc(&d_result, double));
    HANDLE_ERROR(cudaMemcpy(d_result, &result, sizeof(double), cudaMemcpyHostToDevice));

    size_t* d_i;
    HANDLE_ERROR(genericMalloc(&d_i, size_t));

    for (size_t i = 1; i < max_i; i++) {
        HANDLE_ERROR(cudaMemcpy(d_i, &i, sizeof(size_t), cudaMemcpyHostToDevice));
        RUN_KERNEL(suite, d_result, d_n, d_i);
    }

    HANDLE_ERROR(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));

#ifdef COMPUTE_ON_CPU
    printf("on cpu, with 0<i<%d: Σ(1/%lu^i) = %64.60f\n", max_i, n, result);
#else
    printf("on gpu, with 0<i<%d: Σ(1/%lu^i) = %64.60f\n", max_i, n, result);
#endif

    HANDLE_ERROR(cudaFree(d_result));
    
    return EXIT_SUCCESS;
}
