#include <stdio.h>

__global__
void saxpy(int n, double a, double *x, double *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

#define BLOCK_SIZE 64

int main(int argc, const char * argv[])
{
  if (argc != 2) {
    fprintf(stderr, "usage: %s <n>\n", argv[0]);
  }

  int N = atoi(argv[1]);
  double *x, *y, *d_x, *d_y;
  x = (double*)malloc(N*sizeof(double));
  y = (double*)malloc(N*sizeof(double));

  cudaMalloc(&d_x, N*sizeof(double)); 
  cudaMalloc(&d_y, N*sizeof(double));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaEventRecord(stop);
  cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, N*sizeof(double), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  double maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i]-4.0f));
  }
  if (maxError > 0)
    printf("Max error: %f\n", maxError);
  printf("%f\n", N*sizeof(double)/milliseconds/1e6);
}
