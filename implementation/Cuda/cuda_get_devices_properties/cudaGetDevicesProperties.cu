/*!
 * \file    cudaGetDevicesProperties.cu
 * \author  Cuda By Example Book
 */

#include <stdio.h>

#define HANDLE_ERROR(ans) (handleError((ans), __FILE__, __LINE__))
inline void handleError(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(EXIT_FAILURE);
   }
}

int main(int argc, char * const argv[])
{
    cudaDeviceProp prop;

    int count;
    HANDLE_ERROR( cudaGetDeviceCount(&count) );

    for (int i = 0; i < count; ++i) {
        HANDLE_ERROR( cudaGetDeviceProperties(&prop, i) );
        printf("  --- General Information for device %d ---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability:   %d.%d\n", prop.major, prop.minor);
        printf("Clock rate:  %d\n", prop.clockRate);
        printf("Device copy overlap:  %s\n", prop.deviceOverlap ? "Eneable" : "Disable");
        printf("Kernel execution timeout: %s\n", prop.kernelExecTimeoutEnabled ? "Eneable" : "Disable");
        printf("  --- Memory Information for device %d ---\n", i);
        printf("Total global mem: %ld\n", prop.totalGlobalMem);
        printf("Total const mem: %ld\n", prop.totalConstMem);
        printf("Max mem pitch: %ld\n", prop.memPitch);
        printf("Texture Alignment: %ld\n", prop.textureAlignment);
        printf("  --- MP Information for device %d ---\n", i);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
        printf("Registers per mp: %d\n", prop.regsPerBlock);
        printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max thread per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max threads dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }
    
    return EXIT_SUCCESS;
}
