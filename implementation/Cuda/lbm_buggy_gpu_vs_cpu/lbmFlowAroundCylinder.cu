/*!
 * \file    lbmFlowAroundCylinder.cu
 * \brief   GPU (Cuda) and CPU version running the same code but producing different output...
 *          Especialy display a strange case for ITER=2.
 * \author  Adrien Python
 * \date    22.01.2017
 */

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <libgen.h>

#define RE       220.0       // Reynolds number
#define NX       4//420         // Numer of lattice nodes (width)
#define NY       9 //180         // Numer of lattice nodes (height)
#define LY       ((NY) - 1)  // Height of the domain in lattice units
#define CX       ((NX) / 4)  // X coordinates of the cylinder
#define CY       ((NY) / 2)  // Y coordinates of the cylinder
#define R        ((NY) / 9)  // Cylinder radius
#define ULB      0.04        // Velocity in lattice units
#define NULB     ((ULB) * (R) / (RE))   // Viscoscity in lattice units
#define OMEGA    ((double)1. / (3*(NULB)+0.5))  // Relaxation parameter

#define SQUARE(a) ((a)*(a))

typedef enum { OUT_FIN, OUT_IMG, OUT_UNK } out_mode;

typedef struct {
    int iteration;
    bool obstacles[NX][NY];  // Should reside in lbm_consts but is too big for constant memory
    double u[NX][NY][2];
    double feq[NX][NY][9];
    double fin[NX][NY][9];
    double fout[NX][NY][9];
    double rho[NX][NY];
    double vel[NX][NY][2];
} lbm_vars;

typedef struct {
    size_t col[3][3];
    size_t opp[9];
    ssize_t v[2][9];
    double t[9];
} lbm_consts;

#ifdef COMPUTE_ON_CPU
// Tweak the code to run on CPU
#define cudaMalloc(dst_ptr, size) do { *(dst_ptr) = (lbm_vars*)malloc(size); } while(0)
#define cudaMemcpy(dst, src, size, mode) memcpy(dst, src, size)
#define cudaMemcpyToSymbol(dst, src, size) memcpy(&dst, src, size)
#define cudaFree(ptr) free(ptr)

#define HANDLE_ERROR(ans) ans
#define HANDLE_KERNEL_ERROR(...) do { __VA_ARGS__; } while(0)

#define fory(...) for (int y = 0; y < NY; ++y) { __VA_ARGS__; }
#define forxy(...) fory(for (int x = 0; x < NX; ++x) { __VA_ARGS__; })

#define RUN_KERNEL_1D(kernel, th1, ...) fory(kernel(__VA_ARGS__, y))
#define RUN_KERNEL_2D(kernel, th1, th2, ...) forxy(kernel(__VA_ARGS__, x, y))

#else
// Code for GPU usage only
#define HANDLE_ERROR(ans) (handleError((ans), __FILE__, __LINE__))
inline void handleError(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
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

#define RUN_KERNEL_1D(kernel, th1, ...) HANDLE_KERNEL_ERROR( kernel<<<1, th1>>>(__VA_ARGS__) )
#define RUN_KERNEL_2D(kernel, th1, th2, ...) HANDLE_KERNEL_ERROR( kernel<<<1, th1*th2>>>(__VA_ARGS__) )
#endif

// Constants

#ifndef COMPUTE_ON_CPU
__constant__ 
#endif
lbm_consts d_consts;

const ssize_t V[][9] = {
    { 1, 1, 1, 0, 0, 0,-1,-1,-1 },
    { 1, 0,-1, 1, 0,-1, 1, 0,-1 }
};
const double T[] = { 1./36, 1./9, 1./36, 1./9, 4./9, 1./9, 1./36, 1./9, 1./36 };


/**
 * Setup: cylindrical obstacle and velocity inlet with perturbation
 * Creation of a mask with boolean values, defining the shape of the obstacle.
 */
static void initObstacles(lbm_vars* vars)
{
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            vars->obstacles[x][y] = SQUARE(x-CX) + SQUARE(y-CY) < SQUARE(R);
        }
    }
}

/**
 * Initial velocity profile: almost zero, with a slight perturbation to trigger
 * the instability.
 */
static void initVelocity(lbm_vars* vars)
{
    for (int d = 0; d < 2; d++) {
        for (int x = 0; x < NX; x++) {
            for (int y = 0; y < NY; y++) {
                vars->vel[x][y][d] = (1-d) * ULB * (1 + 0.0001 * sin( y / (double)LY * 2 * M_PI) );
            }
        }
    }
}

static void initRho(lbm_vars* vars)
{
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            vars->rho[x][y] = 1.0;
        }
    }
}

static void initCol(size_t* col, ssize_t v0)
{
    for (int f = 0, i = 0; f < 9 && i < 3; f++) {
        if (V[0][f] == v0) {
            col[i++] = f;
        }
    }
}

static void initOpp(size_t* opp)
{
    for (int f = 0; f < 9; f++) {
        for (int g = 0; g < 9; g++) {
            if (V[0][f] == -V[0][g] && V[1][f] == -V[1][g]) {
                opp[f] = g;
                break;
            }
        }
    }
}

#define EQUILIBRIUM_BODY(v, t) \
    do {                                                                 \
        double usqr = 3./2 * ( SQUARE(u[0]) + SQUARE(u[1]) );            \
                                                                         \
        for (int f = 0; f < 9; f++) {                                    \
            double cu = 3 * ( v[0][f] * u[0] + v[1][f] * u[1] );         \
            feq[f] = rho * t[f] * ( 1 + cu + 0.5 * SQUARE(cu) - usqr );  \
        }                                                                \
    } while(0)

#ifndef COMPUTE_ON_CPU
__host__ 
#endif
static void h_equilibrium(double* feq, double rho, double* u)
{
    EQUILIBRIUM_BODY(V, T);
}

#ifndef COMPUTE_ON_CPU
__device__ 
#endif
static void d_equilibrium(double* feq, double rho, double* u)
{
    EQUILIBRIUM_BODY(d_consts.v, d_consts.t);
}

#ifndef COMPUTE_ON_CPU
__device__ 
#endif
static void macroscopic(double* fin, double* rho, double* u)
{
    
    *rho = u[0] = u[1] = 0;

    for (int f = 0; f < 9; f++) {
        *rho += fin[f];

        u[0] += d_consts.v[0][f] * fin[f];
        u[1] += d_consts.v[1][f] * fin[f];
    }
    
    u[0] /= *rho;
    u[1] /= *rho;
}

#ifndef COMPUTE_ON_CPU
__global__ void lbm_right_wall(lbm_vars *d_vars)
#else
void lbm_right_wall(lbm_vars *d_vars, int y)
#endif
{
#ifndef COMPUTE_ON_CPU
    int y = threadIdx.x;
#endif

    // Right wall: outflow condition.
    for (int i = 0; i < 3; i++) {
        int f = d_consts.col[2][i];
        d_vars->fin[NX-1][y][f] = d_vars->fin[NX-2][y][f]; // TODO Test: retirer condition bord
    }
}

#ifndef COMPUTE_ON_CPU
__global__ void lbm_macro_and_left_wall(lbm_vars *d_vars)
#else
void lbm_macro_and_left_wall(lbm_vars *d_vars, int x, int y)
#endif
{
#ifndef COMPUTE_ON_CPU
    int y = threadIdx.x / NX;
    int x = threadIdx.x % NX;
#endif

    // Compute macroscopic variables, density and velocity
    macroscopic(d_vars->fin[x][y], &d_vars->rho[x][y], d_vars->u[x][y]);
    
    // Left wall: inflow condition
    for (size_t d = 0; d < 2; d++) {
        d_vars->u[0][y][d] = d_vars->vel[0][y][d]; // TODO: collision!
    }
 }

#ifndef COMPUTE_ON_CPU
__global__ void lbm_density(lbm_vars *d_vars)
#else
void lbm_density(lbm_vars *d_vars, int y)
#endif
{
#ifndef COMPUTE_ON_CPU
    int y = threadIdx.x;
#endif
    // Calculate the density
    double s2 = 0, s3 = 0;
    for (size_t i = 0; i < 3; i++) {
        s2 += d_vars->fin[0][y][d_consts.col[1][i]];
        s3 += d_vars->fin[0][y][d_consts.col[2][i]];
    }
    d_vars->rho[0][y] = 1./(1 - d_vars->u[0][y][0]) * (s2 + 2*s3);
}

#ifndef COMPUTE_ON_CPU
__global__ void lbm_equilibrium_1(lbm_vars *d_vars)
#else
void lbm_equilibrium_1(lbm_vars *d_vars, int x, int y)
#endif
{
#ifndef COMPUTE_ON_CPU
    int y = threadIdx.x / NX;
    int x = threadIdx.x % NX;
#endif
   
    // Compute equilibrium
    d_equilibrium(d_vars->feq[x][y], d_vars->rho[x][y], d_vars->u[x][y]);
}

#ifndef COMPUTE_ON_CPU
__global__ void lbm_equilibrium_2(lbm_vars *d_vars)
#else 
void lbm_equilibrium_2(lbm_vars *d_vars, int y)
#endif
{
#ifndef COMPUTE_ON_CPU
    int y = threadIdx.x;
#endif

    for (size_t i = 0, f = d_consts.col[0][i]; i < 3; f = d_consts.col[0][++i]) {
        d_vars->fin[0][y][f] = d_vars->feq[0][y][f] + d_vars->fin[0][y][d_consts.opp[f]] - d_vars->feq[0][y][d_consts.opp[f]];
    }

}

#ifndef COMPUTE_ON_CPU
__global__ void lbm_collision(lbm_vars *d_vars)
#else
void lbm_collision(lbm_vars *d_vars, int x, int y)
#endif
{
#ifndef COMPUTE_ON_CPU
    int y = threadIdx.x / NX;
    int x = threadIdx.x % NX;
#endif
    
    for (size_t f = 0; f < 9; f++) {
        if (d_vars->obstacles[x][y]) {
            // Bounce-back condition for obstacle
            d_vars->fout[x][y][f] = d_vars->fin[x][y][d_consts.opp[f]];
        } else {
            double omega = OMEGA;
            // Collision step
            d_vars->fout[x][y][f] = d_vars->fin[x][y][f] - omega * (d_vars->fin[x][y][f] - d_vars->feq[x][y][f]);

            double ftmp = OMEGA * (d_vars->fin[x][y][f] - d_vars->feq[x][y][f]);

            // On the second iteration, these index have different computation results
            if (d_vars->iteration == 1 && x==0 && y==5 && (f==2||f==6)) {
// This line produce a strange behaviour (at least with ITER=2):
//  - Commented:   The results between GPU and CPU are different.
//  - Uncommented: The results are the same.
//                printf("\tfin - OMEGA*(fin - feq) = :%64.100f\n", d_vars->fin[x][y][f] - ftmp);

                printf("%d,%d,%lu:\n", x,y,f);
                printf("\tfin  = %64.100f (%lu)\n", d_vars->fin[x][y][f], *((long*)(&d_vars->fin[x][y][f])));
                printf("\tfeq  = %64.100f (%lu)\n", d_vars->feq[x][y][f], *((long*)(&d_vars->feq[x][y][f])));
                printf("\tfout = %64.100f (%lu)\n", d_vars->fout[x][y][f], *((long*)(&d_vars->fout[x][y][f])));

// fout have the same result on both CPU and GPU if the computation is split:
// Uncomment to see that results are different in that configuration.
/*
                printf("\n\tSplit computation results:\n");
                printf("\tfin - feq = %64.100f\n", (d_vars->fin[x][y][f] - d_vars->feq[x][y][f]));
                double ftmp = OMEGA * (d_vars->fin[x][y][f] - d_vars->feq[x][y][f]);
                printf("\tftmp = OMEGA*(fin - feq) = %64.100f\n", ftmp);
                printf("\tfout = fin - ftmp = %64.100f\n", d_vars->fin[x][y][f] - ftmp);
*/
            }

        }
    }

}

#ifndef COMPUTE_ON_CPU
__global__ void lbm_streaming(lbm_vars *d_vars)
#else
void lbm_streaming(lbm_vars *d_vars, int x, int y)
#endif
{
#ifndef COMPUTE_ON_CPU
    int y = threadIdx.x / NX;
    int x = threadIdx.x % NX;
#endif
    // Streaming step
    for (size_t f = 0; f < 9; f++) {
        size_t x_dst = (x + NX + d_consts.v[0][f]) % NX;
        size_t y_dst = (y + NY + d_consts.v[1][f]) % NY;
        d_vars->fin[x_dst][y_dst][f] = d_vars->fout[x][y][f];
    }

    // Count iterations
    if (x == 0 && y == 0) {
        d_vars->iteration++;
    }
}

int main(int argc, char * const argv[])
{
    // Read arguments
    ssize_t max_iter = 0;
    
    while (optind < argc) {
        switch (getopt(argc, argv, "i:")) {
            case 'i': { max_iter = strtol(optarg, NULL, 10); break; }
            default : { goto usage; }
        }
    }
    
    // check that execution mode is set
    if (max_iter < 1) {
    usage:
        fprintf(stderr, "usage: %s -i <iter> \n", basename((char*)argv[0]));
        fprintf(stderr, "  -i : Total number of iterations\n");
        return EXIT_FAILURE;
    }

    lbm_consts* h_consts = (lbm_consts*)malloc(sizeof(lbm_consts));
    
    initCol(h_consts->col[0],  1);
    initCol(h_consts->col[1],  0);
    initCol(h_consts->col[2], -1);
    initOpp(h_consts->opp);
    memcpy(h_consts->v, V, sizeof(V));
    memcpy(h_consts->t, T, sizeof(T));
    
    HANDLE_ERROR(cudaMemcpyToSymbol(d_consts, h_consts, sizeof(lbm_consts)));
        
    lbm_vars *h_vars = (lbm_vars*)malloc(sizeof(lbm_vars));
    h_vars->iteration = 0;
    initObstacles(h_vars);
    initVelocity(h_vars);
    initRho(h_vars);
    
    // Initialization of the populations at equilibrium with the given velocity.
    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            h_equilibrium(h_vars->fin[x][y], h_vars->rho[x][y], h_vars->vel[x][y]);
        }
    }
   
    lbm_vars *d_vars;
    HANDLE_ERROR(cudaMalloc(&d_vars, sizeof(lbm_vars)));
    HANDLE_ERROR(cudaMemcpy(d_vars, h_vars, sizeof(lbm_vars), cudaMemcpyHostToDevice));
    for (int time = 0; time < max_iter; time++) {
        RUN_KERNEL_1D(lbm_right_wall,             NY, d_vars);
        RUN_KERNEL_2D(lbm_macro_and_left_wall, NX,NY, d_vars);
        RUN_KERNEL_1D(lbm_density,                NY, d_vars);
        RUN_KERNEL_2D(lbm_equilibrium_1,       NX,NY, d_vars);
        RUN_KERNEL_1D(lbm_equilibrium_2,          NY, d_vars);
        // ITER=2: 
        // fin  OK
        // fout OK
        // feq  OK
        RUN_KERNEL_2D(lbm_collision,           NX,NY, d_vars);
       // ITER=2: 
        // fin  OK
        // fout KO
        // feq  OK
        RUN_KERNEL_2D(lbm_streaming,           NX,NY, d_vars);
    }

    free(h_consts);
    free(h_vars);
    HANDLE_ERROR(cudaFree(d_vars));
    
    return EXIT_SUCCESS;
}
