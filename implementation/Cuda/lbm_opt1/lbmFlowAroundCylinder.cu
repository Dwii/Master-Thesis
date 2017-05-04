/*!
 * \file    lbmFlowAroundCylinder.cu
 * \brief   Cuda version based on lbm_palabos_friendly (standard C).
 * \author  Adrien Python
 * \date    22.01.2017
 */

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <libgen.h>
#include <pgm.h>
#include <timing.h>

#define RE       220.0       // Reynolds number
#define NX       420         // Numer of lattice nodes (width)
#define NY       180         // Numer of lattice nodes (height)
#define LY       ((NY) - 1)  // Height of the domain in lattice units
#define CX       ((NX) / 4)  // X coordinates of the cylinder
#define CY       ((NY) / 2)  // Y coordinates of the cylinder
#define R        ((NY) / 9)  // Cylinder radius
#define ULB      0.04        // Velocity in lattice units
#define NULB     ((ULB) * (R) / (RE))   // Viscoscity in lattice units
#define OMEGA    ((double)1. / (3*(NULB)+0.5))  // Relaxation parameter

#define NB_BLOCKS    1
//#define NB_THREADS 100

#define SQUARE(a) ((a)*(a))
#define GPU_SQUARE(a) (__dmul_rn(a,a))
#define INDEX_2D_FROM_1D(x, y, i) do { (y) = (i)/(NX), (x) = (i)%(NX); } while (0)

typedef enum { OUT_NONE, OUT_FIN, OUT_IMG } out_mode;

typedef struct {
    bool obstacles[NX][NY];  // Should reside in lbm_consts but is too big for constant memory
    double u[NX][NY][2];
    double fin[NX][NY][9];
    double fout[NX][NY][9];
} lbm_vars;

typedef struct {
    size_t col[3][3];
    size_t opp[9];
    ssize_t v[2][9];
    double t[9];
    double vel[NY];
} lbm_consts;

__constant__ lbm_consts d_consts;

const ssize_t V[][9] = {
    { 1, 1, 1, 0, 0, 0,-1,-1,-1 },
    { 1, 0,-1, 1, 0,-1, 1, 0,-1 }
};
const double T[] = { 1./36, 1./9, 1./36, 1./9, 4./9, 1./9, 1./36, 1./9, 1./36 };

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
static void initVelocity(double* vel)
{
    for (int y = 0; y < NY; y++) {
        vel[y] = ULB * (1 + 0.0001 * sin( y / (double)LY * 2 * M_PI) );
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

__host__ static void h_equilibrium(double* feq, double rho, double u)
{
    double usqr = 3./2 * ( SQUARE(u) );

    for (int f = 0; f < 9; f++) {
        double cu = 3 * ( V[0][f] * u );
        feq[f] = rho * T[f] * ( 1 + cu + 0.5 * SQUARE(cu) - usqr );
    }                                                                
}

__device__ static void d_equilibrium(double* feq, double rho, double u0, double u1)
{
    double usqr = __dmul_rn(3./2, __dadd_rn( GPU_SQUARE(u0), GPU_SQUARE(u1) ));
                                                                     
    for (int f = 0; f < 9; f++) {
        double cu = 3 * ( d_consts.v[0][f] * u0 + d_consts.v[1][f] * u1 );
        feq[f] = rho * d_consts.t[f] * ( 1 + cu + 0.5 * SQUARE(cu) - usqr );
    }                                                                
}

__device__ static void macroscopic(double* fin, double* rho, double* u0, double* u1)
{   
    *rho = *u0 = *u1 = 0;

    for (int f = 0; f < 9; f++) {
        *rho += fin[f];

        *u0 += d_consts.v[0][f] * fin[f];
        *u1 += d_consts.v[1][f] * fin[f];
    }

    *u0 /= *rho;
    *u1 /= *rho;
}

__global__ void lbm_computation(lbm_vars *d_vars)
{
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < NX*NY; idx += blockDim.x * gridDim.x) {
        int x, y;
        INDEX_2D_FROM_1D(x, y, idx);

        if (x == NX-1) {
            // Right wall: outflow condition.
            for (int i = 0; i < 3; i++) {
                int f = d_consts.col[2][i];
                d_vars->fin[NX-1][y][f] = d_vars->fin[NX-2][y][f];
            }
        }

        // Compute macroscopic variables, density and velocity
        double rho, u0, u1;
        macroscopic(d_vars->fin[x][y], &rho, &u0, &u1);

        if (x == 0) {
            // Left wall: inflow condition
            u0 = d_consts.vel[y];
            u1 = 0;

            // Calculate the density
            double s2 = 0, s3 = 0;
            for (size_t i = 0; i < 3; i++) {
                s2 += d_vars->fin[0][y][d_consts.col[1][i]];
                s3 += d_vars->fin[0][y][d_consts.col[2][i]];
            }
            rho = 1./(1 - u0) * (s2 + 2*s3);
        }

        // Compute equilibrium
        double feq[9];
        d_equilibrium(feq, rho, u0, u1);

        if (x == 0) {
            for (size_t i = 0, f = d_consts.col[0][i]; i < 3; f = d_consts.col[0][++i]) {
            	d_vars->fin[0][y][f] = feq[f] + d_vars->fin[0][y][d_consts.opp[f]] - feq[d_consts.opp[f]];
            }
        }

        for (size_t f = 0; f < 9; f++) {
            if (d_vars->obstacles[x][y]) {
                // Bounce-back condition for obstacle
                d_vars->fout[x][y][f] = d_vars->fin[x][y][d_consts.opp[f]];
            } else {
                // Collision step
                d_vars->fout[x][y][f] = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(d_vars->fin[x][y][f], - feq[f])), d_vars->fin[x][y][f]);
            }
        }

		d_vars->u[x][y][0] = u0;
		d_vars->u[x][y][1] = u1;
    }
}

__global__ void lbm_streaming(lbm_vars *d_vars)
{
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < NX*NY; idx += blockDim.x * gridDim.x) {
        int x, y;
        INDEX_2D_FROM_1D(x, y, idx);

        // Streaming step
        for (size_t f = 0; f < 9; f++) {
            size_t x_dst = (x + NX + d_consts.v[0][f]) % NX;
            size_t y_dst = (y + NY + d_consts.v[1][f]) % NY;
            d_vars->fin[x_dst][y_dst][f] = d_vars->fout[x][y][f];
        }
    }
}

void output_variables(char* filename, double var[NX][NY][9]) 
{
    FILE* file = fopen(filename, "w");
    for (size_t x = 0; x < NX; x++) {
        for (size_t y = 0; y < NY; y++) {
            for (size_t f = 0; f < 9; ++f) {
                fprintf(file, "%64.60f\n", var[x][y][f]);
            }
        }
    }
    fclose(file);
}

void output_image(char* filename, double u[NX][NY][2]) 
{
    pgm_image* pgm = pgm_create(NX, NY);

    for (size_t x = 0; x < NX; x++) {
        for (size_t y = 0; y < NY; y++) {
            double vel = sqrt( SQUARE(u[x][y][0]) + SQUARE(u[x][y][1]) );
            int color =  255 * min(vel * 10, 1.0);
            pgm_set_pixel(pgm, x, y, color);
        }
    }

    pgm_write(pgm, filename);
    pgm_destroy(pgm);
}

int getThreads(int width, int height) {
    int dev, threads;
    cudaDeviceProp prop;
    HANDLE_ERROR( cudaGetDevice(&dev) );
    HANDLE_ERROR( cudaGetDeviceProperties(&prop, dev) );

    int maxThreads = min(prop.maxThreadsDim[0], prop.maxThreadsPerBlock);
#ifdef NB_THREADS
    threads = NB_THREADS;
#else
    threads = prop.maxThreadsDim[0];
#endif

    if (threads > maxThreads)
        threads = maxThreads;

    return min(threads, width*height);
}

float get_lups(int lattices, int iterations, long ns_time_diff)
{
    return lattices * iterations * 1000000000.0f / ns_time_diff;
}

int main(int argc, char * const argv[])
{
    // Init options to default values
    const char* out_path = ".";
    const char* out_pref = "lbm";
    out_mode out = OUT_NONE;
    ssize_t max_iter = 0;
    size_t out_interval = 0;
    bool print_lups = false;
    bool print_avg_lups = false;

    // Read arguments
    while (optind < argc) {
        switch (getopt(argc, argv, "pfi:I:o:O:lL")) {
            case 'p': { out = OUT_IMG; break; }
            case 'f': { out = OUT_FIN; break; }
            case 'i': { max_iter = strtol(optarg, NULL, 10); break; }
            case 'I': { out_interval = strtol(optarg, NULL, 10); break; }
            case 'o': { out_path = optarg; break; }
            case 'O': { out_pref = optarg; break; }
            case 'l': { print_lups = true; break; }
            case 'L': { print_avg_lups = true; break; }
            default : { goto usage; }
        }
    }

    // check that execution mode is set (output images or fin values)
    if (max_iter < 1) {
    usage:
        fprintf(stderr, "usage: %s (-p | -f) -i <iter> [-I <out_interval>] [-o <out_dir>] [-O <out_prefix>] [-l] [-L]\n", basename((char*)argv[0]));
        fprintf(stderr, "  -p : output pictures\n");
        fprintf(stderr, "  -f : output populations\n");
        fprintf(stderr, "  -i : number of iterations\n");
        fprintf(stderr, "  -I : output interval; (0 if only the last iteration output in required)\n");
        fprintf(stderr, "  -o : output file directory\n");
        fprintf(stderr, "  -O : output filename prefix\n");
        fprintf(stderr, "  -l : print lups at each output interval\n");
        fprintf(stderr, "  -L : print average lups at the end\n");
        return EXIT_FAILURE;
    }

    if (out == OUT_NONE) {
        fprintf(stderr, "No output mode specified.\n");
    }

    lbm_consts* h_consts = (lbm_consts*)malloc(sizeof(lbm_consts));
    
    initCol(h_consts->col[0],  1);
    initCol(h_consts->col[1],  0);
    initCol(h_consts->col[2], -1);
    initOpp(h_consts->opp);
  	initVelocity(h_consts->vel);
    memcpy(h_consts->v, V, sizeof(V));
    memcpy(h_consts->t, T, sizeof(T));
    
    HANDLE_ERROR(cudaMemcpyToSymbol(d_consts, h_consts, sizeof(lbm_consts)));
        
    lbm_vars *h_vars = (lbm_vars*)malloc(sizeof(lbm_vars));
    initObstacles(h_vars);
    
    // Initialization of the populations at equilibrium with the given velocity.
    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            h_equilibrium(h_vars->fin[x][y], 1.0, h_consts->vel[y]);
        }
    }
    
    lbm_vars *d_vars;
    HANDLE_ERROR(cudaMalloc(&d_vars, sizeof(lbm_vars)));
    HANDLE_ERROR(cudaMemcpy(d_vars, h_vars, sizeof(lbm_vars), cudaMemcpyHostToDevice));

    dim3 dimBlock(NB_BLOCKS);
    dim3 dimGrid(getThreads(NX, NY));

    long time_diff, total_time_diff = 0;
    start_time_t start_time;
    timing_start(&start_time);

    for (int iter = 1; iter <= max_iter; iter++) {
        
        HANDLE_KERNEL_ERROR(lbm_computation<<<dimBlock, dimGrid>>>(d_vars));
        HANDLE_KERNEL_ERROR(lbm_streaming  <<<dimBlock, dimGrid>>>(d_vars));

        if ( (!out_interval && iter == max_iter) || (out_interval && iter % out_interval == 0) ) {

            total_time_diff += time_diff = timing_stop(&start_time);
            if ( print_lups ) {
                long iter_diff = out_interval? out_interval : max_iter;
                printf("lups: %.2f\n", get_lups(NX*NY, iter_diff, time_diff));
            }

            HANDLE_ERROR(cudaMemcpy(h_vars, d_vars, sizeof(lbm_vars), cudaMemcpyDeviceToHost));

            char* filename;

            if ( out == OUT_IMG ) {
                if ( asprintf(&filename, "%s/%s%d.pgm", out_path, out_pref, iter) != -1 ) {
                    output_image(filename, h_vars->u);
                    free(filename);
                }
            }

            if (out == OUT_FIN) {
                if ( asprintf(&filename, "%s/%s%d.out", out_path, out_pref, iter) != -1 ) {
                    output_variables(filename, h_vars->fin);
                    free(filename);
                }
            }
            timing_start(&start_time);
        }
    }

    if ( print_avg_lups ) {
        printf("average lups: %.2f\n", get_lups(NX*NY, max_iter, total_time_diff));
    }

    free(h_consts);
    free(h_vars);
    HANDLE_ERROR(cudaFree(d_vars));
    
    return EXIT_SUCCESS;
}
