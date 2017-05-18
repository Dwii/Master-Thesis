/*!
 * \file    lbmFlowAroundCylinder.cu
 * \brief   Cuda version based on lbm_sailfish_hist and lbm_opt1.
 * \author  Adrien Python
 * \date    28.01.2017
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <lbm.h>

#define RE       220.0       // Reynolds number
#define NX       420         // Number of lattice nodes (width)
#define NY       180         // Number of lattice nodes (height)
#define NL       ((NX)*(NY)) // Number of lattice nodes (total)
#define LY       ((NY) - 1)  // Height of the domain in lattice units
#define CX       ((NX) / 4)  // X coordinates of the cylinder
#define CY       ((NY) / 2)  // Y coordinates of the cylinder
#define R        ((NY) / 9)  // Cylinder radius
#define ULB      0.04        // Velocity in lattice units
#define NULB     ((ULB) * (R) / (RE))   // Viscoscity in lattice units
#define OMEGA    ((double)1. / (3*(NULB)+0.5))  // Relaxation parameter

#define BLOCK_SIZE 64

#define SQUARE(a) ((a)*(a))
#define GPU_SQUARE(a) __dmul_rn(a,a)
#define INDEX_2D_FROM_1D(x, y, i) do { (y) = (i)/(NX), (x) = (i)%(NX); } while (0)

#define IDX(x, y) ((x+NX)%(NX) + ( (y+NY)%(NY) )*(NX))

struct lbm_lattices{
    double ne[NL], e[NL], se[NL], n[NL], c[NL], s[NL], nw[NL], w[NL], sw[NL];
};

struct lbm_u {
    double u0[NL];
    double u1[NL];
};

typedef struct {
    bool obstacles[NL];  // Should reside in lbm_consts but is too big for constant memory
    double u0[NL];
    double u1[NL];
    lbm_lattices f0;
    lbm_lattices f1;
} lbm_vars;

typedef struct {
    double vel[NY];
} lbm_consts;

__constant__ lbm_consts d_consts;

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
/*    HANDLE_ERROR( cudaPeekAtLastError() );  */ \
/*    HANDLE_ERROR( cudaDeviceSynchronize() );*/ \
} while(0)

/**
 * Setup: cylindrical obstacle and velocity inlet with perturbation
 * Creation of a mask with boolean values, defining the shape of the obstacle.
 */
static void initObstacles(bool* obstacles)
{
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            obstacles[IDX(x,y)] = SQUARE(x-CX) + SQUARE(y-CY) < SQUARE(R);
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

__host__ static void h_equilibrium(lbm_lattices* f, int index, double rho, double u)
{
    double usqr = 3./2 * ( SQUARE(u) );
    double cu  = 3 * u;
    f->ne[index] = rho * 1./36 * ( 1 + cu + 0.5 * SQUARE(cu) - usqr );
    f->e [index] = rho * 1./9  * ( 1 + cu + 0.5 * SQUARE(cu) - usqr );
    f->se[index] = rho * 1./36 * ( 1 + cu + 0.5 * SQUARE(cu) - usqr );
    f->n [index] = rho * 1./9  * ( 1 - usqr );
    f->c [index] = rho * 4./9  * ( 1 - usqr );
    f->s [index] = rho * 1./9  * ( 1 - usqr );
    f->nw[index] = rho * 1./36 * ( 1 - cu + 0.5 * SQUARE(cu) - usqr );
    f->w [index] = rho * 1./9  * ( 1 - cu + 0.5 * SQUARE(cu) - usqr );
    f->sw[index] = rho * 1./36 * ( 1 - cu + 0.5 * SQUARE(cu) - usqr );
}

__device__ static void d_equilibrium(double* ne, double* e, double* se, double* n, double* c, double* s, double* nw, double* w, double* sw, double rho, double u0, double u1)
{
    double usqr = __dmul_rn(3./2, __dadd_rn( GPU_SQUARE(u0), GPU_SQUARE(u1) ));
   
    double cu_ne = 3 * ( u0 + u1 );
    double cu_se = 3 * ( u0 - u1 );

    double cu_ns  = 3 * u1;
    double cu_we  = 3 * u0;

    double cu_nw = 3 * ( -u0 + u1 );
    double cu_sw = 3 * ( -u0 - u1 );

    *ne = __dmul_rn(__dmul_rn(rho, 1./36), __dadd_rn(__dadd_rn(__dadd_rn(1, cu_ne) , __dmul_rn(0.5, GPU_SQUARE(cu_ne))), - usqr) );
    *e  = __dmul_rn(__dmul_rn(rho, 1./9 ), __dadd_rn(__dadd_rn(__dadd_rn(1, cu_we) , __dmul_rn(0.5, GPU_SQUARE(cu_we))), - usqr) );
    *se = __dmul_rn(__dmul_rn(rho, 1./36), __dadd_rn(__dadd_rn(__dadd_rn(1, cu_se) , __dmul_rn(0.5, GPU_SQUARE(cu_se))), - usqr) );
    *n  = __dmul_rn(__dmul_rn(rho, 1./9 ), __dadd_rn(__dadd_rn(__dadd_rn(1, cu_ns) , __dmul_rn(0.5, GPU_SQUARE(cu_ns))), - usqr) );
    *c  = __dmul_rn(__dmul_rn(rho, 4./9 ), __dadd_rn(1, - usqr) );
    *s  = __dmul_rn(__dmul_rn(rho, 1./9 ), __dadd_rn(__dadd_rn(__dadd_rn(1,-cu_ns ), __dmul_rn(0.5, GPU_SQUARE(cu_ns))), - usqr) );
    *nw = __dmul_rn(__dmul_rn(rho, 1./36), __dadd_rn(__dadd_rn(__dadd_rn(1, cu_nw) , __dmul_rn(0.5, GPU_SQUARE(cu_nw))), - usqr) );
    *w  = __dmul_rn(__dmul_rn(rho, 1./9 ), __dadd_rn(__dadd_rn(__dadd_rn(1,-cu_we) , __dmul_rn(0.5, GPU_SQUARE(cu_we))), - usqr) );
    *sw = __dmul_rn(__dmul_rn(rho, 1./36), __dadd_rn(__dadd_rn(__dadd_rn(1, cu_sw) , __dmul_rn(0.5, GPU_SQUARE(cu_sw))), - usqr) );
}

__device__ static void macroscopic(double ne, double e, double se, double n, double c, double s, double nw, double w, double sw, double* rho, double* u0, double* u1)
{   
    *rho = ne + e  + se + n  + c  + s + nw + w + sw;
    *u0 = (ne + e  + se - nw - w  - sw) / *rho;
    *u1 = (ne - se + n  - s  + nw - sw) / *rho;
}

__global__ void lbm_right_wall(lbm_lattices* f)
{
    for (int y = blockIdx.y; y < NY; y+=gridDim.y) {
       // Right wall: outflow condition.
       f->nw[IDX(NX-1,y)] = f->nw[IDX(NX-2,y)];
       f->w [IDX(NX-1,y)] = f->w [IDX(NX-2,y)];
       f->sw[IDX(NX-1,y)] = f->sw[IDX(NX-2,y)];
   }
}

__global__ void lbm_computation(lbm_vars *d_vars, lbm_lattices* f0, lbm_lattices* f1)
{

    int tix = threadIdx.x;
    for (int y = blockIdx.y; y < NY; y+=gridDim.y) {
        for (int x = threadIdx.x + blockIdx.x * blockDim.x; x < NX; x += blockDim.x * gridDim.x) {
            size_t gi = IDX(x,y);

            double fin_ne, fin_e, fin_se, fin_n, fin_c, fin_s, fin_nw, fin_w, fin_sw;
            double fout_ne, fout_e, fout_se, fout_n, fout_c, fout_s, fout_nw, fout_w, fout_sw;

            fin_ne = f0->ne[gi];
            fin_e  = f0->e [gi];
            fin_se = f0->se[gi];
            fin_n  = f0->n [gi];
            fin_c  = f0->c [gi];
            fin_s  = f0->s [gi];
            fin_nw = f0->nw[gi];
            fin_w  = f0->w [gi];
            fin_sw = f0->sw[gi];

            // Compute macroscopic variables, density and velocity
            double rho, u0, u1;
            macroscopic(fin_ne, fin_e, fin_se, fin_n, fin_c, fin_s, fin_nw, fin_w, fin_sw, &rho, &u0, &u1);
            
            if (x == 0) {
                // Left wall: inflow condition
                u0 = d_consts.vel[y];
                u1 = 0;

                // Calculate the density
                double s2 = fin_n  + fin_c + fin_s;
                double s3 = fin_nw + fin_w + fin_sw;;
                rho = 1./(1 - u0) * (s2 + 2*s3);
            }
            
            // Compute equilibrium
            double feq_ne, feq_e, feq_se, feq_n, feq_c, feq_s, feq_nw, feq_w, feq_sw;
            d_equilibrium(&feq_ne, &feq_e, &feq_se, &feq_n, &feq_c, &feq_s, &feq_nw, &feq_w, &feq_sw, rho, u0, u1);
 
            if (x == 0) {
                fin_ne = feq_ne + fin_sw - feq_sw;
                fin_e  = feq_e  + fin_w  - feq_w ;
                fin_se = feq_se + fin_nw - feq_nw;
            }

            if (d_vars->obstacles[IDX(x, y)]) {
                // Bounce-back condition for obstacle
                fout_ne = fin_sw; 
                fout_e  = fin_w ; 
                fout_se = fin_nw; 
                fout_n  = fin_s ; 
                fout_c  = fin_c ; 
                fout_s  = fin_n ; 
                fout_nw = fin_se; 
                fout_w  = fin_e ; 
                fout_sw = fin_ne; 
            } else {
                // Collision step
                fout_ne = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_ne, - feq_ne)), fin_ne);
                fout_e  = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_e , - feq_e )), fin_e );
                fout_se = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_se, - feq_se)), fin_se);
                fout_n  = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_n , - feq_n )), fin_n );
                fout_c  = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_c , - feq_c )), fin_c );
                fout_s  = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_s , - feq_s )), fin_s );
                fout_nw = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_nw, - feq_nw)), fin_nw);
                fout_w  = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_w , - feq_w )), fin_w );
                fout_sw = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_sw, - feq_sw)), fin_sw);
            }

    		d_vars->u0[gi] = u0;
    		d_vars->u1[gi] = u1;

            // STREAMING

            // shared variables for in-block propagation
            __shared__ double fo_E [BLOCK_SIZE];
            __shared__ double fo_W [BLOCK_SIZE];
            __shared__ double fo_SE[BLOCK_SIZE];
            __shared__ double fo_SW[BLOCK_SIZE];
            __shared__ double fo_NE[BLOCK_SIZE];
            __shared__ double fo_NW[BLOCK_SIZE];

            // Center 'propagation' (global memory)
            f1->c[gi] = fout_c;

            // N + S propagation (global memory)
            f1->s[IDX(x, y-1)] = fout_s;
            f1->n[IDX(x, y+1)] = fout_n;

            // E propagation in shared memory
            if (tix < blockDim.x-1 && x < NX-1) {
                fo_E [tix+1] = fout_e;
                fo_NE[tix+1] = fout_ne;
                fo_SE[tix+1] = fout_se;
            // E propagation in global memory (at block boundary)
            } else {
                f1->e [IDX(x+1, y  )] = fout_e;
                f1->se[IDX(x+1, y-1)] = fout_se;
                f1->ne[IDX(x+1, y+1)] = fout_ne;
            }

            // W propagation in shared memory
            if (tix > 0) {
                fo_W [tix-1] = fout_w;
                fo_NW[tix-1] = fout_nw;
                fo_SW[tix-1] = fout_sw;
            // W propagation in global memory (at block boundary)
            } else {
                f1->w [IDX(x-1, y  )] = fout_w;
                f1->sw[IDX(x-1, y-1)] = fout_sw;
                f1->nw[IDX(x-1, y+1)] = fout_nw;
            }

            __syncthreads();

            // the leftmost thread is not updated in this block
            if (tix > 0) {
                f1->e [gi         ] = fo_E [tix];
                f1->se[IDX(x, y-1)] = fo_SE[tix];
                f1->ne[IDX(x, y+1)] = fo_NE[tix];
            }

            // the rightmost thread is not updated in this block
            if (tix < blockDim.x-1 && x < NX-1) {
                f1->w [gi         ] = fo_W [tix];
                f1->sw[IDX(x, y-1)] = fo_SW[tix];
                f1->nw[IDX(x, y+1)] = fo_NW[tix];
            }

            __syncthreads(); // only nessessary when NX % BLOCK_SIZE != 0 
        }
    }
}

struct lbm_simulation{
    lbm_vars h_vars, *d_vars;
    dim3 dimComputationGrid, dimComputationBlock;
    dim3 dimRightWallGrid, dimRightWallBlock;
    size_t shared_mem_size;
    bool switch_f0_f1;
};


lbm_simulation* lbm_simulation_create()
{
    lbm_simulation* lbm_sim = (lbm_simulation*) malloc (sizeof(lbm_simulation));
    
    lbm_consts h_consts;
    
    initVelocity(h_consts.vel);
    
    HANDLE_ERROR(cudaMemcpyToSymbol(d_consts, &h_consts, sizeof(lbm_consts)));
        
    initObstacles(lbm_sim->h_vars.obstacles);
    
    // Initialization of the populations at equilibrium with the given velocity.
    lbm_sim->switch_f0_f1 = false;
    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            h_equilibrium(&lbm_sim->h_vars.f0, IDX(x,y), 1.0, h_consts.vel[y]);
        }
    }

    HANDLE_ERROR(cudaMalloc(&lbm_sim->d_vars, sizeof(lbm_vars)));
    HANDLE_ERROR(cudaMemcpy(lbm_sim->d_vars, &lbm_sim->h_vars, sizeof(lbm_vars), cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    dim3 dimComputationGrid(max(1, NX/BLOCK_SIZE), min(NY, prop.maxGridSize[1]));
    dim3 dimComputationBlock(BLOCK_SIZE);
    lbm_sim->dimComputationGrid = dimComputationGrid;
    lbm_sim->dimComputationBlock = dimComputationBlock;

    dim3 dimRightWallGrid(1, min(NY, prop.maxGridSize[1]));
    dim3 dimRightWallBlock(1);
    lbm_sim->dimRightWallGrid = dimRightWallGrid;
    lbm_sim->dimRightWallBlock = dimRightWallBlock;

    lbm_sim->shared_mem_size = 6 * sizeof(double) * BLOCK_SIZE;

    return lbm_sim;
}

void lbm_simulation_destroy(lbm_simulation* lbm_sim)
{
    HANDLE_ERROR(cudaFree(lbm_sim->d_vars));
    free(lbm_sim);
}

void lbm_simulation_update(lbm_simulation* lbm_sim)
{
    if (lbm_sim->switch_f0_f1) {
        HANDLE_KERNEL_ERROR(lbm_right_wall<<<lbm_sim->dimRightWallGrid, lbm_sim->dimRightWallBlock>>>(&lbm_sim->d_vars->f1));
        HANDLE_KERNEL_ERROR(lbm_computation<<<lbm_sim->dimComputationGrid, lbm_sim->dimComputationBlock, lbm_sim->shared_mem_size>>>(lbm_sim->d_vars, &lbm_sim->d_vars->f1, &lbm_sim->d_vars->f0));
    } else {
        HANDLE_KERNEL_ERROR(lbm_right_wall<<<lbm_sim->dimRightWallGrid, lbm_sim->dimRightWallBlock>>>(&lbm_sim->d_vars->f0));
        HANDLE_KERNEL_ERROR(lbm_computation<<<lbm_sim->dimComputationGrid, lbm_sim->dimComputationBlock, lbm_sim->shared_mem_size>>>(lbm_sim->d_vars, &lbm_sim->d_vars->f0, &lbm_sim->d_vars->f1));
    }

    lbm_sim->switch_f0_f1 = ! lbm_sim->switch_f0_f1;
}

void lbm_simulation_get_size(lbm_simulation* lbm_sim, size_t* width, size_t* height)
{
    *width = NX;
    *height = NY;
}

void lbm_lattices_read(lbm_simulation* lbm_sim, lbm_lattices* lat)
{
    lbm_lattices* d_lat = lbm_sim->switch_f0_f1 ? &lbm_sim->d_vars->f1 : &lbm_sim->d_vars->f0;
    HANDLE_ERROR(cudaMemcpy(lat, d_lat, sizeof(lbm_lattices), cudaMemcpyDeviceToHost));
}

void lbm_u_read(lbm_simulation* lbm_sim, lbm_u* u)
{
    HANDLE_ERROR(cudaMemcpy(u->u0, lbm_sim->d_vars->u0, sizeof(double)*NL, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(u->u1, lbm_sim->d_vars->u1, sizeof(double)*NL, cudaMemcpyDeviceToHost));
}

lbm_lattices* lbm_lattices_create()
{
    return (lbm_lattices*) malloc(sizeof(lbm_lattices));
}

void lbm_lattices_destroy(lbm_lattices* lat)
{
    free(lat);
}

lbm_u* lbm_u_create()
{
    return (lbm_u*) malloc(sizeof(lbm_u));
}

void lbm_u_destroy(lbm_u* u)
{
    free(u);
}

void lbm_lattices_at_index(lbm_lattice* lattice, lbm_lattices* lattices, int x, int y)
{
    int gi = IDX(x,y);
    lbm_lattices* lat = (lbm_lattices*) lattices;
    lattice->ne = lat->ne[gi];
    lattice->e  = lat->e [gi];
    lattice->se = lat->se[gi];
    lattice->n  = lat->n [gi];
    lattice->c  = lat->c [gi];
    lattice->s  = lat->s [gi];
    lattice->nw = lat->nw[gi];
    lattice->w  = lat->w [gi];
    lattice->sw = lat->sw[gi];
}

void lbm_u_at_index(double* u0, double* u1, lbm_u* u, int x, int y)
{
    int gi = IDX(x,y);
    *u0 = u->u0[gi];
    *u1 = u->u1[gi];
}
