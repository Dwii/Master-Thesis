/*!
 * \file    lbmcuda.c
 * \brief   LBM cuda library
 * \author  Adrien Python
 * \version 1.0
 * \date    14.06.2017
 */

#include "lbmcuda.h"

#include <stdio.h>

#define BLOCK_SIZE 64

#define SQUARE(a) ((a)*(a))
#define GPU_SQUARE(a) __dmul_rn(a,a)

#define IDX(x, y, z, nx, ny, nz) ((x+(nx))%(nx) + ((y+(ny))%(ny) + ( (z+(nz))%(nz) )*(ny))*(nx) )

typedef struct {
    bool* obstacles;
    lbm_lattices f0;
    lbm_lattices f1;
} lbm_vars;

struct lbm_simulation {
    lbm_vars d_vars;
    dim3 dimComputationGrid, dimComputationBlock;
    dim3 dimRightWallGrid, dimRightWallBlock;
    size_t shared_mem_size;
    bool switch_f0_f1;
    size_t nx, ny, nz;
    double omega;
};

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


#define EQUILIBRIUM(rho, t, cu, usqr) __dmul_rn(__dmul_rn(rho, (t)), __dadd_rn(__dadd_rn(__dadd_rn(1, cu) , __dmul_rn(0.5, GPU_SQUARE(cu))), - usqr) )

__device__ static void equilibrium(double* ne, double* e, double* se, double* n, double* c, double* s, double* nw, double* w, double* sw, 
                                     double* te, double* tn, double* tc, double* ts, double* tw,
                                     double* be, double* bn, double* bc, double* bs, double* bw,
                                     double rho, double u0, double u1, double u2)
{
    double usqr = __dmul_rn(3./2, __dadd_rn( __dadd_rn( GPU_SQUARE(u0), GPU_SQUARE(u1)), GPU_SQUARE(u2) ));

    { double cu = 3 * (  u0 +  u1 ); *ne = EQUILIBRIUM(rho, 1./36, cu, usqr ); }
    { double cu = 3 * (  u0       ); *e  = EQUILIBRIUM(rho, 1./18, cu, usqr ); }
    { double cu = 3 * (  u0 + -u1 ); *se = EQUILIBRIUM(rho, 1./36, cu, usqr ); }
    { double cu = 3 * (        u1 ); *n  = EQUILIBRIUM(rho, 1./18, cu, usqr ); }
    { double cu = 0                ; *c  = EQUILIBRIUM(rho, 1./3 , cu, usqr ); }
    { double cu = 3 * (       -u1 ); *s  = EQUILIBRIUM(rho, 1./18, cu, usqr ); }
    { double cu = 3 * ( -u0 +  u1 ); *nw = EQUILIBRIUM(rho, 1./36, cu, usqr ); }
    { double cu = 3 * ( -u0       ); *w  = EQUILIBRIUM(rho, 1./18, cu, usqr ); }
    { double cu = 3 * ( -u0 + -u1 ); *sw = EQUILIBRIUM(rho, 1./36, cu, usqr ); }
    { double cu = 3 * (  u0 +  u2 ); *te = EQUILIBRIUM(rho, 1./36, cu, usqr ); }
    { double cu = 3 * (  u1 +  u2 ); *tn = EQUILIBRIUM(rho, 1./36, cu, usqr ); }
    { double cu = 3 * (        u2 ); *tc = EQUILIBRIUM(rho, 1./18, cu, usqr ); }
    { double cu = 3 * ( -u1 +  u2 ); *ts = EQUILIBRIUM(rho, 1./36, cu, usqr ); }
    { double cu = 3 * ( -u0 +  u2 ); *tw = EQUILIBRIUM(rho, 1./36, cu, usqr ); }
    { double cu = 3 * (  u0 + -u2 ); *be = EQUILIBRIUM(rho, 1./36, cu, usqr ); }
    { double cu = 3 * (  u1 + -u2 ); *bn = EQUILIBRIUM(rho, 1./36, cu, usqr ); }
    { double cu = 3 * (       -u2 ); *bc = EQUILIBRIUM(rho, 1./18, cu, usqr ); }
    { double cu = 3 * ( -u1 + -u2 ); *bs = EQUILIBRIUM(rho, 1./36, cu, usqr ); }
    { double cu = 3 * ( -u0 + -u2 ); *bw = EQUILIBRIUM(rho, 1./36, cu, usqr ); }
}

__device__ static void macroscopic(double ne, double e, double se, double n, double c, double s, double nw, double w, double sw,
                                   double te, double tn, double tc, double ts, double tw, 
                                   double be, double bn, double bc, double bs, double bw, 
                                   double* rho, double* u0, double* u1, double* u2)
{   
    *rho = ne + e  + se + n  + c  + s  + nw + w  + sw + te + tn + tc + ts + tw + be + bn + bc + bs + bw;
    *u0 = (ne + e  + se - nw - w  - sw + te - tw + be - bw) / *rho;
    *u1 = (ne - se + n  - s  + nw - sw + tn - ts + bn - bs) / *rho;
    *u2 = (te + tn + tc + ts + tw - be - bn - bc - bs - bw) / *rho;
}

__global__ void lbm_computation(lbm_vars d_vars, lbm_lattices f0, lbm_lattices f1, size_t nx, size_t ny, size_t nz, double omega)
{
    int tix = threadIdx.x;
    for (int z = blockIdx.z; z < nz; z+=gridDim.z) {
        for (int y = blockIdx.y; y < ny; y+=gridDim.y) {
            for (int x = threadIdx.x + blockIdx.x * blockDim.x; x < nx; x += blockDim.x * gridDim.x) {
                size_t gi = IDX(x,y,z,nx,ny,nz);

                double fin_ne, fin_e, fin_se, fin_n, fin_c, fin_s, fin_nw, fin_w, fin_sw,
                       fin_te, fin_tn, fin_tc, fin_ts, fin_tw, 
                       fin_be, fin_bn, fin_bc, fin_bs, fin_bw;
                double fout_ne, fout_e, fout_se, fout_n, fout_c, fout_s, fout_nw, fout_w, fout_sw,
                       fout_te, fout_tn, fout_tc, fout_ts, fout_tw, 
                       fout_be, fout_bn, fout_bc, fout_bs, fout_bw;

                fin_ne = f0.ne[gi];
                fin_e  = f0.e [gi];
                fin_se = f0.se[gi];
                fin_n  = f0.n [gi];
                fin_c  = f0.c [gi];
                fin_s  = f0.s [gi];
                fin_nw = f0.nw[gi];
                fin_w  = f0.w [gi];
                fin_sw = f0.sw[gi];
                fin_te = f0.te[gi];
                fin_tn = f0.tn[gi];
                fin_tc = f0.tc[gi];
                fin_ts = f0.ts[gi];
                fin_tw = f0.tw[gi];
                fin_be = f0.be[gi];
                fin_bn = f0.bn[gi];
                fin_bc = f0.bc[gi];
                fin_bs = f0.bs[gi];
                fin_bw = f0.bw[gi];

                // Compute macroscopic variables, density and velocity
                double rho, u0, u1, u2;
                macroscopic(fin_ne, fin_e, fin_se, fin_n, fin_c, fin_s, fin_nw, fin_w, fin_sw, 
                            fin_te, fin_tn, fin_tc, fin_ts, fin_tw, 
                            fin_be, fin_bn, fin_bc, fin_bs, fin_bw,
                            &rho, &u0, &u1, &u2);

                // Compute equilibrium
                double feq_ne, feq_e, feq_se, feq_n, feq_c, feq_s, feq_nw, feq_w, feq_sw, 
                       feq_te, feq_tn, feq_tc, feq_ts, feq_tw, 
                       feq_be, feq_bn, feq_bc, feq_bs, feq_bw;
                equilibrium(&feq_ne, &feq_e, &feq_se, &feq_n, &feq_c, &feq_s, &feq_nw, &feq_w, &feq_sw, 
                            &feq_te, &feq_tn, &feq_tc, &feq_ts, &feq_tw, 
                            &feq_be, &feq_bn, &feq_bc, &feq_bs, &feq_bw, 
                            rho, u0, u1, u2);       

                if (d_vars.obstacles[IDX(x,y,z,nx,ny,nz)]) {
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

                    fout_te = fin_bw;
                    fout_tn = fin_bs;
                    fout_tc = fin_bc;
                    fout_ts = fin_bn;
                    fout_tw = fin_be;
                    fout_be = fin_tw;
                    fout_bn = fin_ts;
                    fout_bc = fin_tc;
                    fout_bs = fin_tn;
                    fout_bw = fin_tw;

                } else {
                    // Collision step
                    fout_ne = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_ne, - feq_ne)), fin_ne);
                    fout_e  = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_e , - feq_e )), fin_e );
                    fout_se = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_se, - feq_se)), fin_se);
                    fout_n  = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_n , - feq_n )), fin_n );
                    fout_c  = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_c , - feq_c )), fin_c );
                    fout_s  = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_s , - feq_s )), fin_s );
                    fout_nw = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_nw, - feq_nw)), fin_nw);
                    fout_w  = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_w , - feq_w )), fin_w );
                    fout_sw = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_sw, - feq_sw)), fin_sw);

                    fout_te = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_te, - feq_te)), fin_te);
                    fout_tn = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_tn, - feq_tn)), fin_tn);
                    fout_tc = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_tc, - feq_tc)), fin_tc);
                    fout_ts = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_ts, - feq_ts)), fin_ts);
                    fout_tw = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_tw, - feq_tw)), fin_tw);
                    fout_be = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_be, - feq_be)), fin_be);
                    fout_bn = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_bn, - feq_bn)), fin_bn);
                    fout_bc = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_bc, - feq_bc)), fin_bc);
                    fout_bs = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_bs, - feq_bs)), fin_bs);
                    fout_bw = __dadd_rn(__dmul_rn(-omega, __dadd_rn(fin_bw, - feq_bw)), fin_bw);
                }

                // STREAMING

                // shared variables for in-block propagation
                __shared__ double fo_E [BLOCK_SIZE];
                __shared__ double fo_W [BLOCK_SIZE];
                __shared__ double fo_SE[BLOCK_SIZE];
                __shared__ double fo_SW[BLOCK_SIZE];
                __shared__ double fo_NE[BLOCK_SIZE];
                __shared__ double fo_NW[BLOCK_SIZE];

                // Center 'propagation' (global memory)
                f1.c[gi] = fout_c;

                // N + S propagation (global memory)
                f1.s[IDX(x, y-1, z, nx,ny,nz)] = fout_s;
                f1.n[IDX(x, y+1, z, nx,ny,nz)] = fout_n;

                // E propagation in shared memory
                if (tix < blockDim.x-1 && x < nx-1) {
                    fo_E [tix+1] = fout_e;
                    fo_NE[tix+1] = fout_ne;
                    fo_SE[tix+1] = fout_se;
                // E propagation in global memory (at block boundary)
                } else {
                    f1.e [IDX(x+1, y  , z, nx,ny,nz)] = fout_e;
                    f1.se[IDX(x+1, y-1, z, nx,ny,nz)] = fout_se;
                    f1.ne[IDX(x+1, y+1, z, nx,ny,nz)] = fout_ne;
                }

                // W propagation in shared memory
                if (tix > 0) {
                    fo_W [tix-1] = fout_w;
                    fo_NW[tix-1] = fout_nw;
                    fo_SW[tix-1] = fout_sw;
                // W propagation in global memory (at block boundary)
                } else {
                    f1.w [IDX(x-1, y  , z, nx,ny,nz)] = fout_w;
                    f1.sw[IDX(x-1, y-1, z, nx,ny,nz)] = fout_sw;
                    f1.nw[IDX(x-1, y+1, z, nx,ny,nz)] = fout_nw;
                }

                // Top and Bottom propagation (global memory)
                f1.te[IDX(x+1, y  , z+1, nx,ny,nz)] = fout_te;
                f1.tn[IDX(x  , y+1, z+1, nx,ny,nz)] = fout_tn;
                f1.tc[IDX(x  , y  , z+1, nx,ny,nz)] = fout_tc;
                f1.ts[IDX(x  , y-1, z+1, nx,ny,nz)] = fout_ts;
                f1.tw[IDX(x-1, y  , z+1, nx,ny,nz)] = fout_tw;
                f1.be[IDX(x+1, y  , z-1, nx,ny,nz)] = fout_be;
                f1.bn[IDX(x  , y+1, z-1, nx,ny,nz)] = fout_bn;
                f1.bc[IDX(x  , y  , z-1, nx,ny,nz)] = fout_bc;
                f1.bs[IDX(x  , y-1, z-1, nx,ny,nz)] = fout_bs;
                f1.bw[IDX(x-1, y  , z-1, nx,ny,nz)] = fout_bw;

                __syncthreads();

                // the leftmost thread is not updated in this block
                if (tix > 0) {
                    f1.e [gi                      ] = fo_E [tix];
                    f1.se[IDX(x, y-1, z, nx,ny,nz)] = fo_SE[tix];
                    f1.ne[IDX(x, y+1, z, nx,ny,nz)] = fo_NE[tix];
                }

                // the rightmost thread is not updated in this block
                if (tix < blockDim.x-1 && x < nx-1) {
                    f1.w [gi                      ] = fo_W [tix];
                    f1.sw[IDX(x, y-1, z, nx,ny,nz)] = fo_SW[tix];
                    f1.nw[IDX(x, y+1, z, nx,ny,nz)] = fo_NW[tix];
                }

                __syncthreads(); // only nessessary when NX % BLOCK_SIZE != 0 
           }
        }
    }
}

void lbm_lattices_alloc(lbm_lattices* lat, size_t nl) 
{
    lat->ne = (double*) malloc ( sizeof(double)*nl );
    lat->e  = (double*) malloc ( sizeof(double)*nl );
    lat->se = (double*) malloc ( sizeof(double)*nl );
    lat->n  = (double*) malloc ( sizeof(double)*nl );
    lat->c  = (double*) malloc ( sizeof(double)*nl );
    lat->s  = (double*) malloc ( sizeof(double)*nl );
    lat->nw = (double*) malloc ( sizeof(double)*nl );
    lat->w  = (double*) malloc ( sizeof(double)*nl );
    lat->sw = (double*) malloc ( sizeof(double)*nl );
    lat->te = (double*) malloc ( sizeof(double)*nl );
    lat->tn = (double*) malloc ( sizeof(double)*nl );
    lat->tc = (double*) malloc ( sizeof(double)*nl );
    lat->ts = (double*) malloc ( sizeof(double)*nl );
    lat->tw = (double*) malloc ( sizeof(double)*nl );
    lat->be = (double*) malloc ( sizeof(double)*nl );
    lat->bn = (double*) malloc ( sizeof(double)*nl );
    lat->bc = (double*) malloc ( sizeof(double)*nl );
    lat->bs = (double*) malloc ( sizeof(double)*nl );
    lat->bw = (double*) malloc ( sizeof(double)*nl );
}

void lbm_lattices_cuda_alloc(lbm_lattices* lat, size_t nl) 
{
    HANDLE_ERROR(cudaMalloc(&lat->ne, sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->e , sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->se, sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->n , sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->c , sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->s , sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->nw, sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->w , sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->sw, sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->te, sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->tn, sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->tc, sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->ts, sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->tw, sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->be, sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->bn, sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->bc, sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->bs, sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&lat->bw, sizeof(double)*nl ));
}

void lbm_lattices_dealloc(lbm_lattices* lat) 
{
    free(lat->ne);
    free(lat->e );
    free(lat->se);
    free(lat->n );
    free(lat->c );
    free(lat->s );
    free(lat->nw);
    free(lat->w );
    free(lat->sw);
    free(lat->te);
    free(lat->tn);
    free(lat->tc);
    free(lat->ts);
    free(lat->tw);
    free(lat->be);
    free(lat->bn);
    free(lat->bc);
    free(lat->bs);
    free(lat->bw);
}

void lbm_lattices_cuda_dealloc(lbm_lattices* lat) 
{
    HANDLE_ERROR(cudaFree(lat->ne));
    HANDLE_ERROR(cudaFree(lat->e ));
    HANDLE_ERROR(cudaFree(lat->se));
    HANDLE_ERROR(cudaFree(lat->n ));
    HANDLE_ERROR(cudaFree(lat->c ));
    HANDLE_ERROR(cudaFree(lat->s ));
    HANDLE_ERROR(cudaFree(lat->nw));
    HANDLE_ERROR(cudaFree(lat->w ));
    HANDLE_ERROR(cudaFree(lat->sw));
    HANDLE_ERROR(cudaFree(lat->te));
    HANDLE_ERROR(cudaFree(lat->tn));
    HANDLE_ERROR(cudaFree(lat->tc));
    HANDLE_ERROR(cudaFree(lat->ts));
    HANDLE_ERROR(cudaFree(lat->tw));
    HANDLE_ERROR(cudaFree(lat->be));
    HANDLE_ERROR(cudaFree(lat->bn));
    HANDLE_ERROR(cudaFree(lat->bc));
    HANDLE_ERROR(cudaFree(lat->bs));
    HANDLE_ERROR(cudaFree(lat->bw));
}

void lbm_vars_alloc(lbm_vars* vars, size_t nl)
{
    vars->obstacles = (bool*) malloc( sizeof(bool)*nl);
    lbm_lattices_alloc(&vars->f0, nl);
    lbm_lattices_alloc(&vars->f1, nl);
}

void lbm_vars_cuda_alloc(lbm_vars* vars, size_t nl)
{
    HANDLE_ERROR(cudaMalloc(&vars->obstacles, sizeof(bool)*nl));
    lbm_lattices_cuda_alloc(&vars->f0, nl);
    lbm_lattices_cuda_alloc(&vars->f1, nl);
}

void lbm_vars_dealloc(lbm_vars* vars)
{
    free(vars->obstacles);
    lbm_lattices_dealloc(&vars->f0);
    lbm_lattices_dealloc(&vars->f1);
}

void lbm_vars_cuda_dealloc(lbm_vars* vars)
{
    HANDLE_ERROR(cudaFree(vars->obstacles));
    lbm_lattices_cuda_dealloc(&vars->f0);
    lbm_lattices_cuda_dealloc(&vars->f1);
}

lbm_simulation* lbm_simulation_create(size_t nx, size_t ny, size_t nz, double omega)
{
    lbm_simulation* lbm_sim = (lbm_simulation*) malloc (sizeof(lbm_simulation));
    size_t nl = nx*ny*nz;
    lbm_sim->nx = nx;
    lbm_sim->ny = ny;
    lbm_sim->nz = nz;
    lbm_sim->omega = omega;
           
    // Initialization of the populations at equilibrium with the given velocity.
    lbm_sim->switch_f0_f1 = false;

    lbm_vars_cuda_alloc(&lbm_sim->d_vars, nl);

//    HANDLE_ERROR(cudaMemcpy(lbm_sim->d_vars.obstacles, lbm_sim->h_vars.obstacles, sizeof(bool)*nl, cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    dim3 dimComputationGrid(max((unsigned long)1, (unsigned long)nx/BLOCK_SIZE), min((unsigned long)ny, (unsigned long)prop.maxGridSize[1]), min((unsigned long)nz, (unsigned long)prop.maxGridSize[2]));
    dim3 dimComputationBlock(BLOCK_SIZE);
    lbm_sim->dimComputationGrid = dimComputationGrid;
    lbm_sim->dimComputationBlock = dimComputationBlock;

    dim3 dimRightWallGrid(1, min((unsigned long)ny, (unsigned long)prop.maxGridSize[1]));
    dim3 dimRightWallBlock(1);
    lbm_sim->dimRightWallGrid = dimRightWallGrid;
    lbm_sim->dimRightWallBlock = dimRightWallBlock;

    lbm_sim->shared_mem_size = 6 * sizeof(double) * BLOCK_SIZE;

    return lbm_sim;
}

void lbm_simulation_destroy(lbm_simulation* lbm_sim)
{
    lbm_vars_cuda_dealloc(&lbm_sim->d_vars);
    free(lbm_sim);
}

void lbm_simulation_update(lbm_simulation* lbm_sim)
{
    if (lbm_sim->switch_f0_f1) {
        HANDLE_KERNEL_ERROR(lbm_computation<<<lbm_sim->dimComputationGrid, lbm_sim->dimComputationBlock, lbm_sim->shared_mem_size>>>(lbm_sim->d_vars, lbm_sim->d_vars.f1, lbm_sim->d_vars.f0, lbm_sim->nx, lbm_sim->ny, lbm_sim->nz, lbm_sim->omega));
    } else {
        HANDLE_KERNEL_ERROR(lbm_computation<<<lbm_sim->dimComputationGrid, lbm_sim->dimComputationBlock, lbm_sim->shared_mem_size>>>(lbm_sim->d_vars, lbm_sim->d_vars.f0, lbm_sim->d_vars.f1, lbm_sim->nx, lbm_sim->ny, lbm_sim->nz, lbm_sim->omega));
    }

    lbm_sim->switch_f0_f1 = ! lbm_sim->switch_f0_f1;
}

void lbm_lattices_write(lbm_simulation* lbm_sim, lbm_lattices* h_lat, size_t nl)
{
    lbm_lattices* d_lat = lbm_sim->switch_f0_f1 ? &lbm_sim->d_vars.f1 : &lbm_sim->d_vars.f0;

    HANDLE_ERROR(cudaMemcpy(d_lat->ne, h_lat->ne, sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->e , h_lat->e , sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->se, h_lat->se, sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->n , h_lat->n , sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->c , h_lat->c , sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->s , h_lat->s , sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->nw, h_lat->nw, sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->w , h_lat->w , sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->sw, h_lat->sw, sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->te, h_lat->te, sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->tn, h_lat->tn, sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->tc, h_lat->tc, sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->ts, h_lat->ts, sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->tw, h_lat->tw, sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->be, h_lat->be, sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->bn, h_lat->bn, sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->bc, h_lat->bc, sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->bs, h_lat->bs, sizeof(double)*nl, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lat->bw, h_lat->bw, sizeof(double)*nl, cudaMemcpyHostToDevice));
}

void lbm_lattices_read(lbm_simulation* lbm_sim, lbm_lattices* h_lat)
{
    size_t nl = lbm_sim->nx * lbm_sim->ny * lbm_sim->nz;

    lbm_lattices* d_lat = lbm_sim->switch_f0_f1 ? &lbm_sim->d_vars.f1 : &lbm_sim->d_vars.f0;

    HANDLE_ERROR(cudaMemcpy(h_lat->ne, d_lat->ne, sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->e , d_lat->e , sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->se, d_lat->se, sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->n , d_lat->n , sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->c , d_lat->c , sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->s , d_lat->s , sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->nw, d_lat->nw, sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->w , d_lat->w , sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->sw, d_lat->sw, sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->te, d_lat->te, sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->tn, d_lat->tn, sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->tc, d_lat->tc, sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->ts, d_lat->ts, sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->tw, d_lat->tw, sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->be, d_lat->be, sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->bn, d_lat->bn, sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->bc, d_lat->bc, sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->bs, d_lat->bs, sizeof(double)*nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_lat->bw, d_lat->bw, sizeof(double)*nl, cudaMemcpyDeviceToHost));
}

lbm_lattices* lbm_lattices_create(size_t nl)
{
    lbm_lattices* lat = (lbm_lattices*) malloc(sizeof(lbm_lattices));
    lbm_lattices_alloc(lat, nl);
    return lat;
}

void lbm_lattices_destroy(lbm_lattices* lat)
{
    lbm_lattices_dealloc(lat);
    free(lat);
}

