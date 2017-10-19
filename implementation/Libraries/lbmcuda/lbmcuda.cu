/*!
 * \file    lbmFlowAroundCylinder.cu
 * \brief   Cuda version based on lbm_sailfish_hist and lbm_opt1.
 * \author  Adrien Python
 * \date    28.01.2017
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "lbmcuda.h"

#define BLOCK_SIZE 64

#define SQUARE(a) ((a)*(a))
#define GPU_SQUARE(a) __dmul_rn(a,a)

typedef struct {
    lbm_u u;
    lbm_lattices f0;
    lbm_lattices f1;
} lbm_vars;

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


#define PBIDX(x, y, z, nx, ny, nz) ( ( x*(nz*ny) + y*(nz) + z) * 19)

__global__ void lbm_load_subdomain_to_gpu(lbm_lattices lat, double* pal_lat, lbm_box_3d subdomain, size_t snx, size_t sny, size_t snz, size_t nx, size_t ny, size_t nz)
{
    for (int z = subdomain.z0 + blockIdx.z; z <= subdomain.z1; z+=gridDim.z) {
        for (int y = subdomain.y0 + blockIdx.y; y <= subdomain.y1; y+=gridDim.y) {
            for (int x = subdomain.x0 + threadIdx.x + blockIdx.x * blockDim.x; x <= subdomain.x1; x += blockDim.x * gridDim.x) {
                size_t gi = IDX(x,y,z,nx,ny,nz);

                size_t px = x - subdomain.x0;
                size_t py = y - subdomain.y0;
                size_t pz = z - subdomain.z0;
                size_t pbi = PBIDX(px,py,pz,snx,sny,snz); // palabos data base index

                lat.c [gi] = pal_lat[pbi +  0] + 1./3 ;
                lat.w [gi] = pal_lat[pbi +  1] + 1./18;
                lat.s [gi] = pal_lat[pbi +  2] + 1./18;
                lat.bc[gi] = pal_lat[pbi +  3] + 1./18;
                lat.sw[gi] = pal_lat[pbi +  4] + 1./36;
                lat.nw[gi] = pal_lat[pbi +  5] + 1./36;
                lat.bw[gi] = pal_lat[pbi +  6] + 1./36;
                lat.tw[gi] = pal_lat[pbi +  7] + 1./36;
                lat.bs[gi] = pal_lat[pbi +  8] + 1./36;
                lat.ts[gi] = pal_lat[pbi +  9] + 1./36;
                lat.e [gi] = pal_lat[pbi + 10] + 1./18;
                lat.n [gi] = pal_lat[pbi + 11] + 1./18;
                lat.tc[gi] = pal_lat[pbi + 12] + 1./18;
                lat.ne[gi] = pal_lat[pbi + 13] + 1./36;
                lat.se[gi] = pal_lat[pbi + 14] + 1./36;
                lat.te[gi] = pal_lat[pbi + 15] + 1./36;
                lat.be[gi] = pal_lat[pbi + 16] + 1./36;
                lat.tn[gi] = pal_lat[pbi + 17] + 1./36;
                lat.bn[gi] = pal_lat[pbi + 18] + 1./36;
            }
        }
    }
}

__global__ void lbm_load_subdomain_from_gpu(lbm_lattices lat, double* pal_lat, lbm_box_3d subdomain, size_t snx, size_t sny, size_t snz, size_t nx, size_t ny, size_t nz)
{
     for (int z = subdomain.z0 + blockIdx.z; z <= subdomain.z1; z+=gridDim.z) {
        for (int y = subdomain.y0 + blockIdx.y; y <= subdomain.y1; y+=gridDim.y) {
            for (int x = subdomain.x0 + threadIdx.x + blockIdx.x * blockDim.x; x <= subdomain.x1; x += blockDim.x * gridDim.x) {
                size_t gi = IDX(x,y,z,nx,ny,nz);

                size_t px = x - subdomain.x0;
                size_t py = y - subdomain.y0;
                size_t pz = z - subdomain.z0;
                size_t pbi = PBIDX(px,py,pz,snx,sny,snz); // palabos data base index

                pal_lat[pbi +  0] = lat.c [gi] - 1./3 ;
                pal_lat[pbi +  1] = lat.w [gi] - 1./18;
                pal_lat[pbi +  2] = lat.s [gi] - 1./18;
                pal_lat[pbi +  3] = lat.bc[gi] - 1./18;
                pal_lat[pbi +  4] = lat.sw[gi] - 1./36;
                pal_lat[pbi +  5] = lat.nw[gi] - 1./36;
                pal_lat[pbi +  6] = lat.bw[gi] - 1./36;
                pal_lat[pbi +  7] = lat.tw[gi] - 1./36;
                pal_lat[pbi +  8] = lat.bs[gi] - 1./36;
                pal_lat[pbi +  9] = lat.ts[gi] - 1./36;
                pal_lat[pbi + 10] = lat.e [gi] - 1./18;
                pal_lat[pbi + 11] = lat.n [gi] - 1./18;
                pal_lat[pbi + 12] = lat.tc[gi] - 1./18;
                pal_lat[pbi + 13] = lat.ne[gi] - 1./36;
                pal_lat[pbi + 14] = lat.se[gi] - 1./36;
                pal_lat[pbi + 15] = lat.te[gi] - 1./36;
                pal_lat[pbi + 16] = lat.be[gi] - 1./36;
                pal_lat[pbi + 17] = lat.tn[gi] - 1./36;
                pal_lat[pbi + 18] = lat.bn[gi] - 1./36;
            }
        }
    }
}

__global__ void lbm_computation(lbm_vars d_vars, lbm_lattices f0, lbm_lattices f1, size_t nx, size_t ny, size_t nz, double omega)
{
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
                double feq_ne, feq_e, feq_se, feq_n, feq_c, feq_s, feq_nw, feq_w, feq_sw, 
                       feq_te, feq_tn, feq_tc, feq_ts, feq_tw, 
                       feq_be, feq_bn, feq_bc, feq_bs, feq_bw;

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
                equilibrium(&feq_ne, &feq_e, &feq_se, &feq_n, &feq_c, &feq_s, &feq_nw, &feq_w, &feq_sw, 
                            &feq_te, &feq_tn, &feq_tc, &feq_ts, &feq_tw, 
                            &feq_be, &feq_bn, &feq_bc, &feq_bs, &feq_bw, 
                            rho, u0, u1, u2);       

                d_vars.u.u0[gi] = u0;
                d_vars.u.u1[gi] = u1;
                d_vars.u.u2[gi] = u2;

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

                // Streaming
                f1.c [IDX(x  , y  , z  , nx,ny,nz)] = fout_c;
                f1.s [IDX(x  , y-1, z  , nx,ny,nz)] = fout_s;
                f1.n [IDX(x  , y+1, z  , nx,ny,nz)] = fout_n;
                f1.e [IDX(x+1, y  , z  , nx,ny,nz)] = fout_e;
                f1.se[IDX(x+1, y-1, z  , nx,ny,nz)] = fout_se;
                f1.ne[IDX(x+1, y+1, z  , nx,ny,nz)] = fout_ne;
                f1.w [IDX(x-1, y  , z  , nx,ny,nz)] = fout_w;
                f1.sw[IDX(x-1, y-1, z  , nx,ny,nz)] = fout_sw;
                f1.nw[IDX(x-1, y+1, z  , nx,ny,nz)] = fout_nw;
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
            }
        }
    }
}

struct lbm_simulation{
    lbm_vars h_vars, d_vars;
    double* d_pal_lat;
    dim3 dimComputationGrid, dimComputationBlock;
    size_t shared_mem_size;
    bool switch_f0_f1;
    size_t nx, ny, nz, nl;
    double omega;
    cudaDeviceProp prop;
};

void lbm_lattices_alloc(lbm_lattices* lat, size_t nl) {
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

void lbm_lattices_cuda_alloc(lbm_lattices* lat, size_t nl) {
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

void lbm_lattices_dealloc(lbm_lattices* lat) {
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

void lbm_lattices_cuda_dealloc(lbm_lattices* lat) {
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

void lbm_u_alloc(lbm_u* u, size_t nl) {
    u->u0 = (double*) malloc( sizeof(double)*nl );
    u->u1 = (double*) malloc( sizeof(double)*nl );
    u->u2 = (double*) malloc( sizeof(double)*nl );
}

void lbm_u_cuda_alloc(lbm_u* u, size_t nl) {
    HANDLE_ERROR(cudaMalloc(&u->u0, sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&u->u1, sizeof(double)*nl ));
    HANDLE_ERROR(cudaMalloc(&u->u2, sizeof(double)*nl ));
}

void lbm_u_dealloc(lbm_u* u) {
    free(u->u0);
    free(u->u1);
    free(u->u2);
}

void lbm_u_cuda_dealloc(lbm_u* u) {
    HANDLE_ERROR(cudaFree(u->u0));
    HANDLE_ERROR(cudaFree(u->u1));
    HANDLE_ERROR(cudaFree(u->u2));
}

void lbm_vars_alloc(lbm_vars* vars, size_t nl)
{
    lbm_u_alloc(&vars->u, nl);
    lbm_lattices_alloc(&vars->f0, nl);
    lbm_lattices_alloc(&vars->f1, nl);
}

void lbm_vars_cuda_alloc(lbm_vars* vars, size_t nl)
{
    lbm_u_cuda_alloc(&vars->u, nl);
    lbm_lattices_cuda_alloc(&vars->f0, nl);
    lbm_lattices_cuda_alloc(&vars->f1, nl);
}

void lbm_vars_dealloc(lbm_vars* vars)
{
    lbm_u_dealloc(&vars->u);
    lbm_lattices_dealloc(&vars->f0);
    lbm_lattices_dealloc(&vars->f1);
}

void lbm_vars_cuda_dealloc(lbm_vars* vars)
{
    lbm_u_cuda_dealloc(&vars->u);
    lbm_lattices_cuda_dealloc(&vars->f0);
    lbm_lattices_cuda_dealloc(&vars->f1);
}

lbm_simulation* lbm_simulation_create(size_t nx, size_t ny, size_t nz, double omega)
{
    lbm_simulation* lbm_sim = (lbm_simulation*) malloc (sizeof(lbm_simulation));

    lbm_sim->nx = nx;
    lbm_sim->ny = ny;
    lbm_sim->nz = nz;
    lbm_sim->nl = nx * ny * nz;

    lbm_sim->omega = omega;
   
    lbm_vars_alloc(&lbm_sim->h_vars, lbm_sim->nl);
    
    lbm_sim->switch_f0_f1 = false;

    lbm_vars_cuda_alloc(&lbm_sim->d_vars, lbm_sim->nl);

    HANDLE_ERROR(cudaMalloc(&lbm_sim->d_pal_lat, sizeof(double)*lbm_sim->nl*19 ));

    cudaGetDeviceProperties(&lbm_sim->prop, 0);

    dim3 dimComputationGrid(max((unsigned long)1, (unsigned long)nx/BLOCK_SIZE), min((unsigned long)ny, (unsigned long)lbm_sim->prop.maxGridSize[1]), min((unsigned long)nz, (unsigned long)lbm_sim->prop.maxGridSize[2]));
    dim3 dimComputationBlock(BLOCK_SIZE);
    lbm_sim->dimComputationGrid = dimComputationGrid;
    lbm_sim->dimComputationBlock = dimComputationBlock;

    lbm_sim->shared_mem_size = 0; //6 * sizeof(double) * BLOCK_SIZE;

    if ( cudaDeviceSetCacheConfig (cudaFuncCachePreferL1) != cudaSuccess)
        fprintf(stderr, "cudaFuncSetCacheConfig failed\n");

    return lbm_sim;
}

void lbm_simulation_destroy(lbm_simulation* lbm_sim)
{
    lbm_vars_dealloc(&lbm_sim->h_vars);
    lbm_vars_cuda_dealloc(&lbm_sim->d_vars);
    HANDLE_ERROR(cudaFree(lbm_sim->d_pal_lat));
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

inline void lbm_lattices_cuda_memcpy(lbm_lattices* lat0, lbm_lattices* lat1, size_t gi0, size_t gi1, size_t n, enum cudaMemcpyKind kind)
{
    size_t size = sizeof(double) * n;
    HANDLE_ERROR(cudaMemcpy(&lat0->ne[gi0], &lat1->ne[gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->e [gi0], &lat1->e [gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->se[gi0], &lat1->se[gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->n [gi0], &lat1->n [gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->c [gi0], &lat1->c [gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->s [gi0], &lat1->s [gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->nw[gi0], &lat1->nw[gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->w [gi0], &lat1->w [gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->sw[gi0], &lat1->sw[gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->te[gi0], &lat1->te[gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->tn[gi0], &lat1->tn[gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->tc[gi0], &lat1->tc[gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->ts[gi0], &lat1->ts[gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->tw[gi0], &lat1->tw[gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->be[gi0], &lat1->be[gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->bn[gi0], &lat1->bn[gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->bc[gi0], &lat1->bc[gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->bs[gi0], &lat1->bs[gi1], size, kind));
    HANDLE_ERROR(cudaMemcpy(&lat0->bw[gi0], &lat1->bw[gi1], size, kind));
}

inline void lbm_lattices_cuda_memcpy_subdomain(lbm_simulation* lbm_sim, lbm_lattices* lat0, lbm_lattices* lat1, lbm_box_3d subdomain, enum cudaMemcpyKind kind)
{
    int y1, z1;
    size_t snx = std::abs(subdomain.x0 - subdomain.x1) + 1;
    size_t sny = std::abs(subdomain.y0 - subdomain.y1) + 1;
    size_t snz = std::abs(subdomain.z0 - subdomain.z1) + 1;

    // Extend sub domain range to reduce cudaMemcpy calls
    if (snx >= lbm_sim->nx-4 && snz > 1) {
        snx = lbm_sim->nx;
        subdomain.x0 = 0;
        subdomain.x1 = lbm_sim->nx-1;

        if (snz >= lbm_sim->nz-4) {
            snz = lbm_sim->nz;
            subdomain.z0 = 0;
            subdomain.z1 = lbm_sim->nz-1;
        }
    }

    // Number of contiguous cells
    size_t ncc = snx; 

    if (snx == lbm_sim->nx) {
        // (Partial ?) contiguous plane XZ axis
        ncc *= snz;
        z1 = subdomain.z0;

        if ( snz == lbm_sim->nz ) {
            // Contiguous planes on XZ axis
            ncc *= sny;
            y1 = subdomain.y0; 
        } else {
            // Partial planes  on XZ axis
            y1 = subdomain.y1;
        }
    } else {
        // (#sny*snz) Partial contiguous lines on X axis
        y1 = subdomain.y1;
        z1 = subdomain.z1;
    }

    // Copy
    int x = subdomain.x0;

    for (int y = subdomain.y0; y <= y1; y++) {
        for (int z = subdomain.z0; z <= z1; z++) {
            int gi = IDX(x,y,z,lbm_sim->nx,lbm_sim->ny,lbm_sim->nz);
            lbm_lattices_cuda_memcpy(lat0, lat1, gi, gi, ncc, kind);
        }
    }
}

void lbm_read_palabos_subdomain(lbm_simulation* lbm_sim, double* pal_lat, lbm_box_3d subdomain)
{
    size_t snx = std::abs(subdomain.x0 - subdomain.x1) + 1;
    size_t sny = std::abs(subdomain.y0 - subdomain.y1) + 1;
    size_t snz = std::abs(subdomain.z0 - subdomain.z1) + 1;
    lbm_lattices* d_lat = lbm_sim->switch_f0_f1 ? &lbm_sim->d_vars.f1 : &lbm_sim->d_vars.f0;
    
    dim3 dimGrid(
        max((unsigned long)1, (unsigned long)snx/BLOCK_SIZE), 
        min((unsigned long)sny, (unsigned long)lbm_sim->prop.maxGridSize[1]), 
        min((unsigned long)snz, (unsigned long)lbm_sim->prop.maxGridSize[2])
    );
    dim3 dimBlock(BLOCK_SIZE);

    HANDLE_KERNEL_ERROR(lbm_load_subdomain_from_gpu<<<dimGrid, dimBlock, lbm_sim->shared_mem_size>>>(
        *d_lat, 
        lbm_sim->d_pal_lat, 
        subdomain, 
        snx, sny, snz, 
        lbm_sim->nx, lbm_sim->ny, lbm_sim->nz)
    );
    size_t size = sizeof(double) * snx * sny * snz * 19;

    HANDLE_ERROR(cudaMemcpy(pal_lat, lbm_sim->d_pal_lat, size, cudaMemcpyDeviceToHost));
}

void lbm_write_palabos_subdomain(lbm_simulation* lbm_sim, const double* pal_lat, lbm_box_3d subdomain)
{
    size_t snx = std::abs(subdomain.x0 - subdomain.x1) + 1;
    size_t sny = std::abs(subdomain.y0 - subdomain.y1) + 1;
    size_t snz = std::abs(subdomain.z0 - subdomain.z1) + 1;
    lbm_lattices* d_lat = lbm_sim->switch_f0_f1 ? &lbm_sim->d_vars.f1 : &lbm_sim->d_vars.f0;
    
    dim3 dimGrid(
        max((unsigned long)1, (unsigned long)snx/BLOCK_SIZE), 
        min((unsigned long)sny, (unsigned long)lbm_sim->prop.maxGridSize[1]), 
        min((unsigned long)snz, (unsigned long)lbm_sim->prop.maxGridSize[2])
    );
    dim3 dimBlock(BLOCK_SIZE);

    size_t size = sizeof(double) * snx * sny * snz * 19;
    HANDLE_ERROR(cudaMemcpy(lbm_sim->d_pal_lat, pal_lat, size, cudaMemcpyHostToDevice));

    HANDLE_KERNEL_ERROR(lbm_load_subdomain_to_gpu<<<dimGrid, dimBlock, lbm_sim->shared_mem_size>>>(
        *d_lat, 
        lbm_sim->d_pal_lat, 
        subdomain, 
        snx, sny, snz, 
        lbm_sim->nx, lbm_sim->ny, lbm_sim->nz)
    );
}

void lbm_lattices_read_subdomain(lbm_simulation* lbm_sim, lbm_lattices* h_lat, lbm_box_3d subdomain)
{
    lbm_lattices* d_lat = lbm_sim->switch_f0_f1 ? &lbm_sim->d_vars.f1 : &lbm_sim->d_vars.f0;
    lbm_lattices_cuda_memcpy_subdomain(lbm_sim, h_lat, d_lat, subdomain, cudaMemcpyDeviceToHost);
}

void lbm_lattices_read(lbm_simulation* lbm_sim, lbm_lattices* h_lat)
{
    size_t nl = lbm_sim->nl;
    lbm_lattices* d_lat = lbm_sim->switch_f0_f1 ? &lbm_sim->d_vars.f1 : &lbm_sim->d_vars.f0;
    lbm_lattices_cuda_memcpy(h_lat, d_lat, 0, 0, nl, cudaMemcpyDeviceToHost);
}

void lbm_lattices_write_subdomain(lbm_simulation* lbm_sim, lbm_lattices* h_lat, lbm_box_3d subdomain)
{
    lbm_lattices* d_lat = lbm_sim->switch_f0_f1 ? &lbm_sim->d_vars.f1 : &lbm_sim->d_vars.f0;
    lbm_lattices_cuda_memcpy_subdomain(lbm_sim, d_lat, h_lat, subdomain, cudaMemcpyHostToDevice);
}

void lbm_lattices_write(lbm_simulation* lbm_sim, lbm_lattices* h_lat)
{
    size_t nl = lbm_sim->nl;
    lbm_lattices* d_lat = lbm_sim->switch_f0_f1 ? &lbm_sim->d_vars.f1 : &lbm_sim->d_vars.f0;
    lbm_lattices_cuda_memcpy(d_lat, h_lat, 0, 0, nl, cudaMemcpyHostToDevice);
}


void lbm_u_read(lbm_simulation* lbm_sim, lbm_u* u)
{
    HANDLE_ERROR(cudaMemcpy(u->u0, lbm_sim->d_vars.u.u0, sizeof(double)*lbm_sim->nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(u->u1, lbm_sim->d_vars.u.u1, sizeof(double)*lbm_sim->nl, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(u->u2, lbm_sim->d_vars.u.u2, sizeof(double)*lbm_sim->nl, cudaMemcpyDeviceToHost));
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

lbm_u* lbm_u_create(size_t nx, size_t ny, size_t nz)
{
    lbm_u* u = (lbm_u*) malloc(sizeof(lbm_u));
    lbm_u_alloc(u, nx*ny*nz);
    return u;
}

void lbm_u_destroy(lbm_u* u)
{
    lbm_u_dealloc(u);
    free(u);
}
