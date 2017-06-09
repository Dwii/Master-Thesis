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
#define NX       100//420         // Numer of lattice nodes (width)
#define NY       100//180         // Number of lattice nodes (height)
#define NZ       10 //30          // Number of lattice nodes (depth)
#define NL       ((NX)*(NY)*(NZ))    // Number of lattice nodes (total)
#define LY       ((NY) - 1)  // Height of the domain in lattice units
#define CX       ((NX) / 4)  // X coordinates of the cylinder
#define CY       ((NY) / 2)  // Y coordinates of the cylinder
#define CZ       ((NZ) / 2)  // Y coordinates of the cylinder
#define R        ((NY) / 9)  // Cylinder radius
#define ULB      0.04        // Velocity in lattice units
#define NULB     ((ULB) * 10 / (RE))   // Viscoscity in lattice units
#define OMEGA    ((double)1. / (3*(NULB)+0.5))  // Relaxation parameter

#define BLOCK_SIZE 64

#define SQUARE(a) ((a)*(a))
#define GPU_SQUARE(a) __dmul_rn(a,a)

#define IDX(x, y, z) ((x+NX)%(NX) + ((y+NY)%(NY) + ( (z+NZ)%(NZ) )*(NY))*(NX) )

struct lbm_lattices{
    // Middle plane
    double ne[NL];  // [ 1, 1,  0]   1./36   (1./36)
    double  e[NL];  // [ 1, 0,  0]   1./18   (1./9 )
    double se[NL];  // [ 1,-1,  0]   1./36   (1./36)
    double  n[NL];  // [ 0, 1,  0]   1./18   (1./9 )
    double  c[NL];  // [ 0, 0,  0]   1./3    (4./9 )
    double  s[NL];  // [ 0,-1,  0]   1./18   (1./9 )
    double nw[NL];  // [-1, 1,  0]   1./36   (1./36)
    double  w[NL];  // [-1, 0,  0]   1./18   (1./9 )
    double sw[NL];  // [-1,-1,  0]   1./36   (1./36)
    // Top plane
    double te[NL];  // [ 1, 0,  1]   1./36
    double tn[NL];  // [ 0, 1,  1]   1./36
    double tc[NL];  // [ 0, 0,  1]   1./18
    double ts[NL];  // [ 0,-1,  1]   1./36
    double tw[NL];  // [-1, 0,  1]   1./36
    // Bottom plane
    double be[NL];  // [ 1, 0, -1]   1./36
    double bn[NL];  // [ 0, 1, -1]   1./36
    double bc[NL];  // [ 0, 0, -1]   1./18
    double bs[NL];  // [ 0,-1, -1]   1./36
    double bw[NL];  // [-1, 0, -1]   1./36
};

struct lbm_u {
    double u0[NL];
    double u1[NL];
    double u2[NL];
};

typedef struct {
    bool obstacles[NL];  // Should reside in lbm_consts but is too big for constant memory
    double u0[NL];
    double u1[NL];
    double u2[NL];
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
            for (int z = 0; z < NZ; z++) {
                obstacles[IDX(x,y,z)] = false; //SQUARE(x-CX) + SQUARE(y-CY) < SQUARE(R);
            }
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
        vel[y] = 0; // ULB * (1 + 0.0001 * sin( y / (double)LY * 2 * M_PI) );
    }
}

#define H_EQUILIBRIUM(rho, t, cu, usqr) ((rho) * (t) * ( 1 + (cu) + 0.5 * SQUARE(cu) - (usqr) ))
#define D_EQUILIBRIUM(rho, t, cu, usqr) __dmul_rn(__dmul_rn(rho, (t)), __dadd_rn(__dadd_rn(__dadd_rn(1, cu) , __dmul_rn(0.5, GPU_SQUARE(cu))), - usqr) )


__host__ static void h_equilibrium(lbm_lattices* f, int index, double rho, double u0, double u1, double u2)
{
    double usqr = 3./2 * ( SQUARE(u0) + SQUARE(u1) + SQUARE(u2) );

    { double cu = 3 * (  u0 +  u1 ); f->ne[index] = H_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * (  u0       ); f->e [index] = H_EQUILIBRIUM(rho, 1./18, cu, usqr ); } 
    { double cu = 3 * (  u0 + -u1 ); f->se[index] = H_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * (        u1 ); f->n [index] = H_EQUILIBRIUM(rho, 1./18, cu, usqr ); } 
    { double cu = 0                ; f->c [index] = H_EQUILIBRIUM(rho, 1./3 , cu, usqr ); } 
    { double cu = 3 * (       -u1 ); f->s [index] = H_EQUILIBRIUM(rho, 1./18, cu, usqr ); } 
    { double cu = 3 * ( -u0 +  u1 ); f->nw[index] = H_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * ( -u0       ); f->w [index] = H_EQUILIBRIUM(rho, 1./18, cu, usqr ); } 
    { double cu = 3 * ( -u0 + -u1 ); f->sw[index] = H_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * (  u0 +  u2 ); f->te[index] = H_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * (  u1 +  u2 ); f->tn[index] = H_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * (        u2 ); f->tc[index] = H_EQUILIBRIUM(rho, 1./18, cu, usqr ); } 
    { double cu = 3 * ( -u1 +  u2 ); f->ts[index] = H_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * ( -u0 +  u2 ); f->tw[index] = H_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * (  u0 + -u2 ); f->be[index] = H_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * (  u1 + -u2 ); f->bn[index] = H_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * (       -u2 ); f->bc[index] = H_EQUILIBRIUM(rho, 1./18, cu, usqr ); } 
    { double cu = 3 * ( -u1 + -u2 ); f->bs[index] = H_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * ( -u0 + -u2 ); f->bw[index] = H_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
}


__device__ static void d_equilibrium(double* ne, double* e, double* se, double* n, double* c, double* s, double* nw, double* w, double* sw, 
                                     double* te, double* tn, double* tc, double* ts, double* tw,
                                     double* be, double* bn, double* bc, double* bs, double* bw,
                                     double rho, double u0, double u1, double u2)
{
    double usqr = __dmul_rn(3./2, __dadd_rn( __dadd_rn( GPU_SQUARE(u0), GPU_SQUARE(u1)), GPU_SQUARE(u2) ));

    { double cu = 3 * (  u0 +  u1 ); *ne = D_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * (  u0       ); *e  = D_EQUILIBRIUM(rho, 1./18, cu, usqr ); } 
    { double cu = 3 * (  u0 + -u1 ); *se = D_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * (        u1 ); *n  = D_EQUILIBRIUM(rho, 1./18, cu, usqr ); } 
    { double cu = 0                ; *c  = D_EQUILIBRIUM(rho, 1./3 , cu, usqr ); } 
    { double cu = 3 * (       -u1 ); *s  = D_EQUILIBRIUM(rho, 1./18, cu, usqr ); } 
    { double cu = 3 * ( -u0 +  u1 ); *nw = D_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * ( -u0       ); *w  = D_EQUILIBRIUM(rho, 1./18, cu, usqr ); } 
    { double cu = 3 * ( -u0 + -u1 ); *sw = D_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * (  u0 +  u2 ); *te = D_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * (  u1 +  u2 ); *tn = D_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * (        u2 ); *tc = D_EQUILIBRIUM(rho, 1./18, cu, usqr ); } 
    { double cu = 3 * ( -u1 +  u2 ); *ts = D_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * ( -u0 +  u2 ); *tw = D_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * (  u0 + -u2 ); *be = D_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * (  u1 + -u2 ); *bn = D_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * (       -u2 ); *bc = D_EQUILIBRIUM(rho, 1./18, cu, usqr ); } 
    { double cu = 3 * ( -u1 + -u2 ); *bs = D_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
    { double cu = 3 * ( -u0 + -u2 ); *bw = D_EQUILIBRIUM(rho, 1./36, cu, usqr ); } 
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

__global__ void lbm_right_wall(lbm_lattices* f)
{
    for (int z = blockIdx.z; z < NZ; z+=gridDim.z) {
        for (int y = blockIdx.y; y < NY; y+=gridDim.y) {
           // Right wall: outflow condition.
           f->nw[IDX(NX-1,y,z)] = f->nw[IDX(NX-2,y,z)];
           f->w [IDX(NX-1,y,z)] = f->w [IDX(NX-2,y,z)];
           f->sw[IDX(NX-1,y,z)] = f->sw[IDX(NX-2,y,z)];
        }
    }
}

__global__ void lbm_computation(lbm_vars *d_vars, lbm_lattices* f0, lbm_lattices* f1)
{
    int tix = threadIdx.x;

    for (int z = blockIdx.z; z < NZ; z+=gridDim.z) {
        for (int y = blockIdx.y; y < NY; y+=gridDim.y) {
            for (int x = threadIdx.x + blockIdx.x * blockDim.x; x < NX; x += blockDim.x * gridDim.x) {
                size_t gi = IDX(x,y,z);

                double fin_ne, fin_e, fin_se, fin_n, fin_c, fin_s, fin_nw, fin_w, fin_sw,
                       fin_te, fin_tn, fin_tc, fin_ts, fin_tw, 
                       fin_be, fin_bn, fin_bc, fin_bs, fin_bw;
                double fout_ne, fout_e, fout_se, fout_n, fout_c, fout_s, fout_nw, fout_w, fout_sw,
                       fout_te, fout_tn, fout_tc, fout_ts, fout_tw, 
                       fout_be, fout_bn, fout_bc, fout_bs, fout_bw;

                fin_ne = f0->ne[gi];
                fin_e  = f0->e [gi];
                fin_se = f0->se[gi];
                fin_n  = f0->n [gi];
                fin_c  = f0->c [gi];
                fin_s  = f0->s [gi];
                fin_nw = f0->nw[gi];
                fin_w  = f0->w [gi];
                fin_sw = f0->sw[gi];
                fin_te = f0->te[gi];
                fin_tn = f0->tn[gi];
                fin_tc = f0->tc[gi];
                fin_ts = f0->ts[gi];
                fin_tw = f0->tw[gi];
                fin_be = f0->be[gi];
                fin_bn = f0->bn[gi];
                fin_bc = f0->bc[gi];
                fin_bs = f0->bs[gi];
                fin_bw = f0->bw[gi];

                // Compute macroscopic variables, density and velocity
                double rho, u0, u1, u2;
                macroscopic(fin_ne, fin_e, fin_se, fin_n, fin_c, fin_s, fin_nw, fin_w, fin_sw, 
                            fin_te, fin_tn, fin_tc, fin_ts, fin_tw, 
                            fin_be, fin_bn, fin_bc, fin_bs, fin_bw,
                            &rho, &u0, &u1, &u2);

                
//                if (x == 0) {
//                    // Left wall: inflow condition
//                    u0 = d_consts.vel[y];
//                    u1 = 0;
//                    u2 = 0;
//
//                    // Calculate the density
//                    double s2 = fin_n  + fin_c + fin_s + fin_tn + fin_tc + fin_ts + fin_bn + fin_bc + fin_bs;
//                    double s3 = fin_nw + fin_w + fin_sw + fin_tw + fin_bw;
//                    rho = 1./(1 - u0) * (s2 + 2*s3);
//                }
//          

                // Compute equilibrium
                double feq_ne, feq_e, feq_se, feq_n, feq_c, feq_s, feq_nw, feq_w, feq_sw, 
                       feq_te, feq_tn, feq_tc, feq_ts, feq_tw, 
                       feq_be, feq_bn, feq_bc, feq_bs, feq_bw;
                d_equilibrium(&feq_ne, &feq_e, &feq_se, &feq_n, &feq_c, &feq_s, &feq_nw, &feq_w, &feq_sw, 
                              &feq_te, &feq_tn, &feq_tc, &feq_ts, &feq_tw, 
                              &feq_be, &feq_bn, &feq_bc, &feq_bs, &feq_bw, 
                              rho, u0, u1, u2);       

//     
//                if (x == 0) {
//                    fin_ne = feq_ne + fin_sw - feq_sw;
//                    fin_e  = feq_e  + fin_w  - feq_w ;
//                    fin_se = feq_se + fin_nw - feq_nw;
//                }
//

                if (d_vars->obstacles[IDX(x, y, z)]) {
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
                    fout_ne = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_ne, - feq_ne)), fin_ne);
                    fout_e  = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_e , - feq_e )), fin_e );
                    fout_se = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_se, - feq_se)), fin_se);
                    fout_n  = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_n , - feq_n )), fin_n );
                    fout_c  = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_c , - feq_c )), fin_c );
                    fout_s  = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_s , - feq_s )), fin_s );
                    fout_nw = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_nw, - feq_nw)), fin_nw);
                    fout_w  = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_w , - feq_w )), fin_w );
                    fout_sw = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_sw, - feq_sw)), fin_sw);

                    fout_te = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_te, - feq_te)), fin_te);
                    fout_tn = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_tn, - feq_tn)), fin_tn);
                    fout_tc = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_tc, - feq_tc)), fin_tc);
                    fout_ts = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_ts, - feq_ts)), fin_ts);
                    fout_tw = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_tw, - feq_tw)), fin_tw);
                    fout_be = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_be, - feq_be)), fin_be);
                    fout_bn = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_bn, - feq_bn)), fin_bn);
                    fout_bc = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_bc, - feq_bc)), fin_bc);
                    fout_bs = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_bs, - feq_bs)), fin_bs);
                    fout_bw = __dadd_rn(__dmul_rn(-OMEGA, __dadd_rn(fin_bw, - feq_bw)), fin_bw);
                }


        		d_vars->u0[gi] = u0;
        		d_vars->u1[gi] = u1;
                d_vars->u2[gi] = u2;

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
                f1->s[IDX(x, y-1, z)] = fout_s;
                f1->n[IDX(x, y+1, z)] = fout_n;

                // E propagation in shared memory
                if (tix < blockDim.x-1 && x < NX-1) {
                    fo_E [tix+1] = fout_e;
                    fo_NE[tix+1] = fout_ne;
                    fo_SE[tix+1] = fout_se;
                // E propagation in global memory (at block boundary)
                } else {
                    f1->e [IDX(x+1, y  , z)] = fout_e;
                    f1->se[IDX(x+1, y-1, z)] = fout_se;
                    f1->ne[IDX(x+1, y+1, z)] = fout_ne;
                }

                // W propagation in shared memory
                if (tix > 0) {
                    fo_W [tix-1] = fout_w;
                    fo_NW[tix-1] = fout_nw;
                    fo_SW[tix-1] = fout_sw;
                // W propagation in global memory (at block boundary)
                } else {
                    f1->w [IDX(x-1, y  , z)] = fout_w;
                    f1->sw[IDX(x-1, y-1, z)] = fout_sw;
                    f1->nw[IDX(x-1, y+1, z)] = fout_nw;
                }

                // Top and Bottom propagation (global memory)
                f1->te[IDX(x+1, y  , z+1)] = fout_te;
                f1->tn[IDX(x  , y+1, z+1)] = fout_tn;
                f1->tc[IDX(x  , y  , z+1)] = fout_tc;
                f1->ts[IDX(x  , y-1, z+1)] = fout_ts;
                f1->tw[IDX(x-1, y  , z+1)] = fout_tw;
                f1->be[IDX(x+1, y  , z-1)] = fout_be;
                f1->bn[IDX(x  , y+1, z-1)] = fout_bn;
                f1->bc[IDX(x  , y  , z-1)] = fout_bc;
                f1->bs[IDX(x  , y-1, z-1)] = fout_bs;
                f1->bw[IDX(x-1, y  , z-1)] = fout_bw;

                __syncthreads();

                // the leftmost thread is not updated in this block
                if (tix > 0) {
                    f1->e [gi            ] = fo_E [tix];
                    f1->se[IDX(x, y-1, z)] = fo_SE[tix];
                    f1->ne[IDX(x, y+1, z)] = fo_NE[tix];
                }

                // the rightmost thread is not updated in this block
                if (tix < blockDim.x-1 && x < NX-1) {
                    f1->w [gi            ] = fo_W [tix];
                    f1->sw[IDX(x, y-1, z)] = fo_SW[tix];
                    f1->nw[IDX(x, y+1, z)] = fo_NW[tix];
                }

                __syncthreads(); // only nessessary when NX % BLOCK_SIZE != 0 
           }
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

    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("free memory:  %lu\n", free);
    printf("total memory: %lu\n", total);
    printf("sizeof(lbm_vars) = %lu\n", sizeof(lbm_vars));

    lbm_simulation* lbm_sim = (lbm_simulation*) malloc (sizeof(lbm_simulation));
    
    lbm_consts h_consts;
    
    initVelocity(h_consts.vel);
    
    HANDLE_ERROR(cudaMemcpyToSymbol(d_consts, &h_consts, sizeof(lbm_consts)));
        
    initObstacles(lbm_sim->h_vars.obstacles);
    
    // Initialization of the populations at equilibrium with the given velocity.
    lbm_sim->switch_f0_f1 = false;
    for (int z = 0; z < NZ; z++) {
        for (int y = 0; y < NY; y++) {
            for (int x = 0; x < NX; x++) {
                double rho = x == NX/2 && y == NY/2 && z == NZ/2 ? 2.0 : 1.0;
                h_equilibrium(&lbm_sim->h_vars.f0, IDX(x,y,z), rho, h_consts.vel[y], 0, -h_consts.vel[y]);
            }
        }
    }

    HANDLE_ERROR(cudaMalloc(&lbm_sim->d_vars, sizeof(lbm_vars)));
    HANDLE_ERROR(cudaMemcpy(lbm_sim->d_vars, &lbm_sim->h_vars, sizeof(lbm_vars), cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    dim3 dimComputationGrid(max(1, NX/BLOCK_SIZE), min(NY, prop.maxGridSize[1]), min(NZ, prop.maxGridSize[2]));
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
//    HANDLE_ERROR(cudaFree(lbm_sim->d_vars));
    free(lbm_sim);
}
void lbm_simulation_update(lbm_simulation* lbm_sim)
{
    if (lbm_sim->switch_f0_f1) {
//        HANDLE_KERNEL_ERROR(lbm_right_wall<<<lbm_sim->dimRightWallGrid, lbm_sim->dimRightWallBlock>>>(&lbm_sim->d_vars->f1));
        HANDLE_KERNEL_ERROR(lbm_computation<<<lbm_sim->dimComputationGrid, lbm_sim->dimComputationBlock, lbm_sim->shared_mem_size>>>(lbm_sim->d_vars, &lbm_sim->d_vars->f1, &lbm_sim->d_vars->f0));
    } else {
//        HANDLE_KERNEL_ERROR(lbm_right_wall<<<lbm_sim->dimRightWallGrid, lbm_sim->dimRightWallBlock>>>(&lbm_sim->d_vars->f0));
        HANDLE_KERNEL_ERROR(lbm_computation<<<lbm_sim->dimComputationGrid, lbm_sim->dimComputationBlock, lbm_sim->shared_mem_size>>>(lbm_sim->d_vars, &lbm_sim->d_vars->f0, &lbm_sim->d_vars->f1));
    }

    lbm_sim->switch_f0_f1 = ! lbm_sim->switch_f0_f1;
}

void lbm_simulation_get_size(lbm_simulation* lbm_sim, size_t* width, size_t* height, size_t* depth)
{
    *width  = NX;
    *height = NY;
    *depth  = NZ;
}

void lbm_lattices_read(lbm_simulation* lbm_sim, lbm_lattices* lat)
{
//    lbm_lattices* d_lat = &lbm_sim->d_vars->out;
    lbm_lattices* d_lat = lbm_sim->switch_f0_f1 ? &lbm_sim->d_vars->f1 : &lbm_sim->d_vars->f0;
    HANDLE_ERROR(cudaMemcpy(lat, d_lat, sizeof(lbm_lattices), cudaMemcpyDeviceToHost));
}

void lbm_u_read(lbm_simulation* lbm_sim, lbm_u* u)
{
    HANDLE_ERROR(cudaMemcpy(u->u0, lbm_sim->d_vars->u0, sizeof(double)*NL, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(u->u1, lbm_sim->d_vars->u1, sizeof(double)*NL, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(u->u2, lbm_sim->d_vars->u2, sizeof(double)*NL, cudaMemcpyDeviceToHost));
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

void lbm_lattices_at_index(lbm_lattice* lattice, lbm_lattices* lattices, int x, int y, int z)
{
    int gi = IDX(x,y,z);
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
    lattice->te = lat->te[gi];
    lattice->tn = lat->tn[gi];
    lattice->tc = lat->tc[gi];
    lattice->ts = lat->ts[gi];
    lattice->tw = lat->tw[gi];
    lattice->be = lat->be[gi];
    lattice->bn = lat->bn[gi];
    lattice->bc = lat->bc[gi];
    lattice->bs = lat->bs[gi];
    lattice->bw = lat->bw[gi];
}

void lbm_u_at_index(double* u0, double* u1, double* u2, lbm_u* u, int x, int y, int z)
{
    int gi = IDX(x,y,z);
    *u0 = u->u0[gi];
    *u1 = u->u1[gi];
    *u2 = u->u2[gi];
}