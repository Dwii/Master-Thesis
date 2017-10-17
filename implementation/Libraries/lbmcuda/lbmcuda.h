/*!
 * \file    lbm.h
 * \brief   Functions requierd by the lbmmain library to work properly.
 * \author  Adrien Python
 * \date    10.05.2017
 */

#ifndef LBM_SIMULATION
#define LBM_SIMULATION

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define IDX(x, y, z, nx, ny, nz) ((x+(nx))%(nx) + ((z+(nz))%(nz) + ( (y+(ny))%(ny) )*(nz))*(nx) )

typedef struct {
    // Middle plane
    double* ne;  // [ 1, 1,  0]   1./36   (1./36)
    double*  e;  // [ 1, 0,  0]   1./18   (1./9 )
    double* se;  // [ 1,-1,  0]   1./36   (1./36)
    double*  n;  // [ 0, 1,  0]   1./18   (1./9 )
    double*  c;  // [ 0, 0,  0]   1./3    (4./9 )
    double*  s;  // [ 0,-1,  0]   1./18   (1./9 )
    double* nw;  // [-1, 1,  0]   1./36   (1./36)
    double*  w;  // [-1, 0,  0]   1./18   (1./9 )
    double* sw;  // [-1,-1,  0]   1./36   (1./36)
    // Top plane
    double* te;  // [ 1, 0,  1]   1./36
    double* tn;  // [ 0, 1,  1]   1./36
    double* tc;  // [ 0, 0,  1]   1./18
    double* ts;  // [ 0,-1,  1]   1./36
    double* tw;  // [-1, 0,  1]   1./36
    // Bottom plane
    double* be;  // [ 1, 0, -1]   1./36
    double* bn;  // [ 0, 1, -1]   1./36
    double* bc;  // [ 0, 0, -1]   1./18
    double* bs;  // [ 0,-1, -1]   1./36
    double* bw;  // [-1, 0, -1]   1./36
} lbm_lattices;

typedef struct {
    double* u0;
    double* u1;
    double* u2;
} lbm_u;

typedef struct lbm_simulation lbm_simulation;

typedef struct lbm_lattice {
    double ne, e, se, n, c, s, nw, w, sw, te, tn, tc, ts, tw, be, bn, bc, bs, bw;
} lbm_lattice;

lbm_simulation* lbm_simulation_create(size_t width, size_t height, size_t depth, double omega);
void lbm_simulation_destroy(lbm_simulation* lbm_sim);
void lbm_simulation_update(lbm_simulation* lbm_sim);

lbm_lattices* lbm_lattices_create(size_t size);
void lbm_lattices_destroy(lbm_lattices* lat);
void lbm_lattices_read(lbm_simulation* lbm_sim, lbm_lattices* lat);
void lbm_lattices_write(lbm_simulation* lbm_sim, lbm_lattices* h_lat);

lbm_u* lbm_u_create(size_t width, size_t height, size_t depth);
void lbm_u_destroy(lbm_u* u);
void lbm_u_read(lbm_simulation* lbm_sim, lbm_u* u);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* LBM_SIMULATION */
