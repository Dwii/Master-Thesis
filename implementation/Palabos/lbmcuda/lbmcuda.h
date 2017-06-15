/*!
 * \file    lbmcuda.h
 * \brief   LBM cuda library
 * \author  Adrien Python
 * \version 1.0
 * \date    14.06.2016
 */

#ifndef LBMCUDA_H
#define LBMCUDA_H


#define IDX(x, y, z, nx, ny, nz) ((x+(nx))%(nx) + ((y+(ny))%(ny) + ( (z+(nz))%(nz) )*(ny))*(nx) )

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

typedef struct lbm_simulation lbm_simulation; 

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

lbm_simulation* lbm_simulation_create(size_t nx, size_t ny, size_t nz, double omega);
void lbm_simulation_destroy(lbm_simulation* lbm_sim);
void lbm_simulation_update(lbm_simulation* lbm_sim);

lbm_lattices* lbm_lattices_create(size_t nl);
void lbm_lattices_destroy(lbm_lattices* lat);
void lbm_lattices_write(lbm_simulation* lbm_sim, lbm_lattices* h_lat, size_t nl);
void lbm_lattices_read(lbm_simulation* lbm_sim, lbm_lattices* h_lat);
    
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* LBMCUDA_H */
