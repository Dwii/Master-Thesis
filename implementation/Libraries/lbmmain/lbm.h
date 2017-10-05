/*!
 * \file    lbm.h
 * \brief   Functions requierd by the lbmmain library to work properly.
 * \author  Adrien Python
 * \date    06.05.2017
 */

#ifndef LBM_SIMULATION
#define LBM_SIMULATION

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef struct lbm_lattices lbm_lattices;
typedef struct lbm_u lbm_u;
typedef struct lbm_simulation lbm_simulation;

typedef struct lbm_lattice {
    double ne, e, se, n, c, s, nw, w, sw;
} lbm_lattice;

lbm_simulation* lbm_simulation_create();
void lbm_simulation_destroy(lbm_simulation* lbm_sim);
void lbm_simulation_update(lbm_simulation* lbm_sim);
void lbm_simulation_get_size(lbm_simulation* lbm_sim, size_t* width, size_t* height);

lbm_lattices* lbm_lattices_create();
void lbm_lattices_destroy(lbm_lattices* lat);
void lbm_lattices_read(lbm_simulation* lbm_sim, lbm_lattices* lat);
void lbm_lattices_at_index(lbm_lattice* lattice, lbm_lattices* lattices, int x, int y);

lbm_u* lbm_u_create();
void lbm_u_destroy(lbm_u* u);
void lbm_u_read(lbm_simulation* lbm_sim, lbm_u* u);
void lbm_u_at_index(double* u0, double* u1, lbm_u* u, int x, int y);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* LBM_SIMULATION */
