/*!
 * \file    lbmSimple.c
 * \brief   Simple LBM 3d implementation.
 * \author  Adrien Python
 * \date    17.05.2017
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <lbm.h>

#define RE       220.0       // Reynolds number
#define NX       100//420         // Numer of lattice nodes (width)
#define NY       100//180         // Numer of lattice nodes (height)
#define NZ       10 //30          // Numer of lattice nodes (depth)
#define NF       19          // Population size
#define ND       3           // Numer of dimensions
#define LY       ((NY) - 1)  // Height of the domain in lattice units
#define CX       ((NX) / 4)  // X coordinates of the cylinder
#define CY       ((NY) / 2)  // Y coordinates of the cylinder
#define R        ((NY) / 9)  // Cylinder radius
#define ULB      0.04        // Velocity in lattice units
#define NULB     ((ULB) * (R) / (RE))   // Viscoscity in lattice units
#define OMEGA    ((double)1. / (3*(NULB)+0.5))  // Relaxation parameter

#define SQUARE(a) ((a)*(a))
#define INDEX_2D_FROM_1D(x, y, i) do { (y) = (i)/(NX), (x) = (i)%(NX); } while (0)

typedef enum { OUT_NONE, OUT_FIN, OUT_IMG } out_mode;

struct lbm_simulation {
//    size_t col[3][3];
    size_t opp[NF];
    bool obstacles[NX][NY][NZ];
    double u[NX][NY][NZ][ND];
    double feq[NX][NY][NZ][NF];
    double fin[NX][NY][NZ][NF];
    double fout[NX][NY][NZ][NF];
    double rho[NX][NY][NZ];
    double vel[NX][NY][NZ][ND];
};

struct lbm_lattices {
    double f[NX][NY][NZ][NF];
};

struct lbm_u {
    double u0[NX][NY][NZ];
    double u1[NX][NY][NZ];
    double u2[NX][NY][NZ];
};

// Constants

const ssize_t V[ND][NF] = {
//   ne  e se  n  c  s nw  w sw te tn tc ts tw be bn bc bs bw
    { 1, 1, 1, 0, 0, 0,-1,-1,-1, 1, 0, 0, 0,-1, 1, 0, 0, 0,-1 },
    { 1, 0,-1, 1, 0,-1, 1, 0,-1, 0, 1, 0,-1, 0, 0, 1, 0,-1, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1 },
};

const double T[NF] = {
//     ne      e     se      n     c      s     nw      w     sw
    1./36, 1./18, 1./36, 1./18, 1./3, 1./18, 1./36, 1./18, 1./36,
//     te     tn     tc     ts     tw
    1./36, 1./36, 1./18, 1./36, 1./36,
//     be     bn     bc     bs     bw
    1./36, 1./36, 1./18, 1./36, 1./36,
};

/**
 * Setup: cylindrical obstacle and velocity inlet with perturbation
 * Creation of a mask with boolean values, defining the shape of the obstacle.
 */
static void initObstacles(lbm_simulation *lbm_sim)
{
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            for (int z = 0; z < NZ; z++) {
                lbm_sim->obstacles[x][y][z] = false;
            }
        }
    }
}

/**
 * Initial velocity profile: almost zero, with a slight perturbation to trigger
 * the instability.
 */
static void initVelocity(lbm_simulation *lbm_sim)
{
    for (int d = 0; d < ND; d++) {
        for (int x = 0; x < NX; x++) {
            for (int y = 0; y < NY; y++) {
                for (int z = 0; z < NZ; z++) {
                    lbm_sim->vel[x][y][z][d] = (1-d) * ULB * (1 + 0.0001 * sin( y / (double)LY * 2 * M_PI) );
                }
            }
        }
    }
}

static void initRho(lbm_simulation *lbm_sim)
{
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            for (int z = 0; z < NZ; z++) {
                lbm_sim->rho[x][y][z] = 1.0;
            }
        }
    }
    lbm_sim->rho[NX/2][NY/2][NZ/2] = 2.0;
}

/*
static void initCol(size_t* col, ssize_t v0)
{
    for (int f = 0, i = 0; f < NF && i < 3; f++) {
        if (V[0][f] == v0) {
            col[i++] = f;
        }
    }
}
*/

static void initOpp(size_t* opp)
{
    for (int f = 0; f < NF; f++) {
        for (int g = 0; g < NF; g++) {
            if (V[0][f] == -V[0][g] && V[1][f] == -V[1][g] && V[2][f] == -V[2][g] ) {
                opp[f] = g;
                break;
            }
        }
    }
}

static void equilibrium(double* feq, double rho, double* u)
{
    double usqr = 3./2 * ( SQUARE(u[0]) + SQUARE(u[1]) + SQUARE(u[2]));
    for (int f = 0; f < NF; f++) {
        double cu = 3 * ( V[0][f] * u[0] + V[1][f] * u[1] + V[2][f] * u[2] );
        feq[f] = rho * T[f] * ( 1 + cu + 0.5 * SQUARE(cu) - usqr );
    }
}

static void macroscopic(double* fin, double* rho, double* u)
{
    
    *rho = u[0] = u[1] = u[2] = 0;
    
    for (int f = 0; f < NF; f++) {
        *rho += fin[f];
        
        for (int d = 0; d < ND; d++) {
            u[d] += V[d][f] * fin[f];
        }
    }
    
    for (int d = 0; d < ND; d++) {
        u[d] /= *rho;
    }
}
int i =0 ;
void lbm_computation(lbm_simulation *lbm_sim, int x, int y, int z)
{
    /*
    // Right wall: outflow condition.
    if (x == NX-1) {
        for (int i = 0; i < 3; i++) {
            size_t f = d_consts.col[2][i];
            d_vars->fin[NX-1][y][z][f] = d_vars->fin[NX-2][y][z][f];
        }
    }
     */
    // Compute macroscopic variables, density and velocity
    macroscopic(lbm_sim->fin[x][y][z], &lbm_sim->rho[x][y][z], lbm_sim->u[x][y][z]);

    /*
    if (x == 0) {
        // Left wall: inflow condition
        for (size_t d = 0; d < ND; d++) {
            d_vars->u[0][y][z][d] = d_vars->vel[0][y][z][d];
        }

        // Calculate the density
        double s2 = 0, s3 = 0;
        for (size_t i = 0; i < 3; i++) {
            s2 += d_vars->fin[0][y][z][d_consts.col[1][i]];
            s3 += d_vars->fin[0][y][z][d_consts.col[2][i]];
        }
        d_vars->rho[0][y][z] = 1./(1 - d_vars->u[0][y][z][0]) * (s2 + 2*s3);
    }
     */
    // Compute equilibrium
    equilibrium(lbm_sim->feq[x][y][z], lbm_sim->rho[x][y][z], lbm_sim->u[x][y][z]);

    /*
    if (x == 0) {
        for (size_t i = 0; i < 3; i++) {
            size_t f = d_consts.col[0][i];
            d_vars->fin[0][y][z][f] = d_vars->feq[0][y][z][f] + d_vars->fin[0][y][z][d_consts.opp[f]] - d_vars->feq[0][y][z][d_consts.opp[f]];
        }
    }
     */
    for (size_t f = 0; f < NF; f++) {
        if (lbm_sim->obstacles[x][y][z]) {
            // Bounce-back condition for obstacle
            lbm_sim->fout[x][y][z][f] = lbm_sim->fin[x][y][z][lbm_sim->opp[f]];
        } else {
            // Collision step
            lbm_sim->fout[x][y][z][f] = lbm_sim->fin[x][y][z][f] - OMEGA * (lbm_sim->fin[x][y][z][f] - lbm_sim->feq[x][y][z][f]);
        }
    }
}

void lbm_streaming(lbm_simulation *lbm_sim, int x, int y, int z)
{
    // Streaming step
    for (size_t f = 0; f < NF; f++) {
        ssize_t x_dst = (x + V[0][f] + NX) % NX;
        ssize_t y_dst = (y + V[1][f] + NY) % NY;
        ssize_t z_dst = (z + V[2][f] + NZ) % NZ;
        //if (0 <= x_dst && x_dst < NX && 0 <= y_dst && y_dst < NY && 0 <= z_dst && z_dst < NZ)
        lbm_sim->fin[x_dst][y_dst][z_dst][f] = lbm_sim->fout[x][y][z][f];
    }
}

lbm_simulation* lbm_simulation_create()
{
    lbm_simulation* lbm_sim = (lbm_simulation*) malloc (sizeof(lbm_simulation));
/*
    initCol(lbm_sim->col[0],  1);
    initCol(lbm_sim->col[1],  0);
    initCol(lbm_sim->col[2], -1);
*/
    initOpp(lbm_sim->opp);
    
    initObstacles(lbm_sim);
    initVelocity(lbm_sim);
    initRho(lbm_sim);

    // Initialization of the populations at equilibrium with the given velocity.
    for (int z = 0; z < NZ; z++) {
        for (int y = 0; y < NY; y++) {
            for (int x = 0; x < NX; x++) {
                equilibrium(lbm_sim->fin[x][y][z], lbm_sim->rho[x][y][z], lbm_sim->vel[x][y][z]);
            }
        }
    }

    return lbm_sim;
}

void lbm_simulation_destroy(lbm_simulation* lbm_sim)
{
    free(lbm_sim);
}

void lbm_simulation_update(lbm_simulation* lbm_sim)
{
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            for (int z = 0; z < NZ; ++z) {
                lbm_computation(lbm_sim, x, y, z);
            }
        }
    }

    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            for (int z = 0; z < NZ; ++z) {
                lbm_streaming(lbm_sim, x, y, z);
            }
        }
    }
}

void lbm_simulation_get_size(lbm_simulation* lbm_sim, size_t* width, size_t* height, size_t* depth)
{
    *width  = NX;
    *height = NY;
    *depth  = NZ;
}

void lbm_lattices_read(lbm_simulation* lbm_sim, lbm_lattices* lat)
{
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            for (int z = 0; z < NZ; ++z) {
                for (int f = 0; f < NF; ++f) {
                    lat->f[x][y][z][f] = lbm_sim->fin[x][y][z][f];
                }
            }
        }
    }
}

void lbm_u_read(lbm_simulation* lbm_sim, lbm_u* u)
{
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            for (int z = 0; z < NZ; ++z) {
                u->u0[x][y][z] = lbm_sim->u[x][y][z][0];
                u->u1[x][y][z] = lbm_sim->u[x][y][z][1];
                u->u2[x][y][z] = lbm_sim->u[x][y][z][2];
            }
        }
    }
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
    lattice->ne = lattices->f[x][y][z][0];
    lattice->e  = lattices->f[x][y][z][1];
    lattice->se = lattices->f[x][y][z][2];
    lattice->n  = lattices->f[x][y][z][3];
    lattice->c  = lattices->f[x][y][z][4];
    lattice->s  = lattices->f[x][y][z][5];
    lattice->nw = lattices->f[x][y][z][6];
    lattice->w  = lattices->f[x][y][z][7];
    lattice->sw = lattices->f[x][y][z][8];
    lattice->te = lattices->f[x][y][z][9];
    lattice->tn = lattices->f[x][y][z][10];
    lattice->tc = lattices->f[x][y][z][11];
    lattice->ts = lattices->f[x][y][z][12];
    lattice->tw = lattices->f[x][y][z][13];
    lattice->be = lattices->f[x][y][z][14];
    lattice->bn = lattices->f[x][y][z][15];
    lattice->bc = lattices->f[x][y][z][16];
    lattice->bs = lattices->f[x][y][z][17];
    lattice->bw = lattices->f[x][y][z][18];
}

void lbm_u_at_index(double* u0, double* u1, double* u2, lbm_u* u, int x, int y, int z)
{
    *u0 = u->u0[x][y][z];
    *u1 = u->u1[x][y][z];
    *u2 = u->u2[x][y][z];
}
