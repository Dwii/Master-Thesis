/*!
 * \file    lbmFlowAroundCylinder.c
 * \brief   Palabos friendly version of lbm_py2c for variables alignments.
 * \author  Adrien Python
 * \date    20.12.2016
 */

#define _GNU_SOURCE

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <libgen.h>
#include <pgm.h>
#include <array.h>

#ifndef NX
#define NX       100//420         // Numer of lattice nodes (width)
#endif
#ifndef NY
#define NY       10//180         // Numer of lattice nodes (height)
#endif
#define RE       220.0       // Reynolds number
#define LY       ((NY) - 1)  // Height of the domain in lattice units
#define CX       ((NX) / 4)  // X coordinates of the cylinder
#define CY       ((NY) / 2)  // Y coordinates of the cylinder
#define R        ((NY) / 9)  // Cylinder radius
#define ULB      0.04        // Velocity in lattice units
#define NULB     ((ULB) * (R) / (RE))   // Viscoscity in lattice units
#define OMEGA    (1. / (3*(NULB)+0.5))  // Relaxation parameter

#define SQUARE(a) ((a)*(a))

typedef enum { OUT_FIN, OUT_IMG, OUT_UNK } out_mode;

const ssize_t V[][9] = {
    { 1, 1, 1, 0, 0, 0,-1,-1,-1 },
    { 1, 0,-1, 1, 0,-1, 1, 0,-1 }
};
const double T[] = { 1./36, 1./9, 1./36, 1./9, 4./9, 1./9, 1./36, 1./9, 1./36 };

/**
 * Setup: cylindrical obstacle and velocity inlet with perturbation
 * Creation of a mask with boolean values, defining the shape of the obstacle.
 */
static void initObstacles(bool** obstacles)
{
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            obstacles[x][y] = SQUARE(x-CX) + SQUARE(y-CY) < SQUARE(R);
        }
    }
}

/**
 * Initial velocity profile: almost zero, with a slight perturbation to trigger
 * the instability.
 */
static void initVelocity(double*** vel)
{
    for (int d = 0; d < 2; d++) {
        for (int x = 0; x < NX; x++) {
            for (int y = 0; y < NY; y++) {
                vel[d][x][y] = (1-d) * ULB * (1 + 0.0001 * sin( y / (double)LY * 2 * M_PI) );
            }
        }
    }
}

/**
 * Equilibrium distribution function.
 */
static void equilibrium(double*** feq, double** rho, double*** u, size_t x, size_t y)
{
    double usqr = 3./2 * ( SQUARE(u[0][x][y]) + SQUARE(u[1][x][y]) );
    
    for (int f = 0; f < 9; f++) {
        double cu = 3 * ( V[0][f] * u[0][x][y] + V[1][f] * u[1][x][y] );
        feq[x][y][f] = rho[x][y] * T[f] * ( 1 + cu + 0.5 * SQUARE(cu) - usqr );
    }
}

static void macroscopic(double*** fin, double** rho, double*** u, size_t x, size_t y)
{
    
    rho[x][y] = 0;
    u[0][x][y] = u[1][x][y] = 0;

    for (int f = 0; f < 9; f++) {
        rho[x][y] += fin[x][y][f];

        u[0][x][y] += V[0][f] * fin[x][y][f];
        u[1][x][y] += V[1][f] * fin[x][y][f];
    }
    
    u[0][x][y] /= rho[x][y];
    u[1][x][y] /= rho[x][y];
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

static bool print_double(size_t dim, void* data, const size_t index[dim], void* args)
{
    (void) dim, (void) index, (void) args;
    
    printf("%64.60f\n", *((double*)data));
    
    return true;
}

static inline void print_doubles(size_t dim, const size_t index[dim], void* array)
{
    array_foreach(dim, index, sizeof(double), array, NULL, print_double, NULL);
}

int main(int argc, char * const argv[])
{
    // Read arguments
    char* img_path = NULL;
    out_mode out = OUT_UNK;
    int max_iter = 0;
    
    while (optind < argc) {
        switch (getopt(argc, argv, "p:fi:")) {
            case 'p': { out = OUT_IMG; img_path = optarg; break; }
            case 'f': { out = OUT_FIN; break; }
            case 'i': { max_iter = strtol(optarg, NULL, 10); break; }
            default : { goto usage; }
        }
    }
    
    // check that execution mode is set (output images or fin values)
    if (out == OUT_UNK && max_iter < 1) {
    usage:
        fprintf(stderr, "usage: %s (-p <path> | -f) -i <iter> \n", basename((char*)argv[0]));
        fprintf(stderr, "  -f : output pictures in <path> directory\n");
        fprintf(stderr, "  -t : output populations values in stdout\n");
        fprintf(stderr, "  -i : Total number of iterations\n");
        return EXIT_FAILURE;
    }
    
    const size_t A_SIZE0[3] = {2, NX, NY};
    const size_t A_SIZE1[3] = {NX, NY, 9};
    const size_t A_SIZE2[2] = {NX, NY};
    
    double*** u = array_create(3, A_SIZE0, sizeof(double));    // velocity to impose on inflow boundary cells
    double*** feq = array_create(3, A_SIZE1, sizeof(double));  //
    double*** fin = array_create(3, A_SIZE1, sizeof(double));  // Incoming populations
    double*** fout = array_create(3, A_SIZE1, sizeof(double)); // Outgoing populations
    double** rho = array_create(2, A_SIZE2, sizeof(double));   // Density
    bool** obstacles = array_create(2, A_SIZE2, sizeof(bool)); // Obstacles definition
    double*** vel = array_create(3, A_SIZE0, sizeof(double));  // Velocity
    
    size_t COL1[3], COL2[3], COL3[3], OPP[9];
    
    initCol(COL1,  1);
    initCol(COL2,  0);
    initCol(COL3, -1);
    initOpp(OPP);
    
    {
        double one = 1;
        array_set(2, A_SIZE2, sizeof(double), rho, &one);
    }
    
    initObstacles(obstacles);
    initVelocity(vel);
    
    // Initialization of the populations at equilibrium with the given velocity.
    
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            equilibrium(fin, rho, vel, x, y);
        }
    }

    pgm_image* pgm = pgm_create(NX, NY);
    
    for (int time = 0; time < max_iter; time++) {
        
        // Right wall: outflow condition.
        for (size_t i = 0; i < 3; i++) {
            for (size_t y = 0, f = COL3[i]; y < NY; y++) {
                fin[NX-1][y][f] = fin[NX-2][y][f];
            }
        }
        
        for (size_t y = 0; y < NY; y++) {
            // Compute macroscopic variables, density and velocity
            for (size_t x = 0; x < NX; x++) {
                macroscopic(fin, rho, u, x, y);
            }
            
            // Left wall: inflow condition
            for (size_t d = 0; d < 2; d++) {
                u[d][0][y] = vel[d][0][y];
            }
            
            // Calculate the density
            double s2 = 0, s3 = 0;
            for (size_t i = 0; i < 3; i++) {
                s2 += fin[0][y][COL2[i]];
                s3 += fin[0][y][COL3[i]];
            }
            rho[0][y] = 1./(1 - u[0][0][y]) * (s2 + 2*s3);
            
            // Compute equilibrium
            for (size_t x = 0; x < NX; x++) {
                equilibrium(feq, rho, u, x, y);
            }
            
            for (size_t i = 0, f = COL1[i]; i < 3; f = COL1[++i]) {
                fin[0][y][f] = feq[0][y][f] + fin[0][y][OPP[f]] - feq[0][y][OPP[f]];
            }
            
            for (size_t x = 0; x < NX; x++) {
                for (size_t f = 0; f < 9; f++) {
                    if (obstacles[x][y]) {
                        // Bounce-back condition for obstacle
                        fout[x][y][f] = fin[x][y][OPP[f]];
                    } else {
                        // Collision step
                        fout[x][y][f] = fin[x][y][f] - OMEGA * (fin[x][y][f] - feq[x][y][f]);
                    }
                }
            }
        }
        
        // Streaming step
        for (size_t x = 0; x < NX; x++) {
            for (size_t y = 0; y < NY; y++) {
                for (size_t f = 0; f < 9; f++) {
                    size_t x_dst = (x + NX + V[0][f]) % NX;
                    size_t y_dst = (y + NY + V[1][f]) % NY;
                    fin[x_dst][y_dst][f] = fout[x][y][f];
                }
            }
        }
        
        // Visualization of the velocity.
        if (time % 100 == 0 && out == OUT_IMG) {
            double vel[NX][NY];
            for (size_t x = 0; x < NX; x++) {
                for (size_t y = 0; y < NY; y++) {
                    vel[x][y] = sqrt( SQUARE(u[0][x][y]) + SQUARE(u[1][x][y]) );
                    int color =  255 * vel[x][y] * 10;
                    pgm_set_pixel(pgm, x, y, color);
                }
            }
            // build image file path and create it
            char* filename;
            asprintf(&filename, "%s/vel_%d.pgm", img_path, time/100);
            pgm_write(pgm, filename);
            free(filename);
        }
    }
    
    if (out == OUT_FIN) {
        print_doubles(3, A_SIZE1, fin);
    }

    pgm_destroy(pgm);

    array_destroy(3, A_SIZE0, sizeof(double), u);
    array_destroy(3, A_SIZE1, sizeof(double), feq);
    array_destroy(3, A_SIZE1, sizeof(double), fin);
    array_destroy(3, A_SIZE1, sizeof(double), fout);
    array_destroy(2, A_SIZE2, sizeof(double), rho);
    array_destroy(2, A_SIZE2, sizeof(bool), obstacles);
    array_destroy(3, A_SIZE0, sizeof(double), vel);
    
    return EXIT_SUCCESS;
}
