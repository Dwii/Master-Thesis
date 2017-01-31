/*!
 * \file    lbmFlowAroundCylinder.c
 * \brief   (Almost) "One to one" C implementation of lbmFlowAroundCylinder.py
 * \author  Adrien Python
 * \date    21.11.2016
 */

#include <libc.h>
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

const ssize_t V[][2] = {
    { 0, 0},
    {-1, 1}, {-1, 0}, {-1,-1}, { 0,-1},
    { 1,-1}, { 1, 0}, { 1, 1}, { 0, 1}
};

const double T[] = {
    4./9,
    1./36, 1./9, 1./36, 1./9,
    1./36, 1./9, 1./36, 1./9
};

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
static void equilibrium(double*** feq, double** rho, double*** u)
{
    double usqr[NX][NY];
    
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            usqr[x][y] = 3./2 * ( SQUARE(u[0][x][y]) + SQUARE(u[1][x][y]) );
        }
    }
    
    double cu[NX][NY];
    
    for (int f = 0; f < 9; f++) {
        for (int x = 0; x < NX; x++) {
            for (int y = 0; y < NY; y++) {
                cu[x][y] = 3 * ( V[f][0] * u[0][x][y] + V[f][1] * u[1][x][y] );
                feq[f][x][y] = rho[x][y] * T[f] * ( 1 + cu[x][y] + 0.5 * SQUARE(cu[x][y]) - usqr[x][y] );
            }
        }
    }
}

static void macroscopic(double*** fin, double** rho, double*** u)
{
    // density (rho) is the sum of the nine populations
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            rho[x][y] = 0;
            for (int f = 0; f < 9; f++) {
                rho[x][y] += fin[f][x][y];
            }
        }
    }
    
    {
        double zero = 0;
        array_set3(2, NX, NY, sizeof(double), u, &zero);
    }
    
    for (int f = 0; f < 9; f++) {
        for (int x = 0; x < NX; x++) {
            for (int y = 0; y < NY; y++) {
                u[0][x][y] += V[f][0] * fin[f][x][y];
                u[1][x][y] += V[f][1] * fin[f][x][y];
            }
        }
    }
    
    for (int d = 0; d < 2; d++) {
        for (int x = 0; x < NX; x++) {
            for (int y = 0; y < NY; y++) {
                u[d][x][y] /= rho[x][y];
            }
        }
    }
}

static void initCol(size_t* col, ssize_t v0)
{
    for (int f = 0, i = 0; f < 9 && i < 3; f++) {
        if (V[f][0] == v0) {
            col[i++] = f;
        }
    }
}

static void initOpp(size_t* opp)
{
    for (int f = 0; f < 9; f++) {
        for (int g = 0; g < 9; g++) {
            if (V[f][0] == -V[g][0] && V[f][1] == -V[g][1]) {
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
    ssize_t max_iter = 0;
    
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
        fprintf(stderr, "  -p : output pictures in <path> directory\n");
        fprintf(stderr, "  -t : output populations values in stdout\n");
        fprintf(stderr, "  -i : Total number of iterations\n");
        return EXIT_FAILURE;
    }
    
    const size_t A_SIZE0[3] = {2, NX, NY};
    const size_t A_SIZE1[3] = {9, NX, NY};
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

    (void) COL1;
    
    {
        double one = 1;
        array_set(2, A_SIZE2, sizeof(double), rho, &one);
    }
    
    initObstacles(obstacles);
    initVelocity(vel);
    
    // Initialization of the populations at equilibrium with the given velocity.
    equilibrium(fin, rho, vel);
    
    pgm_image* pgm = pgm_create(NX, NY);
    
    for (int time = 0; time < max_iter; time++) {
        
        // Right wall: outflow condition.
        for (int i = 0; i < 3; i++) {
            for (size_t y = 0, f = COL3[i]; y < NY; y++) {
                fin[f][NX-1][y] = fin[f][NX-2][y];
            }
        }
        
        // Compute macroscopic variables, density and velocity
        macroscopic(fin, rho, u);
        
        // Left wall: inflow condition
        for (size_t d = 0; d < 2; d++) {
            for (size_t y = 0; y < NY; y++) {
                u[d][0][y] = vel[d][0][y];
            }
        }
        
        // Calculate the density
        for (size_t y = 0; y < NY; y++) {
            double s2 = 0, s3 = 0;
            for (size_t i = 0; i < 3; i++) {
                s2 += fin[COL2[i]][0][y];
                s3 += fin[COL3[i]][0][y];
            }
            rho[0][y] = 1./(1 - u[0][0][y]) * (s2 + 2*s3);
        }
        
        // Compute equilibrium
        equilibrium(feq, rho, u);
        for (size_t i = 0; i < 3; i++) {
            for (size_t y = 0, f = COL1[i]; y < NY; y++) {
                fin[f][0][y] = feq[f][0][y] + fin[OPP[f]][0][y] - feq[OPP[f]][0][y];
            }
        }
     
        // Collision step
        for (size_t f = 0; f < 9; f++) {
            for (size_t x = 0; x < NX; x++) {
                for (size_t y = 0; y < NY; y++) {
                    fout[f][x][y] = fin[f][x][y] - OMEGA * (fin[f][x][y] - feq[f][x][y]);
                }
            }
        }
        
        // Bounce-back condition for obstacle
        for (size_t f = 0; f < 9; f++) {
            for (size_t x = 0; x < NX; x++) {
                for (size_t y = 0; y < NY; y++) {
                    if (obstacles[x][y]) {
                        fout[f][x][y] = fin[OPP[f]][x][y];
                    }
                }
            }
        }
        
        // Streaming step
        for (size_t f = 0; f < 9; f++) {
            array_roll2_to(NX, NY, sizeof(double), fout[f], fin[f], V[f]);
        }
        
        // Visualization of the velocity.
        if (time % 100 == 0 && out == OUT_IMG) {
            double vel[NX][NY];
            for (size_t x = 0; x < NX; x++) {
                for (size_t y = 0; y < NY; y++) {
                    vel[x][y] = sqrt( SQUARE(u[0][x][y]) + SQUARE(u[1][x][y]) );
                    int color =  fmin(255, 255 * vel[x][y] * 10);
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
