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

#define MAX_ITER 2000000     // Total number of time iterations
#define RE       220.0       // Reynolds number
#define NX       420         // Numer of lattice nodes (width)
#define NY       180         // Numer of lattice nodes (height)
#define LY       ((NY) - 1)  // Height of the domain in lattice units
#define CX       ((NX) / 4)  // X coordinates of the cylinder
#define CY       ((NY) / 2)  // Y coordinates of the cylinder
#define R        ((NY) / 9)  // Cylinder radius
#define ULB      0.04        // Velocity in lattice units
#define NULB     ((ULB) * (R) / (RE))   // Viscoscity in lattice units
#define OMEGA    (1. / (3*(NULB)+0.5))  // Relaxation parameter

#define SQUARE(a) ((a)*(a))

const ssize_t V[][2] = { {1,1}, {1,0}, {1,-1}, {0,1}, {0,0}, {0,-1}, {-1,1}, {-1,0}, {-1,-1} };
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

int main(int argc, const char * argv[])
{

    // Read arguments (hopfuly the images path)
    if (argc != 2) {
        fprintf(stderr, "usage: %s <img_path>\n", basename((char*)argv[0]));
        return EXIT_FAILURE;
    }

    const char* img_path = argv[1];
    
    
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
    
    const int COL1[] = { 0, 1, 2};
    const int COL2[] = { 3, 4, 5};
    const int COL3[] = { 6, 7, 8};

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
    
    for (int time = 0; time < MAX_ITER; time++) {
        
        // Right wall: outflow condition.
        for (int i = 0; i < 3; i++) {
            for (int y = 0, f = COL3[i]; y < NY; y++) {
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
        for (size_t y = 0; y < NY; y++) {
            fin[0][0][y] = feq[0][0][y] + fin[8][0][y] - feq[8][0][y];
            fin[1][0][y] = feq[1][0][y] + fin[7][0][y] - feq[7][0][y];
            fin[2][0][y] = feq[2][0][y] + fin[6][0][y] - feq[6][0][y];
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
                        fout[f][x][y] = fin[8-f][x][y];
                    }
                }
            }
        }
        
        // Streaming step
        for (size_t f = 0; f < 9; f++) {
            array_roll2_to(NX, NY, sizeof(double), fout[f], fin[f], V[f]);
        }
        
        // Visualization of the velocity.
        if (time % 100 == 0) {
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
