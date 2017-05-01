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

typedef enum { OUT_NONE, OUT_FIN, OUT_IMG } out_mode;

typedef struct {
    double u[NX][NY][2];
    double feq[NX][NY][9];
    double fin[NX][NY][9];
    double fout[NX][NY][9];
    double rho[NX][NY];
    double vel[NX][NY][2];
} lbm_vars;

typedef struct {
    bool obstacles[NX][NY];
    size_t col[3][3];
    size_t opp[9];
} lbm_consts;


const ssize_t V[][9] = {
    { 1, 1, 1, 0, 0, 0,-1,-1,-1 },
    { 1, 0,-1, 1, 0,-1, 1, 0,-1 }
};
const double T[] = { 1./36, 1./9, 1./36, 1./9, 4./9, 1./9, 1./36, 1./9, 1./36 };

/**
 * Setup: cylindrical obstacle and velocity inlet with perturbation
 * Creation of a mask with boolean values, defining the shape of the obstacle.
 */
static void initObstacles(lbm_consts* consts)
{
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            consts->obstacles[x][y] = SQUARE(x-CX) + SQUARE(y-CY) < SQUARE(R);
        }
    }
}

/**
 * Initial velocity profile: almost zero, with a slight perturbation to trigger
 * the instability.
 */
static void initVelocity(lbm_vars* vars)
{
    for (int d = 0; d < 2; d++) {
        for (int x = 0; x < NX; x++) {
            for (int y = 0; y < NY; y++) {
                vars->vel[x][y][d] = (1-d) * ULB * (1 + 0.0001 * sin( y / (double)LY * 2 * M_PI) );
            }
        }
    }
}


static void initRho(lbm_vars* vars)
{
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            vars->rho[x][y] = 1.0;
        }
    }
}

/**
 * Equilibrium distribution function.
 */
static void equilibrium(double* feq, double rho, double* u)
{
    double usqr = 3./2 * ( SQUARE(u[0]) + SQUARE(u[1]) );

    for (int f = 0; f < 9; f++) {
        double cu = 3 * ( V[0][f] * u[0] + V[1][f] * u[1] );
        feq[f] = rho * T[f] * ( 1 + cu + 0.5 * SQUARE(cu) - usqr );
    }
}

static void macroscopic(double* fin, double* rho, double* u)
{
    
    *rho = u[0] = u[1] = 0;

    for (int f = 0; f < 9; f++) {
        *rho += fin[f];

        u[0] += V[0][f] * fin[f];
        u[1] += V[1][f] * fin[f];
    }
    
    u[0] /= *rho;
    u[1] /= *rho;
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


static inline void output_variables(char* filename, lbm_vars* vars)
{
    FILE* file = fopen(filename, "w");
    for (size_t x = 0; x < NX; x++) {
        for (size_t y = 0; y < NY; y++) {
            for (size_t f = 0; f < 9; ++f) {
                fprintf(file, "%64.60f\n", vars->fin[x][y][f]);
            }
        }
    }
    fclose(file);
}

void output_image(char* filename, lbm_vars* vars) 
{
    pgm_image* pgm = pgm_create(NX, NY);
    
    for (size_t x = 0; x < NX; x++) {
        for (size_t y = 0; y < NY; y++) {
            double vel = sqrt( SQUARE(vars->u[x][y][0]) + SQUARE(vars->u[x][y][1]) );
            int color =  255 * vel * 10;
            pgm_set_pixel(pgm, x, y, color);
        }
    }

    pgm_write(pgm, filename);
    pgm_destroy(pgm);
}

int main(int argc, char * const argv[])
{
    // Init options to default values
    const char* out_path = ".";
    const char* out_pref = "lbm";
    out_mode out = OUT_NONE;
    ssize_t max_iter = 0;
    size_t out_interval = 0;

    // Read arguments
    while (optind < argc) {
        switch (getopt(argc, argv, "pfi:I:o:O:")) {
            case 'p': { out = OUT_IMG; break; }
            case 'f': { out = OUT_FIN; break; }
            case 'i': { max_iter = strtol(optarg, NULL, 10); break; }
            case 'I': { out_interval = strtol(optarg, NULL, 10); break; }
            case 'o': { out_path = optarg; break; }
            case 'O': { out_pref = optarg; break; }
            default : { goto usage; }
        }
    }
    
    // check that execution mode is set (output images or fin values)
    if (max_iter < 1) {
    usage:
        fprintf(stderr, "usage: %s (-p | -f) -i <iter> [-I <out_interval>] [-o <out_dir>] [-O <out_prefix>]\n", basename((char*)argv[0]));
        fprintf(stderr, "  -p : output pictures\n");
        fprintf(stderr, "  -f : output populations\n");
        fprintf(stderr, "  -i : number of iterations\n");
        fprintf(stderr, "  -I : output interval; (0 if only the last iteration output in required)\n");
        fprintf(stderr, "  -o : output file directory\n");
        fprintf(stderr, "  -O : output filename prefix\n");
        return EXIT_FAILURE;
    }

    if (out == OUT_NONE) {
        fprintf(stderr, "No output mode specified.\n");
    }

    lbm_consts* consts = (lbm_consts*)malloc(sizeof(lbm_consts));
    
    initObstacles(consts);
    initCol(consts->col[0],  1);
    initCol(consts->col[1],  0);
    initCol(consts->col[2], -1);
    initOpp(consts->opp);
    
    lbm_vars *vars = (lbm_vars*)malloc(sizeof(lbm_vars));
    initVelocity(vars);
    initRho(vars);
    
    // Initialization of the populations at equilibrium with the given velocity.
    
    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            equilibrium(vars->fin[x][y], vars->rho[x][y], vars->vel[x][y]);
        }
    }

    pgm_image* pgm = pgm_create(NX, NY);
    
    for (int iter = 1; iter <= max_iter; iter++) {
        
        // Right wall: outflow condition.
        for (int i = 0; i < 3; i++) {
            for (int y = 0, f = consts->col[2][i]; y < NY; y++) {
                vars->fin[NX-1][y][f] = vars->fin[NX-2][y][f];
            }
        }
        
        for (size_t y = 0; y < NY; y++) {
            // Compute macroscopic variables, density and velocity
            for (size_t x = 0; x < NX; x++) {
                macroscopic(vars->fin[x][y], &vars->rho[x][y], vars->u[x][y]);
            }
            
            // Left wall: inflow condition
            for (size_t d = 0; d < 2; d++) {
                vars->u[0][y][d] = vars->vel[0][y][d];
            }
            
            // Calculate the density
            double s2 = 0, s3 = 0;
            for (size_t i = 0; i < 3; i++) {
                s2 += vars->fin[0][y][consts->col[1][i]];
                s3 += vars->fin[0][y][consts->col[2][i]];
            }
            vars->rho[0][y] = 1./(1 - vars->u[0][y][0]) * (s2 + 2*s3);
            
            // Compute equilibrium
            for (size_t x = 0; x < NX; x++) {
                equilibrium(vars->feq[x][y], vars->rho[x][y], vars->u[x][y]);
            }
            
            for (size_t i = 0, f = consts->col[0][i]; i < 3; f = consts->col[0][++i]) {
                vars->fin[0][y][f] = vars->feq[0][y][f] + vars->fin[0][y][consts->opp[f]] - vars->feq[0][y][consts->opp[f]];
            }
            
            for (size_t x = 0; x < NX; x++) {
                for (size_t f = 0; f < 9; f++) {
                    if (consts->obstacles[x][y]) {
                        // Bounce-back condition for obstacle
                        vars->fout[x][y][f] = vars->fin[x][y][consts->opp[f]];
                    } else {
                        // Collision step
                        vars->fout[x][y][f] = vars->fin[x][y][f] - OMEGA * (vars->fin[x][y][f] - vars->feq[x][y][f]);
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
                    vars->fin[x_dst][y_dst][f] = vars->fout[x][y][f];
                }
            }
        }
        
        if ( (!out_interval && iter == max_iter) || (out_interval && iter % out_interval == 0) ) {

            char* filename;

            if ( out == OUT_IMG ) {
                if ( asprintf(&filename, "%s/%s%d.pgm", out_path, out_pref, iter) != -1 ) {
                    output_image(filename, vars);
                    free(filename);
                }
            }

            if (out == OUT_FIN) {
                if ( asprintf(&filename, "%s/%s%d.out", out_path, out_pref, iter) != -1 ) {
                    output_variables(filename, vars);
                    free(filename);
                }
            }
        }
    }
    
    
    pgm_destroy(pgm);
    free(consts);
    free(vars);
    
    return EXIT_SUCCESS;
}
