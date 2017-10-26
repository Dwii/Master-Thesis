/*!
 * \file    lbmSimple.c
 * \brief   Test lbmcuda library.
 * \author  Adrien Python
 * \date    06.10.2017
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <timing.h>
#include <pgm.h>
#include <math.h>
#include <stdbool.h>
#include <libgen.h>
#include <lbmcuda.h>

#define RE       220.0       // Reynolds number
#define ULB      0.04        // Velocity in lattice units
#define NULB     ((ULB) * 10 / (RE))   // Viscoscity in lattice units
#define OMEGA    ((double)1. / (3*(NULB)+0.5))  // Relaxation parameter

#define SQUARE(n) ((n)*(n))
#define H_EQUILIBRIUM(rho, t, cu, usqr) ((rho) * (t) * ( 1 + (cu) + 0.5 * SQUARE(cu) - (usqr) ))

static void equilibrium(lbm_lattices* f, int index, double rho, double u0, double u1, double u2)
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

void lbm_lattices_at_index(lbm_lattice* lattice, lbm_lattices* lattices, int x, int y, int z, size_t nx, size_t ny, size_t nz)
{
    int gi = IDX(x,y,z,nx,ny,nz);
    lattice->ne = lattices->ne[gi];
    lattice->e  = lattices->e [gi];
    lattice->se = lattices->se[gi];
    lattice->n  = lattices->n [gi];
    lattice->c  = lattices->c [gi];
    lattice->s  = lattices->s [gi];
    lattice->nw = lattices->nw[gi];
    lattice->w  = lattices->w [gi];
    lattice->sw = lattices->sw[gi];
    lattice->te = lattices->te[gi];
    lattice->tn = lattices->tn[gi];
    lattice->tc = lattices->tc[gi];
    lattice->ts = lattices->ts[gi];
    lattice->tw = lattices->tw[gi];
    lattice->be = lattices->be[gi];
    lattice->bn = lattices->bn[gi];
    lattice->bc = lattices->bc[gi];
    lattice->bs = lattices->bs[gi];
    lattice->bw = lattices->bw[gi];
}

void lbm_u_at_index(double* u0, double* u1, double* u2, lbm_u* u, int x, int y, int z, size_t nx, size_t ny, size_t nz)
{
    int gi = IDX(x,y,z,nx,ny,nz);
    *u0 = u->u0[gi];
    *u1 = u->u1[gi];
    *u2 = u->u2[gi];
}


void lbm_simulation_init(lbm_simulation* lbm_sim, size_t width, size_t height, size_t depth)
{
    size_t nl = width*height*depth;
    lbm_lattices* fin = lbm_lattices_create(nl);

    // Initialization of the populations at equilibrium with the given velocity.
    for (size_t z = 0; z < depth; z++) {
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                double rho = x == width/2 && y == height/2 && z == depth/2 ? 2.0 : 1.0;
                equilibrium(fin, IDX(x,y,z,width,height,depth), rho, 0, 0, 0);
            }
        }
    }
    
    lbm_lattices_write(lbm_sim, fin);
    
    lbm_lattices_destroy(fin);
}

typedef enum { OUT_NONE, OUT_FIN, OUT_FINP, OUT_IMG } out_mode;

static void output_lattices(char* filename, size_t width, size_t height, size_t depth, lbm_lattices* lattices)
{
    FILE* file = fopen(filename, "w");
    for (size_t x = 0; x < width; x++) {
        for (size_t y = 0; y < height; y++) {
            for (size_t z = 0; z < depth; z++) {
                lbm_lattice lattice;
                lbm_lattices_at_index(&lattice, lattices, x, y, z, width, height, depth);
                fprintf(file, "%64.60f\n", lattice.ne);
                fprintf(file, "%64.60f\n", lattice.e );
                fprintf(file, "%64.60f\n", lattice.se);
                fprintf(file, "%64.60f\n", lattice.n );
                fprintf(file, "%64.60f\n", lattice.c );
                fprintf(file, "%64.60f\n", lattice.s );
                fprintf(file, "%64.60f\n", lattice.nw);
                fprintf(file, "%64.60f\n", lattice.w );
                fprintf(file, "%64.60f\n", lattice.sw);
                fprintf(file, "%64.60f\n", lattice.te);
                fprintf(file, "%64.60f\n", lattice.tn);
                fprintf(file, "%64.60f\n", lattice.tc);
                fprintf(file, "%64.60f\n", lattice.ts);
                fprintf(file, "%64.60f\n", lattice.tw);
                fprintf(file, "%64.60f\n", lattice.be);
                fprintf(file, "%64.60f\n", lattice.bn);
                fprintf(file, "%64.60f\n", lattice.bc);
                fprintf(file, "%64.60f\n", lattice.bs);
                fprintf(file, "%64.60f\n", lattice.bw);
            }
        }
    }
    fclose(file);
}

static void output_palabos_lattices(char* filename, size_t width, size_t height, size_t depth, lbm_lattices* lattices)
{
    FILE* file = fopen(filename, "w");
    for (size_t x = 0; x < width; x++) {
        for (size_t y = 0; y < height; y++) {
            for (size_t z = 0; z < depth; z++) {
                lbm_lattice lattice;
                lbm_lattices_at_index(&lattice, lattices, x, y, z, width, height, depth);
                fprintf(file, "%.60f ", lattice.c  - 1./3 );
                fprintf(file, "%.60f ", lattice.w  - 1./18);
                fprintf(file, "%.60f ", lattice.s  - 1./18);
                fprintf(file, "%.60f ", lattice.bc - 1./18);
                fprintf(file, "%.60f ", lattice.sw - 1./36);
                fprintf(file, "%.60f ", lattice.nw - 1./36);
                fprintf(file, "%.60f ", lattice.bw - 1./36);
                fprintf(file, "%.60f ", lattice.tw - 1./36);
                fprintf(file, "%.60f ", lattice.bs - 1./36);
                fprintf(file, "%.60f ", lattice.ts - 1./36);
                fprintf(file, "%.60f ", lattice.e  - 1./18);
                fprintf(file, "%.60f ", lattice.n  - 1./18);
                fprintf(file, "%.60f ", lattice.tc - 1./18);
                fprintf(file, "%.60f ", lattice.ne - 1./36);
                fprintf(file, "%.60f ", lattice.se - 1./36);
                fprintf(file, "%.60f ", lattice.te - 1./36);
                fprintf(file, "%.60f ", lattice.be - 1./36);
                fprintf(file, "%.60f ", lattice.tn - 1./36);
                fprintf(file, "%.60f ", lattice.bn - 1./36);
            }
        }
    }
    fclose(file);
}

static void output_image(char* filename, size_t width, size_t height, size_t depth, size_t z, lbm_u* u)
{
    pgm_image* pgm = pgm_create(width, height);
    
    for (size_t x = 0; x < width; x++) {
        for (size_t y = 0; y < height; y++) {
            double u0, u1, u2;
            lbm_u_at_index(&u0, &u1, &u2, u, x, y, z, width, height, depth);
            double vel = sqrt( 100* u0*u0 + 100* u1*u1 + 100* u2*u2);
            int color =  255 * fmin(vel * 100, 1.0);
            pgm_set_pixel(pgm, x, y, color);
        }
    }
    
    pgm_write(pgm, filename);
    pgm_destroy(pgm);
}

float get_lups(long lattices, long iterations, long ns_time_diff)
{
    return lattices * iterations * 1000000000.0f / ns_time_diff;
}

int main(int argc, char * const argv[])
{
    // Init options to default values
    const char* out_path = ".";
    const char* out_pref = "lbm";
    out_mode out = OUT_NONE;
    ssize_t max_iter = 0;
    size_t out_interval = 0;
    bool print_lups = false;
    bool print_avg_lups = false;
    size_t width, height, depth;
    width = height = depth = 0;
    
    // Read arguments
    while (optind < argc) {
        switch (getopt(argc, argv, "pfFi:I:o:O:lLx:y:z:")) {
            case 'p': { out = OUT_IMG; break; }
            case 'f': { out = OUT_FIN; break; }
            case 'F': { out = OUT_FINP; break; }
            case 'i': { max_iter = strtol(optarg, NULL, 10); break; }
            case 'I': { out_interval = strtol(optarg, NULL, 10); break; }
            case 'o': { out_path = optarg; break; }
            case 'O': { out_pref = optarg; break; }
            case 'l': { print_lups = true; break; }
            case 'L': { print_avg_lups = true; break; }
            case 'x': { width  = strtol(optarg, NULL, 10); break; }
            case 'y': { height = strtol(optarg, NULL, 10); break; }
            case 'z': { depth  = strtol(optarg, NULL, 10); break; }
            default : { goto usage; }
        }
    }
    
    // check that execution mode is set (output images or fin values)
    if (max_iter < 1 || width <= 0 || height <= 0 || depth <= 0) {
    usage:
        fprintf(stderr, "usage: %s (-p | -f | -F) -i <iter> [-I <out_interval>] [-o <out_dir>] [-O <out_prefix>] [-l] [-L] -x <nx> -y <ny> -z <nz>\n", basename((char*)argv[0]));
        fprintf(stderr, "  -p : output pictures\n");
        fprintf(stderr, "  -f : output populations\n");
        fprintf(stderr, "  -F : output populations formated like Palabos\n");
        fprintf(stderr, "  -i : number of iterations\n");
        fprintf(stderr, "  -I : output interval; (0 if only the last iteration output is required)\n");
        fprintf(stderr, "  -o : output file directory\n");
        fprintf(stderr, "  -O : output filename prefix\n");
        fprintf(stderr, "  -l : print lups at each output interval\n");
        fprintf(stderr, "  -L : print average lups at the end\n");
        fprintf(stderr, "  -x : width\n");
        fprintf(stderr, "  -y : height\n");
        fprintf(stderr, "  -z : depth\n");
        return EXIT_FAILURE;
    }
    
    if (out == OUT_NONE) {
        fprintf(stderr, "No output mode specified.\n");
    }
    
    lbm_simulation* lbm_sim = lbm_simulation_create(width, height, depth, OMEGA);
    lbm_simulation_init(lbm_sim, width, height, depth);

    lbm_u* u = lbm_u_create(width, height, depth);
    lbm_lattices* fin = lbm_lattices_create(width * height * depth);
    
#define READ_WRITE
#define BOUNDARY
#define USE_KERNEL_COPY

#if defined(READ_WRITE) && defined(BOUNDARY)
    lbm_box_3d subdomain0w = {       0, width-1,        0,        0,        0, depth-1 };
    lbm_box_3d subdomain1w = {       0, width-1, height-1, height-1,        0, depth-1 };
    lbm_box_3d subdomain2w = {       0, width-1,        1, height-2,        0,       0 };
    lbm_box_3d subdomain3w = {       0, width-1,        1, height-2,  depth-1, depth-1 };
    lbm_box_3d subdomain4w = {       0,       0,        1, height-2,        1, depth-2 };
    lbm_box_3d subdomain5w = { width-1, width-1,        1, height-2,        1, depth-2 };

    lbm_box_3d subdomain0r = {       1, width-2,        1,        1,        1, depth-2 };
    lbm_box_3d subdomain1r = {       1, width-2, height-2, height-2,        1, depth-2 };
    lbm_box_3d subdomain2r = {       1, width-2,        2, height-3,        1,       1 };
    lbm_box_3d subdomain3r = {       1, width-2,        2, height-3,  depth-2, depth-2 };
    lbm_box_3d subdomain4r = {       1,       1,        2, height-3,        2, depth-3 };
    lbm_box_3d subdomain5r = { width-2, width-2,        2, height-3,        2, depth-3 };

#ifdef USE_KERNEL_COPY
    double* data = malloc(width*height*depth*sizeof(double)*19);
#endif

#endif

    long time_diff, total_time_diff = 0;
    start_time_t start_time;
    timing_start(&start_time);

    for (int iter = 1; iter <= max_iter; iter++) {
        
#ifdef READ_WRITE
#ifdef BOUNDARY
#ifdef USE_KERNEL_COPY
        lbm_write_palabos_subdomain(lbm_sim, data, subdomain0w);
        lbm_write_palabos_subdomain(lbm_sim, data, subdomain1w);
        lbm_write_palabos_subdomain(lbm_sim, data, subdomain2w);
        lbm_write_palabos_subdomain(lbm_sim, data, subdomain3w);
        lbm_write_palabos_subdomain(lbm_sim, data, subdomain4w);
        lbm_write_palabos_subdomain(lbm_sim, data, subdomain5w);
#else
        lbm_lattices_write_subdomain(lbm_sim, fin, subdomain0w);
        lbm_lattices_write_subdomain(lbm_sim, fin, subdomain1w);
        lbm_lattices_write_subdomain(lbm_sim, fin, subdomain2w);
        lbm_lattices_write_subdomain(lbm_sim, fin, subdomain3w);
        lbm_lattices_write_subdomain(lbm_sim, fin, subdomain4w);
        lbm_lattices_write_subdomain(lbm_sim, fin, subdomain5w);
#endif
#else
    lbm_lattices_write(lbm_sim, fin);
#endif
#endif
        
        lbm_simulation_update(lbm_sim);
        
#ifdef READ_WRITE
#ifdef BOUNDARY
#ifdef USE_KERNEL_COPY
        lbm_read_palabos_subdomain(lbm_sim, data, subdomain0r);
        lbm_read_palabos_subdomain(lbm_sim, data, subdomain1r);
        lbm_read_palabos_subdomain(lbm_sim, data, subdomain2r);
        lbm_read_palabos_subdomain(lbm_sim, data, subdomain3r);
        lbm_read_palabos_subdomain(lbm_sim, data, subdomain4r);
        lbm_read_palabos_subdomain(lbm_sim, data, subdomain5r);
#else
        lbm_lattices_read_subdomain(lbm_sim, fin, subdomain0r);
        lbm_lattices_read_subdomain(lbm_sim, fin, subdomain1r);
        lbm_lattices_read_subdomain(lbm_sim, fin, subdomain2r);
        lbm_lattices_read_subdomain(lbm_sim, fin, subdomain3r);
        lbm_lattices_read_subdomain(lbm_sim, fin, subdomain4r);
        lbm_lattices_read_subdomain(lbm_sim, fin, subdomain5r);
#endif
#else
        lbm_lattices_read(lbm_sim, fin);
#endif
#endif
        if ( (!out_interval && iter == max_iter) || (out_interval && iter % out_interval == 0) ) {
            
            time_diff = timing_stop(&start_time);
            total_time_diff += time_diff;
            if ( print_lups ) {
                size_t iter_diff = out_interval? out_interval : (size_t)max_iter;
                printf("lups: %.2f\n", get_lups(width*height*depth, iter_diff, time_diff));
                fflush(stdout);
            }
            
            char* filename;
            
            if ( out == OUT_IMG ) {
                lbm_u_read(lbm_sim, u);
                
                if ( asprintf(&filename, "%s/%s%d.pgm", out_path, out_pref, iter) != -1 ) {
                    output_image(filename, width, height, depth, depth/2, u);
                    free(filename);
                }
            }
            
            if (out == OUT_FIN) {
                
                lbm_lattices_read(lbm_sim, fin);
                
                if ( asprintf(&filename, "%s/%s%d.out", out_path, out_pref, iter) != -1) {
                    output_lattices(filename, width, height, depth, fin);
                    free(filename);
                }
            }
            
            if (out == OUT_FINP) {
                
                lbm_lattices_read(lbm_sim, fin);
                
                if ( asprintf(&filename, "%s/%s%06d.dat", out_path, out_pref, iter) != -1) {
                    output_palabos_lattices(filename, width, height, depth, fin);
                    free(filename);
                }
            }
            
            timing_start(&start_time);
        }
    }
    
    if ( print_avg_lups ) {
        printf("average lups: %.2f\n", get_lups(width*height*depth, max_iter, total_time_diff));
    }

#if defined(READ_WRITE) && defined(BOUNDARY) && defined(USE_KERNEL_COPY)
    free(data);
#endif

    lbm_u_destroy(u);
    lbm_lattices_destroy(fin);
    lbm_simulation_destroy(lbm_sim);
    
    return EXIT_SUCCESS;
}
