/*!
 * \file    lbmmain.c
 * \brief   Common LBM 3D command line interface (main function).
 * \author  Adrien Python
 * \date    10.05.2017
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <libgen.h>
#include <pgm.h>
#include <timing.h>
#include "lbm.h"

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

    lbm_simulation* lbm_sim = lbm_simulation_create(width, height, depth);

    long time_diff, total_time_diff = 0;
    start_time_t start_time;
    timing_start(&start_time);

    lbm_u* u = lbm_u_create(width, height, depth);
    lbm_lattices* fin = lbm_lattices_create(width, height, depth);

    for (int iter = 1; iter <= max_iter; iter++) {

        lbm_simulation_update(lbm_sim);

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
                lbm_u_read(lbm_sim, u, width, height, depth);

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

    lbm_u_destroy(u);
    lbm_lattices_destroy(fin);
    lbm_simulation_destroy(lbm_sim);
   
    return EXIT_SUCCESS;
}
