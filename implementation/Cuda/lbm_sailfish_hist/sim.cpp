#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <pgm.h>
#include <libgen.h>
#include <timing.h>

#include "sim.h"

#define CX       ((LAT_W) / 4)  // X coordinates of the cylinder
#define CY       ((LAT_H) / 2)  // Y coordinates of the cylinder
#define R        ((LAT_H) / 9)  // Cylinder radius

#define SQUARE(a) ((a)*(a))

void output_image(char* filename, struct SimState *state)
{
    pgm_image* pgm = pgm_create(LAT_W, LAT_H);

    int x, y, i = 0;
    
    for (y = 0; y < LAT_H; y++) {
        for (x = 0; x < LAT_W; x++) {
            if (state->map[i] == GEO_WALL) {
                pgm_set_pixel(pgm, x, y, 0);
            } else {
                int color = fmin(255, sqrtf(state->vx[i]*state->vx[i] + state->vy[i]*state->vy[i]) / 0.1f * 255.0f);
                pgm_set_pixel(pgm, x, LAT_H-y-1, color);
            }
            i++;
        }
    }

    pgm_write(pgm, filename);
    pgm_destroy(pgm);
}

static void initObstacles(struct SimState* state)
{
    for (int x = 0; x < LAT_W; x++) {
        for (int y = 0; y < LAT_H; y++) {
            if (SQUARE(x-CX) + SQUARE(y-CY) < SQUARE(R)) {
                state->map[y * LAT_W + x] = GEO_WALL;
            }
        }
    }
    SimUpdateMap(*state);
}

float get_lups(int lattices, int iterations, long ns_time_diff)
{
    return lattices * iterations * 1000000000.0f / ns_time_diff;
}

int main(int argc, char **argv)
{

    char* out_path = NULL;
    char* out_pre  = NULL;
    ssize_t max_iter;
    if (argc >= 2)
    {
        max_iter = strtol(argv[1], NULL, 10);
        if (argc == 4) {
            out_path = argv[2];
            out_pre  = argv[3];
        }
    } else {
        fprintf(stderr, "usage: %s <iter> [<out_dir> <out_prefix>]\n", basename(argv[0]));
        return EXIT_FAILURE;
    }


	struct SimState state;

	SimInit(&state);
    initObstacles(&state);
    
	int last_x, last_y;

    long time_diff, total_time_diff = 0;
	start_time_t start_time;
    timing_start(&start_time);

    for (int iter = 0; iter < max_iter; iter++) {
		SimUpdate(iter, state);
        if (iter % 100 == 0) {
            total_time_diff += time_diff = timing_stop(&start_time);
            
            printf("lups = %.4f\n", get_lups(LAT_H * LAT_W, 100, time_diff));

            if (out_path && out_pre) {
                char* filename;
                if ( asprintf(&filename, "%s/%s%d.pgm", out_path, out_pre, iter) != -1) {
                    output_image(filename, &state);
                    free(filename);
                }
            }

            timing_start(&start_time);
        }
	}

    printf("average lups: %.2f\n", get_lups(LAT_H * LAT_W, max_iter, total_time_diff));

	SimCleanup(&state);

    return 0;
}
