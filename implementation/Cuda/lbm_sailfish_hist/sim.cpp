#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <pgm.h>

#include <libgen.h>

#include "sim.h"
#include "vis.h"

#define CX       ((LAT_W) / 4)  // X coordinates of the cylinder
#define CY       ((LAT_H) / 2)  // Y coordinates of the cylinder
#define R        ((LAT_H) / 9)  // Cylinder radius

#define SQUARE(a) ((a)*(a))


#ifdef __APPLE__
typedef struct start_time_t {
    double base;
    uint64_t time;
} start_time_t;

// Thanks to Jens Gustedt
// http://stackoverflow.com/questions/5167269/clock-gettime-alternative-in-mac-os-x
#include <mach/mach_time.h>
#include <libc.h>
#define ORWL_NANO (+1.0E-9)
#define ORWL_GIGA UINT64_C(1000000000)

#else
typedef struct timespec start_time_t;
#endif


/// @brief Starts the high resolution timer.
/// @param start Where to store the start time.
void timing_start(start_time_t *start) {
#ifdef __APPLE__
    mach_timebase_info_data_t tb;
    memset(&tb, 0, sizeof(tb));
    mach_timebase_info(&tb);
    start->base = tb.numer;
    start->base /= tb.denom;
    start->time = mach_absolute_time();
#else
    clock_gettime(CLOCK_MONOTONIC, start);
#endif
}

/// @brief Stops the high resolution timer and computes the elapsed time.
/// @param start Start time.
/// @return Elapsed time in nanoseconds.
long timing_stop(start_time_t *start) {
    
    long diff;
#ifdef __APPLE__
    diff = (mach_absolute_time() - start->time) * start->base;
    struct timespec t_diff;
    t_diff.tv_sec = diff * ORWL_NANO;
    t_diff.tv_nsec = diff - (t_diff.tv_sec * ORWL_GIGA);
    diff = (t_diff.tv_sec) * 1000000000 + (t_diff.tv_nsec);
#else
    struct timespec finish;
    clock_gettime(CLOCK_MONOTONIC, &finish);
    diff = (finish.tv_sec-start->tv_sec) * 1000000000 + (finish.tv_nsec-start->tv_nsec);
#endif
    return diff;
}

void output_image(char* filename, struct SimState *state)
{
    pgm_image* pgm = pgm_create(LAT_W, LAT_H);

    int x, y, i = 0;
    
    for (y = 0; y < LAT_H; y++) {
        for (x = 0; x < LAT_W; x++) {
            int color = sqrtf(state->vx[i]*state->vx[i] + state->vy[i]*state->vy[i]) / 0.1f * 255.0f;
            if (state->map[i] == GEO_WALL) {
                pgm_set_pixel(pgm, x, y, 0);
            } else {
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

	start_time_t start_time;

    timing_start(&start_time);

    for (int iter = 0; iter < max_iter; iter++) {
		SimUpdate(iter, state);
        if (iter % 100 == 0) {
            long timediff = timing_stop(&start_time);
            
            float lups = LAT_H * LAT_W * 100 * 1000000000.0f / timediff;
            printf("lups = %.4f\n", lups);

            if (out_path && out_pre) {
                char* filename;
                asprintf(&filename, "%s/%s%d.pgm", out_path, out_pre, iter);
                output_image(filename, &state);
                free(filename);
            }

            timing_start(&start_time);
        }
	}

	SimCleanup(&state);

    return 0;
}
