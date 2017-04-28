/*!
 * \file    timing.h
 * \brief   Timing measurment and computation tools.
 * \author  Adrien Python
 * \version 1.0
 * \date    28.04.2017
 */

#ifndef TIMING_H
#define TIMING_H

#ifdef __APPLE__
#include <stdint.h>
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
#include <time.h>
typedef struct timespec start_time_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void timing_start(start_time_t *start);
long timing_stop(start_time_t *start);
    
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* TIMING_H */
