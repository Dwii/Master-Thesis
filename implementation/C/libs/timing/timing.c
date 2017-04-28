/*!
 * \file    timing.c
 * \brief   Timing measurment and computation tools.
 * \author  Adrien Python
 * \version 1.0
 * \date    28.04.2017
 */

#include "timing.h"

/**
 * \brief Record the start time.
 * \param   start  recored start time
 * \return  void
 */
void timing_start(start_time_t *start)
{
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

/**
 * \brief Stop the timer and computes the elapsed time.
 * \param   start  previously recored start time
 * \return  elapsed time in nanoseconds
 */
long timing_stop(start_time_t *start)
{
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
