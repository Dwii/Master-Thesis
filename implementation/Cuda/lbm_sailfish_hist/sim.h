#ifndef __SIM_H_
#define __SIM_H_ 1

#define GEO_FLUID 0
#define GEO_WALL 1
#define GEO_INFLOW 2

#define LAT_H 180
#define LAT_W 420
#define BLOCK_SIZE 64

#ifdef USE_FLOATS
#define double float
#define __dadd_rn __fadd_rn
#define __dmul_rn __fmul_rn
#else
#define float double
#define __fadd_rn __dadd_rn
#define __fmul_rn __dmul_rn
#endif

struct Dist {
	float *fC, *fE, *fW, *fS, *fN, *fSE, *fSW, *fNE, *fNW;
};

struct SimState {
	int *map, *dmap;

	// macroscopic quantities on the video card
	float *dvx, *dvy, *drho;

	// macroscopic quantities in RAM
	float *vx, *vy, *rho;

	float *lat[9];
	Dist d1, d2;
};

void SimInit(struct SimState *state);
void SimCleanup(struct SimState *state);
void SimUpdate(int iter, struct SimState state);
void SimUpdateMap(struct SimState state);

#endif  /* __SIM_H_ */
