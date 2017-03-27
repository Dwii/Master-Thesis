__constant__ float tau0 = 3.5f;    // relaxation time
__constant__ float visc = 1.0f;   // viscosity

#define BLOCK_SIZE 64
#define DIST_SIZE 74304u
#define OPTION_SAVE_MACRO_FIELDS 1
#define OPTION_BULK 2

#define INVALID_NODE 0xffffffff

#define DT 1.0f

#include <stdio.h>

// Additional geometry parameters (velocities, pressures, etc)
__constant__ float node_params[2] = {
    1.00000000000000005551e-01f,
    0.00000000000000000000e+00f,
};

// OpenCL compatibility code.
__device__ int inline get_local_size(int i)
{
    if (i == 0) {
        return blockDim.x;
    } else {
        return blockDim.y;
    }
}

__device__ int inline get_global_size(int i)
{
    if (i == 0) {
        return blockDim.x * gridDim.x;
    } else {
        return blockDim.y * gridDim.y;
    }
}

__device__ int inline get_group_id(int i)
{
    if (i == 0) {
        return blockIdx.x;
    } else {
        return blockIdx.y;
    }
}

__device__ int inline get_local_id(int i)
{
    if (i == 0) {
        return threadIdx.x;
    } else {
        return threadIdx.y;
    }
}

__device__ int inline get_global_id(int i)
{
    if (i == 0) {
        return threadIdx.x + blockIdx.x * blockDim.x;
    } else {
        return threadIdx.y + blockIdx.y * blockDim.y;
    }
}

typedef struct Dist {
    float fC;
    float fE;
    float fN;
    float fW;
    float fS;
    float fNE;
    float fNW;
    float fSW;
    float fSE;
} Dist;

// Functions for checking whether a node is of a given specific type.
__device__ inline bool is_NTGhost(unsigned int type) {
    return type == 4;
}
__device__ inline bool isNTRegularizedVelocity(unsigned int type) {
    return type == 3;
}
__device__ inline bool is_NTFluid(unsigned int type) {
    return type == 1;
}
__device__ inline bool isNTFullBBWall(unsigned int type) {
    return type == 2;
}

// Returns true is the node does not require any special processing
// to calculate macroscopic fields.
__device__ inline bool NTUsesStandardMacro(unsigned int type) {
    return (false || is_NTFluid(type) || isNTFullBBWall(type) );
}

// Wet nodes are nodes that undergo a standard collision procedure.
__device__ inline bool isWetNode(unsigned int type) {
    return (false || isNTRegularizedVelocity(type) || is_NTFluid(type) );
}

// Wet nodes are nodes that undergo a standard collision procedure.
__device__ inline bool isExcludedNode(unsigned int type) {
    return (false || is_NTGhost(type) );
}

__device__ inline bool isPropagationOnly(unsigned int type) {
    return (false);
}

// Internal helper, do not use directly.
__device__ inline void _storeNodeScratchSpace(unsigned int scratch_id,
                                              unsigned int num_values, float *buffer,  float *g_buffer) {
    for (int i = 0; i < num_values; i++) {
        g_buffer[i + scratch_id * num_values] = buffer[i];
        
    }
}

// Internal helper, do not use directly.
__device__ inline void _loadNodeScratchSpace(unsigned int scratch_id,
                                             unsigned int num_values,  float *g_buffer, float *buffer) {
    for (int i = 0; i < num_values; i++) {
        buffer[i] = g_buffer[i + scratch_id * num_values];
    }
}

// Reads values from node scratch space (in global memory) into a local buffer.
//
// scratch_id: scratch space ID for nodes of type 'type'
// type: node type
// g_buffer: pointer to a buffer in the global memory used for scratch
//       space
// buffer: pointer to a local buffer where the values will be saved
__device__ inline void loadNodeScratchSpace(unsigned int scratch_id, unsigned int type,  float *g_buffer, float* buffer)
{ }

// Stores values from a local buffer into the node scratch space in global memory.
//
// Arguments: see loadNodeScratchSpace
__device__ inline void storeNodeScratchSpace(unsigned int scratch_id,unsigned int type, float* buffer,  float* g_buffer)
{ }

__device__ inline unsigned int decodeNodeType(unsigned int nodetype) {
    return nodetype & 7;
}

__device__ inline unsigned int decodeNodeOrientation(unsigned int nodetype) {
    return nodetype >> 5;
}

// Returns the node's scratch ID, to be passed to (load,store)NodeScratchSpace as scratch_id.
__device__ inline unsigned int decodeNodeScratchId(unsigned int nodetype) {
    return (nodetype >> 5) & 0;
}

__device__ inline unsigned int decodeNodeParamIdx(unsigned int nodetype) {
    return (nodetype >> 3) & 3;
}

__device__ inline unsigned int getGlobalIdx(int gx, int gy) {
    return gx + 288 * gy;
}

__device__ inline void decodeGlobalIdx(unsigned int gi, int *gx, int *gy) {
    *gx = gi % 288;
    *gy = gi / 288;
}

__device__ void die(void) {
    asm("trap;");
}

__device__ void checkInvalidValues(Dist* d, int gx, int gy) {
    bool valid = true;
    if (!isfinite(d->fC)) {
        valid = false;
        printf("ERR(subdomain=0): Invalid value of fC (%f) at: (%d, %d)\n", d->fC, gx, gy);
    }
    if (!isfinite(d->fE)) {
        valid = false;
        printf("ERR(subdomain=0): Invalid value of fE (%f) at: (%d, %d)\n", d->fE, gx, gy);
    }
    if (!isfinite(d->fN)) {
        valid = false;
        printf("ERR(subdomain=0): Invalid value of fN (%f) at: (%d, %d)\n", d->fN, gx, gy);
    }
    if (!isfinite(d->fW)) {
        valid = false;
        printf("ERR(subdomain=0): Invalid value of fW (%f) at: (%d, %d)\n", d->fW, gx, gy);
    }
    if (!isfinite(d->fS)) {
        valid = false;
        printf("ERR(subdomain=0): Invalid value of fS (%f) at: (%d, %d)\n", d->fS, gx, gy);
    }
    if (!isfinite(d->fNE)) {
        valid = false;
        printf("ERR(subdomain=0): Invalid value of fNE (%f) at: (%d, %d)\n", d->fNE, gx, gy);
    }
    if (!isfinite(d->fNW)) {
        valid = false;
        printf("ERR(subdomain=0): Invalid value of fNW (%f) at: (%d, %d)\n", d->fNW, gx, gy);
    }
    if (!isfinite(d->fSW)) {
        valid = false;
        printf("ERR(subdomain=0): Invalid value of fSW (%f) at: (%d, %d)\n", d->fSW, gx, gy);
    }
    if (!isfinite(d->fSE)) {
        valid = false;
        printf("ERR(subdomain=0): Invalid value of fSE (%f) at: (%d, %d)\n", d->fSE, gx, gy);
    }
    
    if (!valid) {
        die();
    }
}

// Load the distributions from din to dout, for the node with the index 'idx'.


// Performs propagation when reading distributions from global memory.
// This implements the propagate-on-read scheme.


// Implements the propagate-on-read scheme for the AA access pattern, where the
// distributions are not located in their natural slots, but the opposite ones
// (e.g. fNE is located where fSW normally is). This ensures that within a single
// timestep, the distributions are read from and written to the exact same places
// in global memory.


__device__ inline void getDist(Dist *dout,  const float *__restrict__ din, unsigned int gi) {
    dout->fC = din[gi + DIST_SIZE * 0 + (unsigned int)0];
    dout->fE = din[gi + DIST_SIZE * 1 + (unsigned int)0];
    dout->fN = din[gi + DIST_SIZE * 2 + (unsigned int)0];
    dout->fW = din[gi + DIST_SIZE * 3 + (unsigned int)0];
    dout->fS = din[gi + DIST_SIZE * 4 + (unsigned int)0];
    dout->fNE = din[gi + DIST_SIZE * 5 + (unsigned int)0];
    dout->fNW = din[gi + DIST_SIZE * 6 + (unsigned int)0];
    dout->fSW = din[gi + DIST_SIZE * 7 + (unsigned int)0];
    dout->fSE = din[gi + DIST_SIZE * 8 + (unsigned int)0];
}

// Returns a node parameter which is a vector (in 'out').
__device__ inline void node_param_get_vector(const int idx, float *out) {
    out[0] = node_params[idx];
    out[1] = node_params[idx + 1];
}

// Returns a node parameter which is a scalar.
__device__ inline float node_param_get_scalar(const int idx) {
    return node_params[idx];
}

// Add comments for the Guo density implementation.
__device__ inline void bounce_back(Dist *fi)
{
    float t;
    
    t = fi->fE;
    fi->fE = fi->fW;
    fi->fW = t;
    t = fi->fN;
    fi->fN = fi->fS;
    fi->fS = t;
    t = fi->fNE;
    fi->fNE = fi->fSW;
    fi->fSW = t;
    t = fi->fNW;
    fi->fNW = fi->fSE;
    fi->fSE = t;
}

// Compute the 0th moment of the distributions, i.e. density.
__device__ inline void compute_0th_moment(Dist *fi, float *out)
{
    *out = fi->fC + fi->fE + fi->fN + fi->fNE + fi->fNW + fi->fS + fi->fSE + fi->fSW + fi->fW;
}

// Compute the 1st moments of the distributions, i.e. momentum.
__device__ inline void compute_1st_moment(Dist *fi, float *out, int add, float factor)
{
    if (add) {
        out[0] += factor * (fi->fE + fi->fNE - fi->fNW + fi->fSE - fi->fSW - fi->fW);
        out[1] += factor * (fi->fN + fi->fNE + fi->fNW - fi->fS - fi->fSE - fi->fSW);
    } else {
        out[0] = factor * (fi->fE + fi->fNE - fi->fNW + fi->fSE - fi->fSW - fi->fW);
        out[1] = factor * (fi->fN + fi->fNE + fi->fNW - fi->fS - fi->fSE - fi->fSW);
    }
}

// Compute the 2nd moments of the distributions.  Order of components is:
// 2D: xx, xy, yy
// 3D: xx, xy, xz, yy, yz, zz
__device__ inline void compute_2nd_moment(Dist *fi, float *out)
{
    out[0] = fi->fE + fi->fNE + fi->fNW + fi->fSE + fi->fSW + fi->fW;
    out[1] = fi->fNE - fi->fNW - fi->fSE + fi->fSW;
    out[2] = fi->fN + fi->fNE + fi->fNW + fi->fS + fi->fSE + fi->fSW;
}

// Computes the 2nd moment of the non-equilibrium distribution function
// given the full distribution fuction 'fi'.
__device__ inline void compute_noneq_2nd_moment(Dist* fi, const float rho, float *v0, float *out)
{
    out[0] = fi->fE + fi->fNE + fi->fNW + fi->fSE + fi->fSW + fi->fW - rho*((v0[0]*v0[0]) + 1.0f* (1.0f / 3.0f));
    out[1] = fi->fNE - fi->fNW - fi->fSE + fi->fSW - rho*v0[0]*v0[1];
    out[2] = fi->fN + fi->fNE + fi->fNW + fi->fS + fi->fSE + fi->fSW - rho*((v0[1]*v0[1]) + 1.0f* (1.0f / 3.0f));
}

// Compute the 1st moments of the distributions and divide it by the 0-th moment
// i.e. compute velocity.
__device__ inline void compute_1st_div_0th(Dist *fi, float *out, float zero)
{
    out[0] = (fi->fE + fi->fNE - fi->fNW + fi->fSE - fi->fSW - fi->fW)/zero;
    out[1] = (fi->fN + fi->fNE + fi->fNW - fi->fS - fi->fSE - fi->fSW)/zero;
}

__device__ inline void compute_macro_quant(Dist *fi, float *rho, float *v)
{
    compute_0th_moment(fi, rho);
    compute_1st_div_0th(fi, v, *rho);
}

__device__ inline void get0thMoment(Dist *fi, int node_type, int orientation, float *out)
{
    compute_0th_moment(fi, out);
}

// Common code for the equilibrium and Zou-He density boundary conditions.

//
// Get macroscopic density rho and velocity v given a distribution fi, and
// the node class node_type.
//
__device__ inline void getMacro(Dist *fi, int ncode, int node_type, int orientation, float *rho, float *v0)
{
    if (NTUsesStandardMacro(node_type) || orientation == 0) {
        compute_macro_quant(fi, rho, v0);
    }
    else if (isNTRegularizedVelocity(node_type)) {
        
        int node_param_idx = decodeNodeParamIdx(ncode);
        // We're dealing with a boundary node, for which some of the distributions
        // might be meaningless.  Fill them with the values of the opposite
        // distributions.
        
        switch (orientation) {
            case 1: {
                // fE is undefined.
                fi->fE = fi->fW;
                
                // fNE is undefined.
                fi->fNE = fi->fSW;
                
                // fSE is undefined.
                fi->fSE = fi->fNW;
                
                break;
            }
            case 2: {
                // fN is undefined.
                fi->fN = fi->fS;
                
                // fNE is undefined.
                fi->fNE = fi->fSW;
                
                // fNW is undefined.
                fi->fNW = fi->fSE;
                
                break;
            }
            case 3: {
                // fW is undefined.
                fi->fW = fi->fE;
                
                // fNW is undefined.
                fi->fNW = fi->fSE;
                
                // fSW is undefined.
                fi->fSW = fi->fNE;
                
                break;
            }
            case 4: {
                // fS is undefined.
                fi->fS = fi->fN;
                
                // fSW is undefined.
                fi->fSW = fi->fNE;
                
                // fSE is undefined.
                fi->fSE = fi->fNW;
                
                break;
            }
        }
        
        
        *rho = fi->fC + fi->fE + fi->fN + fi->fNE + fi->fNW + fi->fS + fi->fSE + fi->fSW + fi->fW;
        node_param_get_vector(node_param_idx, v0
                              );
        
        switch (orientation) {
            case 1:
                *rho = (*rho)/(-v0[0] + 1.0f);
                break;
            case 2:
                *rho = (*rho)/(-v0[1] + 1.0f);
                break;
            case 3:
                *rho = (*rho)/(v0[0] + 1.0f);
                break;
            case 4:
                *rho = (*rho)/(v0[1] + 1.0f);
                break;
        }
        
    }
}

// Uses extrapolation/other schemes to compute missing distributions for some implementations
// of boundary condtitions.
__device__ inline void fixMissingDistributions(Dist *fi,  float *dist_in, int ncode, int node_type, int orientation, unsigned int gi, float *__restrict__ ivx, float *__restrict__ ivy, float *gg0m0)
{ }


// TODO: Check whether it is more efficient to actually recompute
// node_type and orientation instead of passing them as variables.
__device__ inline void postcollisionBoundaryConditions(Dist *fi, int ncode, int node_type, int orientation, float *rho, float *v0, unsigned int gi,  float *dist_out)
{ }


__device__ inline void precollisionBoundaryConditions(Dist *fi, int ncode, int node_type, int orientation, float *rho, float *v0)
{
    if (isNTFullBBWall(node_type)) {
        bounce_back(fi);
    }
    else if (isNTRegularizedVelocity(node_type)) {
        
        // Bounce-back of the non-equilibrium parts.
        switch (orientation) {
                
            case 1:
                fi->fE = fi->fW + (2.0f* (1.0f / 3.0f))*(*rho)*v0[0];
                
                fi->fNE = fi->fSW + (*rho)*((1.0f* (1.0f * (1.0f / 6.0f)))*v0[0] + (1.0f* (1.0f * (1.0f / 6.0f)))*v0[1]);
                
                fi->fSE = fi->fNW + (*rho)*((1.0f* (1.0f * (1.0f / 6.0f)))*v0[0] - 1.0f* (1.0f * (1.0f / 6.0f))*v0[1]);
                break;
                
            case 2:
                fi->fN = fi->fS + (2.0f* (1.0f / 3.0f))*(*rho)*v0[1];
                
                fi->fNE = fi->fSW + (*rho)*((1.0f* (1.0f * (1.0f / 6.0f)))*v0[0] + (1.0f* (1.0f * (1.0f / 6.0f)))*v0[1]);
                
                fi->fNW = fi->fSE + (*rho)*(-1.0f* (1.0f * (1.0f / 6.0f))*v0[0] + (1.0f* (1.0f * (1.0f / 6.0f)))*v0[1]);
                break;
                
            case 3:
                fi->fW = fi->fE - 2.0f* (1.0f / 3.0f)*(*rho)*v0[0];
                
                fi->fNW = fi->fSE + (*rho)*(-1.0f* (1.0f * (1.0f / 6.0f))*v0[0] + (1.0f* (1.0f * (1.0f / 6.0f)))*v0[1]);
                
                fi->fSW = fi->fNE + (*rho)*(-1.0f* (1.0f * (1.0f / 6.0f))*v0[0] - 1.0f* (1.0f * (1.0f / 6.0f))*v0[1]);
                break;
                
            case 4:
                fi->fS = fi->fN - 2.0f* (1.0f / 3.0f)*(*rho)*v0[1];
                
                fi->fSW = fi->fNE + (*rho)*(-1.0f* (1.0f * (1.0f / 6.0f))*v0[0] - 1.0f* (1.0f * (1.0f / 6.0f))*v0[1]);
                
                fi->fSE = fi->fNW + (*rho)*((1.0f* (1.0f * (1.0f / 6.0f)))*v0[0] - 1.0f* (1.0f * (1.0f / 6.0f))*v0[1]);
                break;
                
            case 0:
                bounce_back(fi);
                return;
        }
        
        float flux[3];
        compute_noneq_2nd_moment(fi, *rho, v0, flux);
        
        
        fi->fC = max(1e-7f,
                     (4.0f* (1.0f / 9.0f))*(*rho)*(-3.0f* (1.0f * (1.0f / 2.0f))*(v0[0]*v0[0]) - 3.0f* (1.0f * (1.0f / 2.0f))*(v0[1]*v0[1])) + (4.0f* (1.0f / 9.0f))*(*rho)
                     +
                     -2.0f* (1.0f / 3.0f)*flux[0] - 2.0f* (1.0f / 3.0f)*flux[2]
                     );
        fi->fE = max(1e-7f,
                     (1.0f* (1.0f / 9.0f))*(*rho)*(v0[0]*(3.0f*v0[0] + 3.0f) - 3.0f* (1.0f * (1.0f / 2.0f))*(v0[1]*v0[1])) + (1.0f* (1.0f / 9.0f))*(*rho)
                     +
                     (1.0f* (1.0f / 3.0f))*flux[0] - 1.0f* (1.0f / 6.0f)*flux[2]
                     );
        fi->fN = max(1e-7f,
                     (1.0f* (1.0f / 9.0f))*(*rho)*(-3.0f* (1.0f * (1.0f / 2.0f))*(v0[0]*v0[0]) + v0[1]*(3.0f*v0[1] + 3.0f)) + (1.0f* (1.0f / 9.0f))*(*rho)
                     +
                     -1.0f* (1.0f / 6.0f)*flux[0] + (1.0f* (1.0f / 3.0f))*flux[2]
                     );
        fi->fW = max(1e-7f,
                     (1.0f* (1.0f / 9.0f))*(*rho)*(v0[0]*(3.0f*v0[0] - 3.0f) - 3.0f* (1.0f * (1.0f / 2.0f))*(v0[1]*v0[1])) + (1.0f* (1.0f / 9.0f))*(*rho)
                     +
                     (1.0f* (1.0f / 3.0f))*flux[0] - 1.0f* (1.0f / 6.0f)*flux[2]
                     );
        fi->fS = max(1e-7f,
                     (1.0f* (1.0f / 9.0f))*(*rho)*(-3.0f* (1.0f * (1.0f / 2.0f))*(v0[0]*v0[0]) + v0[1]*(3.0f*v0[1] - 3.0f)) + (1.0f* (1.0f / 9.0f))*(*rho)
                     +
                     -1.0f* (1.0f / 6.0f)*flux[0] + (1.0f* (1.0f / 3.0f))*flux[2]
                     );
        fi->fNE = max(1e-7f,
                      (1.0f* (1.0f / 36.0f))*(*rho)*(v0[0]*(3.0f*v0[0] + 9.0f*v0[1] + 3.0f) + v0[1]*(3.0f*v0[1] + 3.0f)) + (1.0f* (1.0f / 36.0f))*(*rho)
                      +
                      (1.0f* (1.0f / 12.0f))*flux[0] + (1.0f* (1.0f / 4.0f))*flux[1] + (1.0f* (1.0f / 12.0f))*flux[2]
                      );
        fi->fNW = max(1e-7f,
                      (1.0f* (1.0f / 36.0f))*(*rho)*(v0[0]*(3.0f*v0[0] - 9.0f*v0[1] - 3.0f) + v0[1]*(3.0f*v0[1] + 3.0f)) + (1.0f* (1.0f / 36.0f))*(*rho)
                      +
                      (1.0f* (1.0f / 12.0f))*flux[0] - 1.0f* (1.0f / 4.0f)*flux[1] + (1.0f* (1.0f / 12.0f))*flux[2]
                      );
        fi->fSW = max(1e-7f,
                      (1.0f* (1.0f / 36.0f))*(*rho)*(v0[0]*(3.0f*v0[0] + 9.0f*v0[1] - 3.0f) + v0[1]*(3.0f*v0[1] - 3.0f)) + (1.0f* (1.0f / 36.0f))*(*rho)
                      +
                      (1.0f* (1.0f / 12.0f))*flux[0] + (1.0f* (1.0f / 4.0f))*flux[1] + (1.0f* (1.0f / 12.0f))*flux[2]
                      );
        fi->fSE = max(1e-7f,
                      (1.0f* (1.0f / 36.0f))*(*rho)*(v0[0]*(3.0f*v0[0] - 9.0f*v0[1] + 3.0f) + v0[1]*(3.0f*v0[1] - 3.0f)) + (1.0f* (1.0f / 36.0f))*(*rho)
                      +
                      (1.0f* (1.0f / 12.0f))*flux[0] - 1.0f* (1.0f / 4.0f)*flux[1] + (1.0f* (1.0f / 12.0f))*flux[2]
                      );
    }
    
    
}

//
// Performs the relaxation step in the BGK model given the density rho,
// the velocity v and the distribution fi.
__device__ inline void BGK_relaxate0(float rho, float *iv0, Dist *d0, int node_type, int ncode)
{
    float v0[2];
    
    Dist feq0;
    
    v0[0] = iv0[0];
    v0[1] = iv0[1];
    
    feq0.fC = (4.0f* (1.0f / 9.0f))*rho*(-3.0f* (1.0f * (1.0f / 2.0f))*(v0[0]*v0[0]) - 3.0f* (1.0f * (1.0f / 2.0f))*(v0[1]*v0[1])) + (4.0f* (1.0f / 9.0f))*rho;
    
    feq0.fE = (1.0f* (1.0f / 9.0f))*rho*(v0[0]*(3.0f*v0[0] + 3.0f) - 3.0f* (1.0f * (1.0f / 2.0f))*(v0[1]*v0[1])) + (1.0f* (1.0f / 9.0f))*rho;
    
    feq0.fN = (1.0f* (1.0f / 9.0f))*rho*(-3.0f* (1.0f * (1.0f / 2.0f))*(v0[0]*v0[0]) + v0[1]*(3.0f*v0[1] + 3.0f)) + (1.0f* (1.0f / 9.0f))*rho;
    
    feq0.fW = (1.0f* (1.0f / 9.0f))*rho*(v0[0]*(3.0f*v0[0] - 3.0f) - 3.0f* (1.0f * (1.0f / 2.0f))*(v0[1]*v0[1])) + (1.0f* (1.0f / 9.0f))*rho;
    
    feq0.fS = (1.0f* (1.0f / 9.0f))*rho*(-3.0f* (1.0f * (1.0f / 2.0f))*(v0[0]*v0[0]) + v0[1]*(3.0f*v0[1] - 3.0f)) + (1.0f* (1.0f / 9.0f))*rho;
    
    feq0.fNE = (1.0f* (1.0f / 36.0f))*rho*(v0[0]*(3.0f*v0[0] + 9.0f*v0[1] + 3.0f) + v0[1]*(3.0f*v0[1] + 3.0f)) + (1.0f* (1.0f / 36.0f))*rho;
    
    feq0.fNW = (1.0f* (1.0f / 36.0f))*rho*(v0[0]*(3.0f*v0[0] - 9.0f*v0[1] - 3.0f) + v0[1]*(3.0f*v0[1] + 3.0f)) + (1.0f* (1.0f / 36.0f))*rho;
    feq0.fSW = (1.0f* (1.0f / 36.0f))*rho*(v0[0]*(3.0f*v0[0] + 9.0f*v0[1] - 3.0f) + v0[1]*(3.0f*v0[1] - 3.0f)) + (1.0f* (1.0f / 36.0f))*rho;
    
    feq0.fSE = (1.0f* (1.0f / 36.0f))*rho*(v0[0]*(3.0f*v0[0] - 9.0f*v0[1] + 3.0f) + v0[1]*(3.0f*v0[1] - 3.0f)) + (1.0f* (1.0f / 36.0f))*rho;
    
    float omega = 2.85714285714285698425e-01f;
    
    d0->fC += omega * (feq0.fC - d0->fC);
    d0->fE += omega * (feq0.fE - d0->fE);
    d0->fN += omega * (feq0.fN - d0->fN);
    d0->fW += omega * (feq0.fW - d0->fW);
    d0->fS += omega * (feq0.fS - d0->fS);
    d0->fNE += omega * (feq0.fNE - d0->fNE);
    d0->fNW += omega * (feq0.fNW - d0->fNW);
    d0->fSW += omega * (feq0.fSW - d0->fSW);
    d0->fSE += omega * (feq0.fSE - d0->fSE);
    
    // FIXME: This should be moved to postcollision boundary conditions.
}

// A kernel to set the node distributions using the equilibrium distributions
// and the macroscopic fields.
__global__ void SetInitialConditions(float *dist1_in,
                                     float *__restrict__ ivx,
                                     float *__restrict__ ivy,
                                     const float *__restrict__ irho,
                                     const int *__restrict__ map
                                     )
{
    
    int lx = get_local_id(0); // ID inside the current block
    int gx = get_global_id(0);
    int gy = get_group_id(1);
    
    unsigned int gi = getGlobalIdx(gx, gy);
    
    // Nothing to do if we're outside of the simulation domain.
    if (gx > 257) {
        return;
    }
    
    // Cache macroscopic fields in local variables.
    float rho = irho[gi] ;
    float v0[2];
    
    v0[0] = ivx[gi];
    v0[1] = ivy[gi];
    
    dist1_in[gi + (0u + (unsigned int)(0 + 0))] =
    (4.0f* (1.0f / 9.0f))*rho*(-3.0f* (1.0f * (1.0f / 2.0f))*(v0[0]*v0[0]) - 3.0f* (1.0f * (1.0f / 2.0f))*(v0[1]*v0[1])) + (4.0f* (1.0f / 9.0f))*rho;
    
    dist1_in[gi + (74304u + (unsigned int)(0 + 0))] =
    (1.0f* (1.0f / 9.0f))*rho*(v0[0]*(3.0f*v0[0] + 3.0f) - 3.0f* (1.0f * (1.0f / 2.0f))*(v0[1]*v0[1])) + (1.0f* (1.0f / 9.0f))*rho;
    
    dist1_in[gi + (148608u + (unsigned int)(0 + 0))] =
    (1.0f* (1.0f / 9.0f))*rho*(-3.0f* (1.0f * (1.0f / 2.0f))*(v0[0]*v0[0]) + v0[1]*(3.0f*v0[1] + 3.0f)) + (1.0f* (1.0f / 9.0f))*rho;
    
    dist1_in[gi + (222912u + (unsigned int)(0 + 0))] =
    (1.0f* (1.0f / 9.0f))*rho*(v0[0]*(3.0f*v0[0] - 3.0f) - 3.0f* (1.0f * (1.0f / 2.0f))*(v0[1]*v0[1])) + (1.0f* (1.0f / 9.0f))*rho;
    
    dist1_in[gi + (297216u + (unsigned int)(0 + 0))] =
    (1.0f* (1.0f / 9.0f))*rho*(-3.0f* (1.0f * (1.0f / 2.0f))*(v0[0]*v0[0]) + v0[1]*(3.0f*v0[1] - 3.0f)) + (1.0f* (1.0f / 9.0f))*rho;
    
    dist1_in[gi + (371520u + (unsigned int)(0 + 0))] =
    (1.0f* (1.0f / 36.0f))*rho*(v0[0]*(3.0f*v0[0] + 9.0f*v0[1] + 3.0f) + v0[1]*(3.0f*v0[1] + 3.0f)) + (1.0f* (1.0f / 36.0f))*rho;
    
    dist1_in[gi + (445824u + (unsigned int)(0 + 0))] =
    (1.0f* (1.0f / 36.0f))*rho*(v0[0]*(3.0f*v0[0] - 9.0f*v0[1] - 3.0f) + v0[1]*(3.0f*v0[1] + 3.0f)) + (1.0f* (1.0f / 36.0f))*rho;
    
    dist1_in[gi + (520128u + (unsigned int)(0 + 0))] =
    (1.0f* (1.0f / 36.0f))*rho*(v0[0]*(3.0f*v0[0] + 9.0f*v0[1] - 3.0f) + v0[1]*(3.0f*v0[1] - 3.0f)) + (1.0f* (1.0f / 36.0f))*rho;
    
    dist1_in[gi + (594432u + (unsigned int)(0 + 0))] =
    (1.0f* (1.0f / 36.0f))*rho*(v0[0]*(3.0f*v0[0] - 9.0f*v0[1] + 3.0f) + v0[1]*(3.0f*v0[1] - 3.0f)) + (1.0f* (1.0f / 36.0f))*rho;
    
    
}

__global__ void PrepareMacroFields(const int *__restrict__ map,
                                   const float *__restrict__ dist_in,
                                   float *orho,
                                   int options
                                   )
{
    int lx = get_local_id(0); // ID inside the current block
    int gx = get_global_id(0);
    int gy = get_group_id(1);
    
    unsigned int gi = getGlobalIdx(gx, gy);
    
    // Nothing to do if we're outside of the simulation domain.
    if (gx > 257) {
        return;
    }
    
    int ncode = map[gi];
    int type = decodeNodeType(ncode);
    
    // Unused nodes do not participate in the simulation.
    if (isExcludedNode(type) || isPropagationOnly(type))
    return;
    
    int orientation = decodeNodeOrientation(ncode);
    
    Dist fi;
    float out;
    getDist(&fi, dist_in, gi);
    get0thMoment(&fi, type, orientation, &out);
    orho[gi] = out;
}

__global__ void CollideAndPropagate(const int *__restrict__ map,
                                    float *__restrict__ dist_in,
                                    float *__restrict__ dist_out,
                                    float *__restrict__ gg0m0,
                                    float *__restrict__ ovx,
                                    float *__restrict__ ovy,
                                    int options
                                    )
{
    int lx = get_local_id(0); // ID inside the current block
    int gx = get_global_id(0);
    int gy = get_group_id(1);
    
    unsigned int gi = getGlobalIdx(gx, gy);
    
    // Nothing to do if we're outside of the simulation domain.
    if (gx > 257) {
        return;
    }
    
    // Shared variables for in-block propagation
    __shared__ float prop_fE[BLOCK_SIZE];
    __shared__ float prop_fNE[BLOCK_SIZE];
    __shared__ float prop_fSE[BLOCK_SIZE];
#define prop_fW prop_fE
#define prop_fSW prop_fNE
#define prop_fNW prop_fSE
    
    
    int ncode = map[gi];
    int type = decodeNodeType(ncode);
    
    // Unused nodes do not participate in the simulation.
    if (isExcludedNode(type)) {
        return;
    }
    
    int orientation = decodeNodeOrientation(ncode);
    
    // Cache the distributions in local variables
    Dist d0;
    if (!isPropagationOnly(type) ) {
        getDist(&d0, dist_in, gi);
        fixMissingDistributions(&d0, dist_in, ncode, type, orientation, gi,
                                ovx, ovy , gg0m0);
        
        // Macroscopic quantities for the current cell
        float g0m0, v[2];
        getMacro(&d0, ncode, type, orientation, &g0m0, v);
        
        precollisionBoundaryConditions(&d0, ncode, type, orientation, &g0m0, v);
        
        if (isWetNode(type)) {
            BGK_relaxate0(g0m0, v, &d0, type, ncode);
        }
        
        postcollisionBoundaryConditions(&d0, ncode, type, orientation, &g0m0, v, gi, dist_out);
        
        if (isWetNode(type) ) {
            checkInvalidValues(&d0, gx, gy);
        }
        
        // Only save the macroscopic quantities if requested to do so.
        if ((options & OPTION_SAVE_MACRO_FIELDS) && isWetNode(type)) {
            gg0m0[gi] = g0m0 ;
            
            ovx[gi] = v[0];
            ovy[gi] = v[1];
        }
        
    }  // propagation only
    
    const bool propagation_only = isPropagationOnly(type);
    
    // Initialize the shared array with invalid sentinel values.  If the sentinel
    // value is not subsequently overridden, it will not be propagated.
    prop_fE[lx] = -1.0f;
    
    __syncthreads();
    
    
    if (!propagation_only ) {
        // Update the 0-th direction distribution
        dist_out[gi] = d0.fC;
        
        // Propagation in directions orthogonal to the X axis (global memory)
        {
            if (gy < 257) {
                dist_out[gi + (148608u + (unsigned int)(0 + 288))] = d0.fN;
            }
        }
        {
            if (gy > 0) {
                dist_out[gi + (297216u + (unsigned int)(0 + -288))] = d0.fS;
            }
        }
        
        // E propagation in shared memory
        if (gx < 257) {
            // Note: propagation to ghost nodes is done directly in global memory as there
            // are no threads running for the ghost nodes.
            if (lx < 63 && gx != 256) {
                prop_fE[lx+1] = d0.fE;
                prop_fNE[lx+1] = d0.fNE;
                prop_fSE[lx+1] = d0.fSE;
                // E propagation in global memory (at right block boundary)
            } else {
                {
                    dist_out[gi + (74304u + (unsigned int)(0 + 1))] = d0.fE;
                }
                {
                    if (gy < 257) {
                        dist_out[gi + (371520u + (unsigned int)(0 + 289))] = d0.fNE;
                    }
                }
                {
                    if (gy > 0) {
                        dist_out[gi + (594432u + (unsigned int)(0 + -287))] = d0.fSE;
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    
    // Save locally propagated distributions into global memory.
    // The leftmost thread is not updated in this block.
    if (lx > 0 && gx < 258 && !propagation_only )
    if (prop_fE[lx] != -1.0f)
    {
        dist_out[gi + (74304u + (unsigned int)(0 + 0))] = prop_fE[lx];

        if (gy < 257) {
            dist_out[gi + (371520u + (unsigned int)(0 + 288))] = prop_fNE[lx];
        }
        
        if (gy > 0) {
            dist_out[gi + (594432u + (unsigned int)(0 + -288))] = prop_fSE[lx];
        }
    }
    
    __syncthreads();
    
    // Refill the propagation buffer with sentinel values.
    prop_fE[lx] = -1.0f;
    
    __syncthreads();
    
    if (!propagation_only ) {
        // W propagation in shared memory
        // Note: propagation to ghost nodes is done directly in global memory as there
        // are no threads running for the ghost nodes.
        if ((lx > 1 || (lx > 0 && gx >= 64)) && !propagation_only) {
            prop_fW[lx-1] = d0.fW;
            prop_fNW[lx-1] = d0.fNW;
            prop_fSW[lx-1] = d0.fSW;
            // W propagation in global memory (at left block boundary)
        } else if (gx > 0) {
            {
                dist_out[gi + (222912u + (unsigned int)(0 + -1))] = d0.fW;
            }
            {
                if (gy < 257) {
                    dist_out[gi + (445824u + (unsigned int)(0 + 287))] = d0.fNW;
                }
            }
            {
                if (gy > 0) {
                    dist_out[gi + (520128u + (unsigned int)(0 + -289))] = d0.fSW;
                }
            }
        }
    }
    
    __syncthreads();
    
    
    // The rightmost thread is not updated in this block.
    if (lx < 63 && gx < 257 && !propagation_only )
    if (prop_fE[lx] != -1.0f)
    {
        dist_out[gi + (222912u + (unsigned int)(0 + 0))] = prop_fW[lx];

        if (gy < 257) {
            dist_out[gi + (445824u + (unsigned int)(0 + 288))] = prop_fNW[lx];
        }
        
        if (gy > 0) {
            dist_out[gi + (520128u + (unsigned int)(0 + -288))] = prop_fSW[lx];
        }
    }
}

// Copies momentum transfer for a force object into a linear buffer
// so that a force can be computed easily via a sum reduction.
// TODO(michalj): Fuse this with summation to improve performance.
__global__ void ComputeForceObjects(const unsigned int *__restrict__ idx,
                                    const unsigned int *__restrict__ idx2,
                                    const float *__restrict__ dist,
                                    float *out,
                                    const unsigned int max_idx
                                    )
{
    const unsigned int gidx = get_global_id(0);
    if (gidx >= max_idx) {
        return;
    }
    const unsigned int gi = idx[gidx];
    const unsigned int gi2 = idx2[gidx];
    const float mx = dist[gi] + dist[gi2];
    out[gidx] = mx;
}

// Applies periodic boundary conditions within a single subdomain.
//  dist: pointer to the distributions array
//  axis: along which axis the PBCs are to be applied (0:x, 1:y, 2:z)
__global__ void ApplyPeriodicBoundaryConditions(float *dist, int axis)
{
    const int idx1 = get_global_id(0);
    unsigned int gi_low, gi_high;
    
    // For single block PBC, the envelope size (width of the ghost node
    // layer) is always 1.
    // TODO(michalj): Generalize this for the case when envelope_size != 1.
    if (axis == 0) {
        if (idx1 >= 258) { return; }
        gi_low = getGlobalIdx(0, idx1);       // ghost node
        gi_high = getGlobalIdx(256, idx1);  // real node
        
        {
            // TODO(michalj): Generalize this for grids with e_i > 1.
            // Load distributions to be propagated from low idx to high idx.
            
            const float ffW = dist[gi_low + DIST_SIZE * 3 + (unsigned int)0];
            
            const float ffNW = dist[gi_low + DIST_SIZE * 6 + (unsigned int)0];
            
            const float ffSW = dist[gi_low + DIST_SIZE * 7 + (unsigned int)0];
            
            
            if (gi_high != INVALID_NODE && isfinite(ffW)) {
                dist[gi_high + DIST_SIZE * 3 + (unsigned int)0] = ffW;
            }
            
            if (isfinite(ffNW)) {
                // Skip distributions which are not populated or cross multiple boundaries.
                if (idx1 > 1 && idx1 <= 256) {
                    dist[gi_high + DIST_SIZE * 6 + (unsigned int)0] = ffNW;
                }
            }
            
            if (isfinite(ffSW)) {
                // Skip distributions which are not populated or cross multiple boundaries.
                if (idx1 < 256 && idx1 >= 1) {
                    dist[gi_high + DIST_SIZE * 7 + (unsigned int)0] = ffSW;
                }
            }
        }  // low to high
        
        {
            // Load distributrions to be propagated from high idx to low idx.
            
            const float ffE = dist[gi_high + DIST_SIZE * 1 + (unsigned int)1];
            
            const float ffNE = dist[gi_high + DIST_SIZE * 5 + (unsigned int)1];
            
            const float ffSE = dist[gi_high + DIST_SIZE * 8 + (unsigned int)1];
            
            
            if (isfinite(ffE) && gi_low != INVALID_NODE) {
                dist[gi_low + DIST_SIZE * 1 + (unsigned int)1] = ffE;
            }
            
            if (isfinite(ffNE)) {
                // Skip distributions which are not populated or cross multiple boundaries.
                if (idx1 > 1 && idx1 <= 256 ) {
                    dist[gi_low + DIST_SIZE * 5 + (unsigned int)1] = ffNE;
                }
            }
            
            if (isfinite(ffSE)) {
                // Skip distributions which are not populated or cross multiple boundaries.
                if (idx1 < 256 && idx1 >= 1 ) {
                    dist[gi_low + DIST_SIZE * 8 + (unsigned int)1] = ffSE;
                }
            }
        }  // high to low
        
    } else if (axis == 1) {
        if (idx1 >= 258) { return; }
        gi_low = getGlobalIdx(idx1, 0);       // ghost node
        gi_high = getGlobalIdx(idx1, 256);  // real node
        
        {
            // TODO(michalj): Generalize this for grids with e_i > 1.
            // Load distributions to be propagated from low idx to high idx.
            
            const float ffS = dist[gi_low + DIST_SIZE * 4 + (unsigned int)0];
            
            const float ffSW = dist[gi_low + DIST_SIZE * 7 + (unsigned int)0];
            
            const float ffSE = dist[gi_low + DIST_SIZE * 8 + (unsigned int)0];
            
            if (gi_high != INVALID_NODE && isfinite(ffS)) {
                dist[gi_high + DIST_SIZE * 4 + (unsigned int)0] = ffS;
            }
            
            if (isfinite(ffSW)) {
                // Skip distributions which are not populated or cross multiple boundaries.
                if (idx1 < 256 && idx1 >= 1) {
                    dist[gi_high + DIST_SIZE * 7 + (unsigned int)0] = ffSW;
                }
            }
            
            if (isfinite(ffSE)) {
                // Skip distributions which are not populated or cross multiple boundaries.
                if (idx1 > 1 && idx1 <= 256) {
                    dist[gi_high + DIST_SIZE * 8 + (unsigned int)0] = ffSE;
                }
            }
        }  // low to high
        
        
        
        {
            // Load distributrions to be propagated from high idx to low idx.
            
            const float ffN = dist[gi_high + DIST_SIZE * 2 + (unsigned int)288];
            
            const float ffNE = dist[gi_high + DIST_SIZE * 5 + (unsigned int)288];
            
            const float ffNW = dist[gi_high + DIST_SIZE * 6 + (unsigned int)288];
            
            
            if (isfinite(ffN) && gi_low != INVALID_NODE) {
                dist[gi_low + DIST_SIZE * 2 + (unsigned int)288] = ffN;
            }
            
            if (isfinite(ffNE)) {
                // Skip distributions which are not populated or cross multiple boundaries.
                if (idx1 > 1 && idx1 <= 256 ) {
                    dist[gi_low + DIST_SIZE * 5 + (unsigned int)288] = ffNE;
                }
            }
            
            if (isfinite(ffNW)) {
                // Skip distributions which are not populated or cross multiple boundaries.
                if (idx1 < 256 && idx1 >= 1 ) {
                    dist[gi_low + DIST_SIZE * 6 + (unsigned int)288] = ffNW;
                }
            }
        }  // high to low
    }
}

// Applies periodic boundary conditions to a scalar field within a single subdomain.
//  dist: pointer to the array with the field data
//  axis: along which axis the PBCs are to be applied (0:x, 1:y, 2:z)
__global__ void ApplyMacroPeriodicBoundaryConditions(float *field, int axis)
{
    const int idx1 = get_global_id(0);
    unsigned int gi_low, gi_high;
    
    // TODO(michalj): Generalize this for the case when envelope_size != 1.
    if (axis == 0) {
        if (idx1 >= 258) { return; }
        gi_low = getGlobalIdx(0, idx1);         // ghost node
        gi_high = getGlobalIdx(256, idx1);    // real node
        
        if ( isfinite(field[gi_high])) {
            field[gi_low] = field[gi_high];
        }
        
        gi_low = getGlobalIdx(1, idx1);         // real node
        gi_high = getGlobalIdx(257, idx1);    // ghost node
        
        if ( isfinite(field[gi_low])) {
            field[gi_high] = field[gi_low];
        }
        
    } else if (axis == 1) {
        if (idx1 >= 258) { return; }
        gi_low = getGlobalIdx(idx1, 0);         // ghost node
        gi_high = getGlobalIdx(idx1, 256);    // real node
        
        if ( isfinite(field[gi_high])) {
            field[gi_low] = field[gi_high];
        }
        
        gi_low = getGlobalIdx(idx1, 1);         // real node
        gi_high = getGlobalIdx(idx1, 257);    // ghost node
        
        if ( isfinite(field[gi_low])) {
            field[gi_high] = field[gi_low];
        }
    }
}

// Collects ghost node data for connections along axes other than X.
// dist: distributions array
// base_gy: where along the X axis to start collecting the data
// face: see LBBlock class constants
// buffer: buffer where the data is to be saved
__global__ void CollectContinuousData(float *dist, int face, int base_gx, int max_lx,  float *buffer)
{
    const int idx = get_global_id(0);
    float tmp;
    
    if (idx >= max_lx) {
        return;
    }
    
    switch (face) {
        case 2: {
            
            const int dist_size = max_lx / 3;
            const int dist_num = idx / dist_size;
            const int gx = idx % dist_size;
            unsigned int gi = getGlobalIdx(base_gx + gx, 0);
            
            switch (dist_num) {
                case 0: {
                    tmp = dist[gi + DIST_SIZE * 4 + (unsigned int)0];
                    break;
                }
                case 1: {
                    tmp = dist[gi + DIST_SIZE * 7 + (unsigned int)0];
                    break;
                }
                case 2: {
                    tmp = dist[gi + DIST_SIZE * 8 + (unsigned int)0];
                    break;
                }
            }
            buffer[idx] = tmp;
            break;
        }
        case 3: {
            
            const int dist_size = max_lx / 3;
            const int dist_num = idx / dist_size;
            const int gx = idx % dist_size;
            unsigned int gi = getGlobalIdx(base_gx + gx, 257);
            
            switch (dist_num) {
                case 0: {
                    tmp = dist[gi + DIST_SIZE * 2 + (unsigned int)0];
                    break;
                }
                case 1: {
                    tmp = dist[gi + DIST_SIZE * 5 + (unsigned int)0];
                    break;
                }
                case 2: {
                    tmp = dist[gi + DIST_SIZE * 6 + (unsigned int)0];
                    break;
                }
            }
            buffer[idx] = tmp;
            break;
        }
    }
}

__global__ void DistributeContinuousData(float *dist, int face, int base_gx,int max_lx,  float *buffer)
{
    
    const int idx = get_global_id(0);
    
    if (idx >= max_lx) {
        return;
    }
    
    switch (face) {
        case 2: {
            
            const int dist_size = max_lx / 3;
            const int dist_num = idx / dist_size;
            const int gx = idx % dist_size;
            const float tmp = buffer[idx];
            unsigned int gi = getGlobalIdx(base_gx + gx, 256);
            
            
            switch (dist_num) {
                case 0: {
                    dist[gi + DIST_SIZE * 4 + (unsigned int)0] = tmp;
                    break;
                }
                case 1: {
                    dist[gi + DIST_SIZE * 7 + (unsigned int)0] = tmp;
                    break;
                }
                case 2: {
                    dist[gi + DIST_SIZE * 8 + (unsigned int)0] = tmp;
                    break;
                }
            }
            
            break;
        }
        case 3: {
            
            const int dist_size = max_lx / 3;
            const int dist_num = idx / dist_size;
            const int gx = idx % dist_size;
            const float tmp = buffer[idx];
            unsigned int gi = getGlobalIdx(base_gx + gx, 1);
            
            
            switch (dist_num) {
                case 0: {
                    dist[gi + DIST_SIZE * 2 + (unsigned int)0] = tmp;
                    break;
                }
                case 1: {
                    dist[gi + DIST_SIZE * 5 + (unsigned int)0] = tmp;
                    break;
                }
                case 2: {
                    dist[gi + DIST_SIZE * 6 + (unsigned int)0] = tmp;
                    break;
                }
            }
            
            break;
        }
    }
}

__global__ void CollectSparseData(unsigned int *idx_array,  float *dist, float *buffer, int max_idx)
{
    int idx = get_global_id(0);
    
    if (idx >= max_idx) {
        return;
    }
    unsigned int gi = idx_array[idx];
    if (gi == INVALID_NODE) return;
    if (gi >= DIST_SIZE * 9) {
        printf("invalid node index detected in sparse coll %d (%d, %d)\n", gi, get_global_id(0), get_global_id(1));
        return;
    }
    buffer[idx] = dist[gi];
}

__global__ void DistributeSparseData(unsigned int *idx_array,  float *dist, float *buffer, int max_idx)
{
    int idx = get_global_id(0);
    if (idx >= max_idx) {
        return;
    }
    unsigned int gi = idx_array[idx];
    if (gi == INVALID_NODE) return;
    if (gi >= DIST_SIZE * 9) {
        printf("invalid node index detected in sparse dist %d (%d, %d)\n", gi, get_global_id(0), get_global_id(1));
        return;
    }
    
    dist[gi] = buffer[idx];
}

__global__ void CollectContinuousMacroData(float *field, int base_gx, int max_lx, int gy, float *buffer)
{
    const int idx = get_global_id(0);
    if (idx >= max_lx) {
        return;
    }
    
    unsigned int gi = getGlobalIdx(base_gx + idx, gy);
    
    buffer[idx] = field[gi];
}

__global__ void DistributeContinuousMacroData(float *field, int base_gx, int max_lx, int gy, float *buffer)
{
    const int idx = get_global_id(0);
    if (idx >= max_lx) {
        return;
    }
    
    unsigned int gi = getGlobalIdx(base_gx + idx, gy);
    
    field[gi] = buffer[idx];
}

