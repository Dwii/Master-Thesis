/* This file is part of the Palabos library.
 *
 * Copyright (C) 2011-2015 FlowKit Sarl
 * Route d'Oron 2
 * 1010 Lausanne, Switzerland
 * E-mail contact: contact@flowkit.com
 *
 * The most recent release of Palabos can be downloaded at 
 * <http://www.palabos.org/>
 *
 * The library Palabos is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * The library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/** \file
  * Flow in a lid-driven 3D cavity. The cavity is square and has no-slip walls,
  * except for the top wall which is diagonally driven with a constant
  * velocity. The benchmark is challenging because of the velocity
  * discontinuities on corner nodes.
  **/

#include "palabos3D.h"
#include "palabos3D.hh"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdbool.h>
#include <libgen.h>
#include <timing.h>

using namespace plb;
using namespace std;

typedef double T;
#define DESCRIPTOR descriptors::D3Q19Descriptor

void cavitySetup( MultiBlockLattice3D<T,DESCRIPTOR>& lattice,
                  IncomprFlowParam<T> const& parameters,
                  OnLatticeBoundaryCondition3D<T,DESCRIPTOR>& boundaryCondition )
{
    const plint nx = parameters.getNx();
    const plint ny = parameters.getNy();
    const plint nz = parameters.getNz();
    Box3D topLid = Box3D(0, nx-1, ny-1, ny-1, 0, nz-1);
    Box3D everythingButTopLid = Box3D(0, nx-1, 0, ny-2, 0, nz-1);

    // All walls implement a Dirichlet velocity condition.
    boundaryCondition.setVelocityConditionOnBlockBoundaries(lattice);

    T u = std::sqrt((T)2)/(T)2 * parameters.getLatticeU();
    initializeAtEquilibrium(lattice, everythingButTopLid, (T)1., Array<T,3>((T)0.,(T)0.,(T)0.) );
    initializeAtEquilibrium(lattice, topLid, (T)1., Array<T,3>(u,(T)0.,u) );
    setBoundaryVelocity(lattice, topLid, Array<T,3>(u,(T)0.,u) );

    lattice.initialize();
}

template<class BlockLatticeT>
void writeGifs(BlockLatticeT& lattice, IncomprFlowParam<T> const& parameters, plint iter)
{
    const plint imSize = 600;
    const plint nx = parameters.getNx();
    const plint ny = parameters.getNy();
    const plint nz = parameters.getNz();
    //const plint zComponent = 2;

    Box3D slice(0, nx-1, 0, ny-1, nz/2, nz/2);
    ImageWriter<T> imageWriter("leeloo");

    //imageWriter.writeScaledGif( createFileName("uz", iter, 6),
    //                            *computeVelocityComponent (lattice, slice, zComponent),
    //                            imSize, imSize );

    imageWriter.writeScaledGif( createFileName("uNorm", iter, 6),
                                *computeVelocityNorm (lattice, slice),
                                imSize, imSize );
    //imageWriter.writeScaledGif( createFileName("omega", iter, 6),
    //                            *computeNorm(*computeVorticity (
    //                                    *computeVelocity(lattice) ), slice ),
    //                            imSize, imSize );
}

template<class BlockLatticeT>
void writeVTK(BlockLatticeT& lattice,
              IncomprFlowParam<T> const& parameters, plint iter)
{
    T dx = parameters.getDeltaX();
    T dt = parameters.getDeltaT();
    VtkImageOutput3D<T> vtkOut(createFileName("vtk", iter, 6), dx);
    vtkOut.writeData<float>(*computeVelocityNorm(lattice), "velocityNorm", dx/dt);
    vtkOut.writeData<3,float>(*computeVelocity(lattice), "velocity", dx/dt);
    vtkOut.writeData<3,float>(*computeVorticity(*computeVelocity(lattice)), "vorticity", 1./dt);
}


SparseBlockStructure3D createRegularDistribution3D (
        std::vector<plint> const& xVal, std::vector<plint> const& yVal, std::vector<plint> const& zVal )
{
    PLB_ASSERT(xVal.size()>=2);
    PLB_ASSERT(yVal.size()>=2);
    PLB_ASSERT(zVal.size()>=2);
    SparseBlockStructure3D dataGeometry (
            Box3D(xVal[0], xVal.back()-1, yVal[0], yVal.back()-1, zVal[0], zVal.back()-1) );
    for (plint iX=0; iX<(plint)xVal.size()-1; ++iX) {
        for (plint iY=0; iY<(plint)yVal.size()-1; ++iY) {
            for (plint iZ=0; iZ<(plint)zVal.size()-1; ++iZ) {
                plint nextID = dataGeometry.nextIncrementalId();
                Box3D domain( xVal[iX], xVal[iX+1]-1, yVal[iY],
                              yVal[iY+1]-1, zVal[iZ], zVal[iZ+1]-1 );
                dataGeometry.addBlock(domain, nextID);
                pcout << "Adding block with ID=" << nextID << ": ["
                      << domain.x0 << "," << domain.x1 << " | "
                      << domain.y0 << "," << domain.y1 << " | "
                      << domain.z0 << "," << domain.z1 << "]" << std::endl;
            }
        }
    }
    return dataGeometry;
}

SparseBlockStructure3D createCavityDistribution3D(
    plint nx, plint ny, plint nz,    // domain size
    plint snx, plint sny, plint snz) // subdomain size
{
    static const plint numval=4;

    plint x[numval] = {0, nx/2-snx/2, nx/2+snx/2+(snx%2), nx};
    plint y[numval] = {0, ny/2-sny/2, nx/2+sny/2+(sny%2), ny};
    plint z[numval] = {0, nz/2-snz/2, nx/2+snz/2+(snz%2), nz};

    std::vector<plint> xVal(x, x+numval);
    std::vector<plint> yVal(y, y+numval);
    std::vector<plint> zVal(z, z+numval);

    return createRegularDistribution3D(xVal, yVal, zVal);
}


MultiBlockLattice3D<T, DESCRIPTOR> palabos_create_lattice(IncomprFlowParam<T> parameters, plint snx, plint sny, plint snz)
{
    // Here the 5x5x5 cover-up is instantiated.
    plint numBlocksX = 3;
    plint numBlocksY = 3;
    plint numBlocksZ = 3;
    plint numBlocks = numBlocksX*numBlocksY*numBlocksZ;
    plint envelopeWidth = 1;
    SparseBlockStructure3D blockStructure (
        createCavityDistribution3D (
            parameters.getNx(), parameters.getNy(), parameters.getNz(),
            snx, sny, snz
        ) 
    );

    // In case of MPI parallelism, the blocks are explicitly assigned to processors,
    // with equal load.
    ExplicitThreadAttribution* threadAttribution = new ExplicitThreadAttribution;
    std::vector<std::pair<plint,plint> > ranges;
    plint numRanges = std::min(numBlocks, (plint)global::mpi().getSize());
    util::linearRepartition(0, numBlocks-1, numRanges, ranges);
    for (pluint iProc=0; iProc<ranges.size(); ++iProc) {
        for (plint blockId=ranges[iProc].first; blockId<=ranges[iProc].second; ++blockId) {
            threadAttribution -> addBlock(blockId, iProc);
            printf("Block %lu is run by processor %lu\n", blockId, iProc);
        }
    }

    // Create a lattice with the above specified internal structure.
    MultiBlockLattice3D<T, DESCRIPTOR> lattice (
        MultiBlockManagement3D ( blockStructure, threadAttribution, envelopeWidth ),
        defaultMultiBlockPolicy3D().getBlockCommunicator(),
        defaultMultiBlockPolicy3D().getCombinedStatistics(),
        defaultMultiBlockPolicy3D().getMultiCellAccess<T,DESCRIPTOR>(),
        new BGKdynamics<T,DESCRIPTOR>(parameters.getOmega()) 
    );

    return lattice;
}

typedef enum { OUT_NONE, OUT_FIN, OUT_FINP, OUT_IMG } out_mode;

float get_lups(long lattices, long iterations, long ns_time_diff)
{
    return lattices * iterations * 1000000000.0f / ns_time_diff;
}

#define CUBE(n) ((n)*(n)*(n))

int main(int argc, char * argv[])
{
    // Init options to default values
    std::string out_path = ".";
    std::string out_pref = "lbm";
    out_mode out = OUT_NONE;
    ssize_t max_iter = 0;
    size_t out_interval = 0;
    bool print_lups = false;
    bool print_avg_lups = false;
    size_t width, height, depth, domain_size;
    width = height = depth = domain_size = 0;
    bool print_avg_energy = false;
    bool copy_boundaries_only = false;
    bool print_time = false;
    bool print_total_time = false;

    // Read arguments
    while (optind < argc) {
        switch (getopt(argc, argv, "pfFi:I:o:O:lLtTx:y:z:N:eb")) {
            case 'p': { out = OUT_IMG; break; }
//            case 'f': { out = OUT_FIN; break; }
            case 'F': { out = OUT_FINP; break; }
            case 'i': { max_iter = strtol(optarg, NULL, 10); break; }
            case 'I': { out_interval = strtol(optarg, NULL, 10); break; }
            case 'o': { out_path = optarg; break; }
            case 'O': { out_pref = optarg; break; }
            case 'l': { print_lups = true; break; }
            case 'L': { print_avg_lups = true; break; }
            case 't': { print_time = true; break; }
            case 'T': { print_total_time = true; break; }
            case 'x': { width  = strtol(optarg, NULL, 10); break; }
            case 'y': { height = strtol(optarg, NULL, 10); break; }
            case 'z': { depth  = strtol(optarg, NULL, 10); break; }
            case 'N': { domain_size = strtol(optarg, NULL, 10); break; }
            case 'e': { print_avg_energy = true; break; }
            case 'b': { copy_boundaries_only = true; break; }
            default : { goto usage; }
        }
    }
    
    // check that execution mode is set (output images or fin values)
    if (max_iter < 1 || width <= 0 || height <= 0 || depth <= 0 || width > domain_size || height > domain_size || depth > domain_size) {
    usage:
        fprintf(stderr, "usage: %s (-p | -f | -F) -i <iter> [-I <out_interval>] [-o <out_dir>] [-O <out_prefix>] [-l] [-L] [-t] [-T] -x <nx> -y <ny> -z <nz> -N <N> [-e] [-b]\n", basename((char*)argv[0]));
        fprintf(stderr, "  -p : output pictures\n");
        fprintf(stderr, "  -f : output populations (UNAVAILABLE)\n");
        fprintf(stderr, "  -F : output populations formated like Palabos\n");
        fprintf(stderr, "  -i : number of iterations\n");
        fprintf(stderr, "  -I : output interval; (0 if only the last iteration output is required)\n");
        fprintf(stderr, "  -o : output file directory\n");
        fprintf(stderr, "  -O : output filename prefix\n");
        fprintf(stderr, "  -l : print lups at each output interval\n");
        fprintf(stderr, "  -L : print average lups at the end\n");
        fprintf(stderr, "  -t : print computation time (in ns) at each output interval\n");
        fprintf(stderr, "  -T : print total computation time  (in ns) at the end\n");
        fprintf(stderr, "  -x : subdomain width\n");
        fprintf(stderr, "  -y : subdomain height\n");
        fprintf(stderr, "  -z : subdomain depth\n");
        fprintf(stderr, "  -N : domain size (NxNxN)\n");        
        fprintf(stderr, "  -e : print average energy\n");
        fprintf(stderr, "  -b : copy only the boundaries between palabos and the coprocessor\n");
        return EXIT_FAILURE;
    }
    
    if (out == OUT_NONE) {
        fprintf(stderr, "No output mode specified.\n");
    }
   
    plbInit();
    global::directories().setOutputDir(out_path + "/");

    IncomprFlowParam<T> parameters(
            (T) 1e-4,  // uMax
            (T) 10.,   // Re
            domain_size, // N
            1.,        // lx
            1.,        // ly
            1.         // lz
    );

    MultiBlockLattice3D<T, DESCRIPTOR> lattice = palabos_create_lattice(parameters, width, height, depth);

    OnLatticeBoundaryCondition3D<T,DESCRIPTOR>* boundaryCondition = createInterpBoundaryCondition3D<T,DESCRIPTOR>();
    cavitySetup(lattice, parameters, *boundaryCondition);

    // @Tomasz: STEP 2
    // Here the co-processor is informed about the domains for which it will need to do computations.
    bool printInfo=true;
    initiateCoProcessors(lattice, BGKdynamics<T,DESCRIPTOR>(parameters.getOmega()).getId(), printInfo);

    if ( copy_boundaries_only ) transferToCoProcessors(lattice); // Transfer all data to co-processors initially.

    long time_diff, total_time_diff = 0;
    start_time_t start_time;
    timing_start(&start_time);
        
    for (int iter = 0; iter <= max_iter; iter++) {
        
        // Execute a time iteration.

        if ( copy_boundaries_only ) {
            transferToCoProcessors(lattice, 1);
        } else {
            transferToCoProcessors(lattice);
        }

        for (int i = 0; i < REPEAT_PALABOS_CNS; i++) {
            lattice.collideAndStream();
        }
        
        if ( copy_boundaries_only ){
            transferFromCoProcessors(lattice, 1);
        } else {
            transferFromCoProcessors(lattice);
        }

        // After transferring back from co-processor to CPU memory, data in the envelopes
        // of the CPU blocks must be synchronized.
        for (int i = 0; i < REPEAT_PALABOS_DO; i++) {
            lattice.duplicateOverlaps(modif::staticVariables);
        }

        if ( (!out_interval && iter == max_iter) || (out_interval && iter % out_interval == 0) ) {
            
            time_diff = timing_stop(&start_time);
            total_time_diff += time_diff;

            if ( print_lups ) {
                size_t iter_diff = out_interval? out_interval : (size_t)max_iter;
                printf("lups: %.2f\n", get_lups(CUBE(domain_size), iter_diff, time_diff));
                fflush(stdout);
            }

            if (print_time) {
                printf("time: %lu\n", time_diff);
            }

            if ( out || print_avg_energy ) {

                if ( copy_boundaries_only ) {
                    transferFromCoProcessors(lattice);
                    lattice.duplicateOverlaps(modif::staticVariables);
                }

                if ( print_avg_energy ) {
                    pcout << "average energy: " << std::setprecision(10) << computeAverageEnergy<T>(lattice) << std::endl;
                }
                
                if ( out == OUT_IMG ) {
                    writeGifs(lattice, parameters, iter);
                }
                
                if (out == OUT_FINP) {
                    plb_ofstream f_file((createFileName(out_path + "/" + out_pref, iter, 6)+".dat").c_str());
                    f_file << std::setprecision( 60 ) << std::fixed << *computeAllPopulations(lattice);
                }
            }

            timing_start(&start_time);
        }
    }

    if ( print_avg_lups ) {
        printf("average lups: %.2f\n", get_lups(CUBE(domain_size), max_iter, total_time_diff));
    }

    if (print_total_time) {
        printf("time: %lu\n", total_time_diff);
    }

    delete boundaryCondition;

    return EXIT_SUCCESS;
}
