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


#include "palabos3D.h"
#include "palabos3D.hh"
#include "palabos2D.h"
#include "palabos2D.hh"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

using namespace plb;
using namespace std;

#ifdef USE_MYDESCRIPTOR

namespace plb {
namespace descriptors {

template <typename T> struct MyD3Q19DescriptorBase
    : public D3Q19Constants<T>, public NoOptimizationRoundOffPolicy<T>
{
    typedef D3Q19DescriptorBase<T> BaseDescriptor;
    enum { numPop=D3Q19Constants<T>::q };
};

template <typename T> struct MyD3Q19Descriptor
    : public MyD3Q19DescriptorBase<T>, public NoExternalFieldBase
{
    static const char name[];
};

template<typename T>
const char MyD3Q19Descriptor<T>::name[] = "MyD3Q19";

}
}

typedef double T;
#define DESCRIPTOR descriptors::MyD3Q19Descriptor

#else

typedef double T;
#define DESCRIPTOR descriptors::D3Q19Descriptor

#endif

void initialSetup(MultiBlockLattice3D<T,DESCRIPTOR>& lattice)
{
    const plint nx = lattice.getNx();
    const plint ny = lattice.getNy();
    const plint nz = lattice.getNz();
    initializeAtEquilibrium(lattice, lattice.getBoundingBox(), (T) 1., Array<T,3>((T)0.,(T)0.,(T)0.) );
    initializeAtEquilibrium(lattice, Box3D(nx/2,nx/2, ny/2,ny/2, nz/2,nz/2), (T) 2.0, Array<T,3>((T)0.,(T)0.,(T)0.) );
    lattice.initialize();
}

template<class BlockLatticeT>
void writeData( std::string prefix, BlockLatticeT& lattice, plint iter)
{
    const plint imSize = 600;
    const plint nx = lattice.getNx();
    const plint ny = lattice.getNy();
    const plint nz = lattice.getNz();

    Box3D slice(0, nx-1, 0, ny-1, nz/2, nz/2);
    ImageWriter<T> imageWriter("leeloo");

    imageWriter.writeScaledGif( createFileName("unorm_slice", iter, 6),
                                *computeVelocityNorm (lattice, slice),
                                imSize, imSize );

    plb_ofstream ofile((createFileName("./tmp/" + prefix + "unorm", iter, 6)+".dat").c_str());
    ofile << std::setprecision( 60 ) << std::fixed << *computeVelocityNorm(lattice);
    plb_ofstream ofile2((createFileName("./tmp/" + prefix + "f", iter, 6)+".dat").c_str());
    ofile2 << std::setprecision( 60 ) << std::fixed << *computeAllPopulations(lattice);
    plb_ofstream ofile3((createFileName("./tmp/" + prefix + "rho", iter, 6)+".dat").c_str());
    ofile3 << std::setprecision( 60 ) << std::fixed << *computeDensity(lattice);
    plb_ofstream ofile4((createFileName("./tmp/" + prefix + "feq", iter, 6)+".dat").c_str());
    ofile4 << std::setprecision( 60 ) << std::fixed << *computeEquilibrium(lattice);
}


int main(int argc, char* argv[])
{

    plbInit(&argc, &argv);
    global::directories().setOutputDir("./tmp/");

    plint nx = 100;
    plint ny = 100;
    plint nz = 10;
    plint maxIter = 20;
    plint r = 10;
    T uLB = 0.04;
    T RE = 220.0;
    T nuLB = uLB*r / RE;
    T omega = 1. / (3*nuLB+0.5);

    MultiBlockLattice3D<T, DESCRIPTOR> lattice(nx, ny, nz, new BGKdynamics<T,DESCRIPTOR>(omega));
    lattice.periodicity().toggleAll(true);

    initialSetup(lattice);

    for (plint iT=0; iT<maxIter; ++iT) {
        writeData("beforecollision", lattice, iT);
        lattice.collide();
        writeData("aftercollision", lattice, iT);
        lattice.stream();
        writeData("afterstreaming", lattice, iT);
    }
}

