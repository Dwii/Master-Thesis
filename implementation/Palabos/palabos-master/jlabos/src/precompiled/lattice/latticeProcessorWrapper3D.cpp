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

#ifdef COMPILE_3D

#include "atomicBlock/dataProcessorWrapper3D.h"
#include "atomicBlock/dataProcessorWrapper3D.hh"
#include "latticeBoltzmann/nearestNeighborLattices3D.h"
#include "latticeBoltzmann/nearestNeighborLattices3D.hh"

namespace plb {

/* *************** Boxed Data Processor functionals ****************** */

template
void applyProcessingFunctional<FLOAT_T, descriptors::DESCRIPTOR_3D> (
        BoxProcessingFunctional3D_L<FLOAT_T,descriptors::DESCRIPTOR_3D>* functional,
        Box3D domain, BlockLattice3D<FLOAT_T,descriptors::DESCRIPTOR_3D>& lattice );
template
void integrateProcessingFunctional<FLOAT_T, descriptors::DESCRIPTOR_3D> (
        BoxProcessingFunctional3D_L<FLOAT_T, descriptors::DESCRIPTOR_3D>* functional,
        Box3D domain, BlockLattice3D<FLOAT_T, descriptors::DESCRIPTOR_3D>& lattice, plint level );

template
void applyProcessingFunctional<FLOAT_T, descriptors::DESCRIPTOR_3D,
                               FLOAT_T, descriptors::DESCRIPTOR_3D> (
        BoxProcessingFunctional3D_LL<FLOAT_T,descriptors::DESCRIPTOR_3D,
                                     FLOAT_T,descriptors::DESCRIPTOR_3D>* functional,
        Box3D domain,
        BlockLattice3D<FLOAT_T,descriptors::DESCRIPTOR_3D>& lattice1,
        BlockLattice3D<FLOAT_T,descriptors::DESCRIPTOR_3D>& lattice2 );
template
void integrateProcessingFunctional<FLOAT_T, descriptors::DESCRIPTOR_3D,
                                   FLOAT_T, descriptors::DESCRIPTOR_3D> (
        BoxProcessingFunctional3D_LL<FLOAT_T, descriptors::DESCRIPTOR_3D,
                                     FLOAT_T, descriptors::DESCRIPTOR_3D>* functional,
        Box3D domain,
        BlockLattice3D<FLOAT_T, descriptors::DESCRIPTOR_3D>& lattice1,
        BlockLattice3D<FLOAT_T, descriptors::DESCRIPTOR_3D>& lattice2, plint level );


template
void applyProcessingFunctional<FLOAT_T, descriptors::DESCRIPTOR_3D> (
        LatticeBoxProcessingFunctional3D<FLOAT_T,descriptors::DESCRIPTOR_3D>* functional,
        Box3D domain, std::vector<BlockLattice3D<FLOAT_T,descriptors::DESCRIPTOR_3D>*> lattices );
template
void integrateProcessingFunctional<FLOAT_T, descriptors::DESCRIPTOR_3D> (
        LatticeBoxProcessingFunctional3D<FLOAT_T, descriptors::DESCRIPTOR_3D>* functional,
        Box3D domain, std::vector<BlockLattice3D<FLOAT_T, descriptors::DESCRIPTOR_3D>*> lattices, plint level );


/* *************** Dotted Data Processor functionals ***************** */

template
void applyProcessingFunctional<FLOAT_T, descriptors::DESCRIPTOR_3D> (
        DotProcessingFunctional3D_L<FLOAT_T,descriptors::DESCRIPTOR_3D>* functional,
        DotList3D const& dotList, BlockLattice3D<FLOAT_T,descriptors::DESCRIPTOR_3D>& lattice );
template
void integrateProcessingFunctional<FLOAT_T, descriptors::DESCRIPTOR_3D> (
        DotProcessingFunctional3D_L<FLOAT_T, descriptors::DESCRIPTOR_3D>* functional,
        DotList3D const& dotList, BlockLattice3D<FLOAT_T, descriptors::DESCRIPTOR_3D>& lattice, plint level );

template
void applyProcessingFunctional<FLOAT_T, descriptors::DESCRIPTOR_3D,
                               FLOAT_T, descriptors::DESCRIPTOR_3D> (
        DotProcessingFunctional3D_LL<FLOAT_T,descriptors::DESCRIPTOR_3D,
                                     FLOAT_T,descriptors::DESCRIPTOR_3D>* functional,
        DotList3D const& dotList,
        BlockLattice3D<FLOAT_T,descriptors::DESCRIPTOR_3D>& lattice1,
        BlockLattice3D<FLOAT_T,descriptors::DESCRIPTOR_3D>& lattice2 );
template
void integrateProcessingFunctional<FLOAT_T, descriptors::DESCRIPTOR_3D,
                                   FLOAT_T, descriptors::DESCRIPTOR_3D> (
        DotProcessingFunctional3D_LL<FLOAT_T, descriptors::DESCRIPTOR_3D,
                                     FLOAT_T, descriptors::DESCRIPTOR_3D>* functional,
        DotList3D const& dotList,
        BlockLattice3D<FLOAT_T, descriptors::DESCRIPTOR_3D>& lattice1,
        BlockLattice3D<FLOAT_T, descriptors::DESCRIPTOR_3D>& lattice2, plint level );


template
void applyProcessingFunctional<FLOAT_T, descriptors::DESCRIPTOR_3D> (
        LatticeDotProcessingFunctional3D<FLOAT_T,descriptors::DESCRIPTOR_3D>* functional,
        DotList3D const& dotList, std::vector<BlockLattice3D<FLOAT_T,descriptors::DESCRIPTOR_3D>*> lattices );
template
void integrateProcessingFunctional<FLOAT_T, descriptors::DESCRIPTOR_3D> (
        LatticeDotProcessingFunctional3D<FLOAT_T, descriptors::DESCRIPTOR_3D>* functional,
        DotList3D const& dotList, std::vector<BlockLattice3D<FLOAT_T, descriptors::DESCRIPTOR_3D>*> lattices, plint level );


/* *************** Bounded Boxed Data Processor functionals ****************** */

template
void applyProcessingFunctional<FLOAT_T, descriptors::DESCRIPTOR_3D> (
        BoundedBoxProcessingFunctional3D_L<FLOAT_T,descriptors::DESCRIPTOR_3D>* functional,
        Box3D domain, BlockLattice3D<FLOAT_T,descriptors::DESCRIPTOR_3D>& lattice, plint boundaryWidth );
template
void integrateProcessingFunctional<FLOAT_T, descriptors::DESCRIPTOR_3D> (
        BoundedBoxProcessingFunctional3D_L<FLOAT_T, descriptors::DESCRIPTOR_3D>* functional,
        Box3D domain, BlockLattice3D<FLOAT_T, descriptors::DESCRIPTOR_3D>& lattice, plint boundaryWidth, plint level );

template
void applyProcessingFunctional<FLOAT_T, descriptors::DESCRIPTOR_3D,
                               FLOAT_T, descriptors::DESCRIPTOR_3D> (
        BoundedBoxProcessingFunctional3D_LL <
            FLOAT_T,descriptors::DESCRIPTOR_3D,
            FLOAT_T,descriptors::DESCRIPTOR_3D >* functional,
        Box3D domain,
        BlockLattice3D<FLOAT_T,descriptors::DESCRIPTOR_3D>& lattice1,
        BlockLattice3D<FLOAT_T,descriptors::DESCRIPTOR_3D>& lattice2,
        plint boundaryWidth );
template
void integrateProcessingFunctional< FLOAT_T, descriptors::DESCRIPTOR_3D,
                                    FLOAT_T, descriptors::DESCRIPTOR_3D > (
        BoundedBoxProcessingFunctional3D_LL <
            FLOAT_T, descriptors::DESCRIPTOR_3D,
            FLOAT_T, descriptors::DESCRIPTOR_3D >* functional,
        Box3D domain,
        BlockLattice3D<FLOAT_T, descriptors::DESCRIPTOR_3D>& lattice1,
        BlockLattice3D<FLOAT_T, descriptors::DESCRIPTOR_3D>& lattice2,
        plint boundaryWidth, plint level );

template
void applyProcessingFunctional<FLOAT_T, descriptors::DESCRIPTOR_3D> (
        BoundedLatticeBoxProcessingFunctional3D<FLOAT_T,descriptors::DESCRIPTOR_3D>* functional,
        Box3D domain, std::vector<BlockLattice3D<FLOAT_T,descriptors::DESCRIPTOR_3D>*> lattices,
        plint boundaryWidth);
template
void integrateProcessingFunctional<FLOAT_T, descriptors::DESCRIPTOR_3D> (
        BoundedLatticeBoxProcessingFunctional3D<FLOAT_T, descriptors::DESCRIPTOR_3D>* functional,
        Box3D domain, std::vector<BlockLattice3D<FLOAT_T, descriptors::DESCRIPTOR_3D>*> lattices,
        plint boundaryWidth, plint level );

}  // namespace plb

#endif  // COMPILE_3D
