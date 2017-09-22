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

#ifndef CO_PROCESSOR_COMMUNICATION_3D_HH
#define CO_PROCESSOR_COMMUNICATION_3D_HH

#include "core/globalDefs.h"
#include "coProcessors/coProcessorCommunication3D.h"
#include "io/parallelIO.h"
#include <iomanip>

namespace plb {

inline std::vector<Box3D> generateSurfaces(Box3D bbox, plint envelopeWidth)
{
    std::vector<Box3D> surfaces;
    PLB_ASSERT( bbox.getNx()>=2 );
    PLB_ASSERT( bbox.getNy()>=2 );
    PLB_ASSERT( bbox.getNz()>=2 );

    plint x0 = bbox.x0;
    plint x1 = bbox.x1;
    plint y0 = bbox.y0;
    plint y1 = bbox.y1;
    plint z0 = bbox.z0;
    plint z1 = bbox.z1;

    //plint ew = envelopeWidth;
    //surfaces.push_back(Box3D(x0,      x1,      y0,      y1,      z0,      z0+ew-1));
    //surfaces.push_back(Box3D(x0,      x1,      y0,      y1,      z1-ew+1, z1));
    //surfaces.push_back(Box3D(x0,      x1,      y0,      y0+ew-1, z0+ew,   z1-ew));
    //surfaces.push_back(Box3D(x0,      x1,      y1-ew+1, y1,      z0+ew,   z1-ew));
    //surfaces.push_back(Box3D(x0,      x0+ew-1, y0+ew,   y1-ew,   z0+ew,   z1-ew));
    //surfaces.push_back(Box3D(x1-ew+1, x1,      y0+ew,   y1-ew,   z0+ew,   z1-ew));

    surfaces.push_back(Box3D(x0,      x1,      y0,      y1,      z0,      z0));
    surfaces.push_back(Box3D(x0,      x1,      y0,      y1,      z1,      z1));
    surfaces.push_back(Box3D(x0,      x1,      y0,      y0,      z0,      z1));
    surfaces.push_back(Box3D(x0,      x1,      y1,      y1,      z0,      z1));
    surfaces.push_back(Box3D(x0,      x0,      y0,      y1,      z0,      z1));
    surfaces.push_back(Box3D(x1,      x1,      y0,      y1,      z0,      z1));

    //surfaces.push_back(bbox);

    return surfaces;
}

template<typename T, template<typename U> class Descriptor>
void transferToCoProcessors(MultiBlockLattice3D<T,Descriptor>& lattice, plint envelopeWidth)
{
    MultiBlockManagement3D const& management = lattice.getMultiBlockManagement();
    ThreadAttribution const& threadAttribution = management.getThreadAttribution();

    std::vector<char> data;
    for (pluint iBlock=0; iBlock<management.getLocalInfo().getBlocks().size(); ++iBlock) {
        plint blockId = management.getLocalInfo().getBlocks()[iBlock];
        plint handle = threadAttribution.getCoProcessorHandle(blockId);
        if (handle>=0) {
            BlockLattice3D<T,Descriptor>& component = lattice.getComponent(blockId);
            std::vector<Box3D> surfaces = generateSurfaces(component.getBoundingBox(), envelopeWidth);
            for (pluint iBox=0; iBox<surfaces.size(); ++iBox) {
                data.clear();
                component.getDataTransfer().send (
                        surfaces[iBox], data, modif::staticVariables );
                global::defaultCoProcessor3D<T>().send (
                        handle, surfaces[iBox], data );
            }
        }
    }
}

template<typename T, template<typename U> class Descriptor>
void transferToCoProcessors(MultiBlockLattice3D<T,Descriptor>& lattice)
{
    MultiBlockManagement3D const& management = lattice.getMultiBlockManagement();
    ThreadAttribution const& threadAttribution = management.getThreadAttribution();

    std::vector<char> data;
    for (pluint iBlock=0; iBlock<management.getLocalInfo().getBlocks().size(); ++iBlock) {
        plint blockId = management.getLocalInfo().getBlocks()[iBlock];
        plint handle = threadAttribution.getCoProcessorHandle(blockId);
        if (handle>=0) {
             BlockLattice3D<T,Descriptor>& component = lattice.getComponent(blockId);
             component.getDataTransfer().send (
                     component.getBoundingBox(), data, modif::staticVariables );
             global::defaultCoProcessor3D<T>().send (
                     handle, component.getBoundingBox(), data );
        }
    }

}

template<typename T, template<typename U> class Descriptor>
void transferFromCoProcessors(MultiBlockLattice3D<T,Descriptor>& lattice, plint envelopeWidth)
{
    MultiBlockManagement3D const& management = lattice.getMultiBlockManagement();
    ThreadAttribution const& threadAttribution = management.getThreadAttribution();

    std::vector<char> data;
    for (pluint iBlock=0; iBlock<management.getLocalInfo().getBlocks().size(); ++iBlock) {
        plint blockId = management.getLocalInfo().getBlocks()[iBlock];
        plint handle = threadAttribution.getCoProcessorHandle(blockId);
        if (handle>=0) {
            BlockLattice3D<T,Descriptor>& component = lattice.getComponent(blockId);
            Box3D reducedDomain = component.getBoundingBox().enlarge(-envelopeWidth);
            std::vector<Box3D> surfaces = generateSurfaces(reducedDomain, envelopeWidth);
            for (pluint iBox=0; iBox<surfaces.size(); ++iBox) {
                data.clear();
                global::defaultCoProcessor3D<T>().receive (
                        handle, surfaces[iBox], data );
                component.getDataTransfer().receive (
                        surfaces[iBox], data, modif::staticVariables );
            }
        }
    }
}

template<typename T, template<typename U> class Descriptor>
void transferFromCoProcessors(MultiBlockLattice3D<T,Descriptor>& lattice)
{
    MultiBlockManagement3D const& management = lattice.getMultiBlockManagement();
    ThreadAttribution const& threadAttribution = management.getThreadAttribution();

    std::vector<char> data;
    for (pluint iBlock=0; iBlock<management.getLocalInfo().getBlocks().size(); ++iBlock) {
        plint blockId = management.getLocalInfo().getBlocks()[iBlock];
        plint handle = threadAttribution.getCoProcessorHandle(blockId);
        if (handle>=0) {
             BlockLattice3D<T,Descriptor>& component = lattice.getComponent(blockId);
             global::defaultCoProcessor3D<T>().receive (
                     handle, component.getBoundingBox(), data );
             component.getDataTransfer().receive (
                     component.getBoundingBox(), data, modif::staticVariables );
        }
    }
}

template<typename T, template<typename U> class Descriptor>
void initiateCoProcessors( MultiBlockLattice3D<T,Descriptor>& lattice,
                           plint dynamicsId, bool printInfo )
{
    std::map<plint,PureDynamics<T> > dynamicsPattern
        = identifyBlocksWithPureDynamics(lattice, dynamicsId);

    if (printInfo) {
        typename std::map<plint,PureDynamics<T> >::const_iterator it = dynamicsPattern.begin();
        plint i=0;
        for (; it != dynamicsPattern.end(); ++it, ++i) {
            plint blockId = it->first;
            bool isPure = it->second.isPure;
            T omega = it->second.omega;
            if (isPure) {
                pcout << std::setw(4) << blockId << ": (" << std::setw(8) << omega << "); ";
            }
            else {
                pcout << std::setw(4) << blockId << ";       ";
            }
            if (i%6==0) pcout << std::endl;
        }
        pcout << std::endl;
    }

    MultiBlockManagement3D const& management = lattice.getMultiBlockManagement();
    std::map<plint, int> coprocessors;
    for (pluint iBlock=0; iBlock<management.getLocalInfo().getBlocks().size(); ++iBlock) {
        plint blockId = management.getLocalInfo().getBlocks()[iBlock];

        typename std::map<plint,PureDynamics<T> >::const_iterator itPure = dynamicsPattern.find(blockId);
        PLB_ASSERT( itPure!=dynamicsPattern.end() );
        bool isPure = itPure->second.isPure;
        T omega = itPure->second.omega;
        if (isPure) {
            SmartBulk3D smartBulk(management, blockId);
            Box3D domain(smartBulk.computeEnvelope());
            int handle;
            global::defaultCoProcessor3D<T>().addDomain (
                    domain.getNx(), domain.getNy(), domain.getNz(), omega, handle );
            coprocessors.insert(std::pair<plint,int>(blockId, handle));
            if (printInfo) {
                pcout << "Le domaine suivant est execute par l accelerateur: ("
                      << domain.x0 << ", " << domain.x1 << "; "
                      << domain.y0 << ", " << domain.y1 << "; "
                      << domain.z0 << ", " << domain.z1 << ")" << std::endl;
            }
        }
        else {
            SmartBulk3D smartBulk(management, blockId);
            Box3D domain(smartBulk.computeEnvelope());
            coprocessors.insert(std::pair<plint,int>(blockId, -1));
            pcout << "Le domaine suivant est execute par Palabos: ("
                  << domain.x0 << ", " << domain.x1 << "; "
                  << domain.y0 << ", " << domain.y1 << "; "
                  << domain.z0 << ", " << domain.z1 << ")" << std::endl;
        }
    }
    lattice.setCoProcessors(coprocessors);
}

}  // namespace plb

#endif  // CO_PROCESSOR_COMMUNICATION_3D_HH

