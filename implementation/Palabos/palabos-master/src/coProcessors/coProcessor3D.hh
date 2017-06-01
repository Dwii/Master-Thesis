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

#ifndef CO_PROCESSOR_3D_HH
#define CO_PROCESSOR_3D_HH

#include "coProcessors/coProcessor3D.h"
#include "basicDynamics/isoThermalDynamics.h"
#include <limits>

namespace plb {
    
    template<typename T>
    int D3Q19ExampleCoProcessor3D<T>::addDomain(plint nx, plint ny, plint nz, T omega, int& domainHandle)
    {
        PLB_ASSERT( (int)domains.size() < std::numeric_limits<int>::max() );
        domainHandle = (int)domains.size();
        Dynamics<T,descriptors::D3Q19Descriptor>* dynamics =
        new BGKdynamics<T,descriptors::D3Q19Descriptor>(omega);
        domains.insert (
                        std::pair<int, BlockLattice3D<T,descriptors::D3Q19Descriptor> > (
                                                                                         domainHandle, BlockLattice3D<T,descriptors::D3Q19Descriptor>(nx, ny, nz, dynamics) ) );
        return 1; // Success.
    }
    
    template<typename T>
    int D3Q19ExampleCoProcessor3D<T>::send(int domainHandle, Box3D const& subDomain, std::vector<char> const& data)
    {
        typename std::map<int, BlockLattice3D<T,descriptors::D3Q19Descriptor> >::iterator it
        = domains.find(domainHandle);
        PLB_ASSERT( it != domains.end() );
        BlockLattice3D<T,descriptors::D3Q19Descriptor>& lattice = it->second;
        lattice.getDataTransfer().receive(subDomain, data, modif::staticVariables);
        return 1; // Success.
    }
    
    template<typename T>
    int D3Q19ExampleCoProcessor3D<T>::receive(int domainHandle, Box3D const& subDomain, std::vector<char>& data) const
    {
        typename std::map<int, BlockLattice3D<T,descriptors::D3Q19Descriptor> >::const_iterator it
        = domains.find(domainHandle);
        PLB_ASSERT( it != domains.end() );
        BlockLattice3D<T,descriptors::D3Q19Descriptor> const& lattice = it->second;
        lattice.getDataTransfer().send(subDomain, data, modif::staticVariables);
        return 1; // Success.
    }
    
    template<typename T>
    int D3Q19ExampleCoProcessor3D<T>::collideAndStream(int domainHandle)
    {
        typename std::map<int, BlockLattice3D<T,descriptors::D3Q19Descriptor> >::iterator it
        = domains.find(domainHandle);
        PLB_ASSERT( it != domains.end() );
        BlockLattice3D<T,descriptors::D3Q19Descriptor>& lattice = it->second;
        lattice.collideAndStream(lattice.getBoundingBox());
        return 1; // Success.
    }
    
    template<typename T>
    D3Q19CudaCoProcessor3D<T>::D3Q19CudaCoProcessor3D()
    {
        
    }
    
    template<typename T>
    D3Q19CudaCoProcessor3D<T>::~D3Q19CudaCoProcessor3D()
    {
        
    }
    
    template<typename T>
    int D3Q19CudaCoProcessor3D<T>::addDomain(plint nx, plint ny, plint nz, T omega, int& domainHandle)
    //int D3Q19CudaCoProcessor3D<T>::addDomain(Box3D const& domain, T omega)
    {
        return 1;
    }
    
    template<typename T>
    int D3Q19CudaCoProcessor3D<T>::send(int domainHandle, Box3D const& subDomain, std::vector<char> const& data)
    {
        return 1;
    }
    template<typename T>
    int D3Q19CudaCoProcessor3D<T>::receive(int domainHandle, Box3D const& subDomain, std::vector<char>& data) const
    {
        return 1;
    }
    
    template<typename T>
    int D3Q19CudaCoProcessor3D<T>::collideAndStream(int domainHandle)
    {
        return 1;
    }
    
}  // namespace plb

#endif  // CO_PROCESSOR_3D_HH

