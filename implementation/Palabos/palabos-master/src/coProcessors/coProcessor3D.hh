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

#define USE_KERNEL_COPY

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

#ifndef PLB_NO_CUDA
    
    /**** Cuda coProcessor ****/

    #define PBIDX(x, y, z, nx, ny, nz) ( ( x*(nz*ny) + y*(nz) + z) * 19)

    template<typename T>
    D3Q19CudaCoProcessor3D<T>::D3Q19CudaCoProcessor3D() 
    {
        lbm_sim = NULL;
        lattices = NULL;
    }
    
    template<typename T>
    D3Q19CudaCoProcessor3D<T>::~D3Q19CudaCoProcessor3D()
    {
        if (lbm_sim) {
            lbm_simulation_destroy(lbm_sim);
        }

        lbm_lattices_destroy(lattices);
    }
    
    template<typename T>
    int D3Q19CudaCoProcessor3D<T>::addDomain(plint nx, plint ny, plint nz, T omega, int& domainHandle)
    //int D3Q19CudaCoProcessor3D<T>::addDomain(Box3D const& domain, T omega)
    {
        this->nx = nx;
        this->ny = ny;
        this->nz = nz;
        this->nl = nx * ny * nz;

        if (lbm_sim) {
            lbm_simulation_destroy(lbm_sim);
        }
        lbm_sim = lbm_simulation_create(nx, ny, nz, omega);
        
        if (lattices) {
            lbm_lattices_destroy(lattices);
        }
        lattices = lbm_lattices_create(nl);

        return 1;
    }

    template<typename T>
    int D3Q19CudaCoProcessor3D<T>::send(int domainHandle, Box3D const& subDomain, std::vector<char> const& data)
    {
        lbm_box_3d subdomain = {subDomain.x0, subDomain.x1, subDomain.y0, subDomain.y1, subDomain.z0, subDomain.z1};

#ifdef USE_KERNEL_COPY
        lbm_write_palabos_subdomain(lbm_sim, (double *)data.data(), subdomain);
#else
        long snx = std::abs(subDomain.x0 - subDomain.x1) + 1;
        long sny = std::abs(subDomain.y0 - subDomain.y1) + 1;
        long snz = std::abs(subDomain.z0 - subDomain.z1) + 1;

        T const* pal_lattices = (const double *)&data[0];

        for (plint x = subDomain.x0; x <= subDomain.x1; x++) {
            for (plint y = subDomain.y0; y <= subDomain.y1; y++) {
                for (plint z = subDomain.z0; z <= subDomain.z1; z++) {
                    size_t gi = IDX(x,y,z,nx,ny,nz);

                    size_t px = x - subDomain.x0;
                    size_t py = y - subDomain.y0;
                    size_t pz = z - subDomain.z0;
                    size_t pbi = PBIDX(px,py,pz,snx,sny,snz); // base index

                    lattices->c [gi] = pal_lattices[pbi +  0] + 1./3 ;
                    lattices->w [gi] = pal_lattices[pbi +  1] + 1./18;
                    lattices->s [gi] = pal_lattices[pbi +  2] + 1./18;
                    lattices->bc[gi] = pal_lattices[pbi +  3] + 1./18;
                    lattices->sw[gi] = pal_lattices[pbi +  4] + 1./36;
                    lattices->nw[gi] = pal_lattices[pbi +  5] + 1./36;
                    lattices->bw[gi] = pal_lattices[pbi +  6] + 1./36;
                    lattices->tw[gi] = pal_lattices[pbi +  7] + 1./36;
                    lattices->bs[gi] = pal_lattices[pbi +  8] + 1./36;
                    lattices->ts[gi] = pal_lattices[pbi +  9] + 1./36;
                    lattices->e [gi] = pal_lattices[pbi + 10] + 1./18;
                    lattices->n [gi] = pal_lattices[pbi + 11] + 1./18;
                    lattices->tc[gi] = pal_lattices[pbi + 12] + 1./18;
                    lattices->ne[gi] = pal_lattices[pbi + 13] + 1./36;
                    lattices->se[gi] = pal_lattices[pbi + 14] + 1./36;
                    lattices->te[gi] = pal_lattices[pbi + 15] + 1./36;
                    lattices->be[gi] = pal_lattices[pbi + 16] + 1./36;
                    lattices->tn[gi] = pal_lattices[pbi + 17] + 1./36;
                    lattices->bn[gi] = pal_lattices[pbi + 18] + 1./36;
                }
            }
        }

        lbm_lattices_write_subdomain(lbm_sim, lattices, subdomain);

#endif

        return 1;
    }
    template<typename T>
    int D3Q19CudaCoProcessor3D<T>::receive(int domainHandle, Box3D const& subDomain, std::vector<char>& data) const
    {
        long snx = std::abs(subDomain.x0 - subDomain.x1) + 1;
        long sny = std::abs(subDomain.y0 - subDomain.y1) + 1;
        long snz = std::abs(subDomain.z0 - subDomain.z1) + 1;
        data.resize(snx*sny*snz*19*sizeof(T));

        lbm_box_3d subdomain = {subDomain.x0, subDomain.x1, subDomain.y0, subDomain.y1, subDomain.z0, subDomain.z1};

#ifdef USE_KERNEL_COPY
        lbm_read_palabos_subdomain(lbm_sim, (double *)data.data(), subdomain);
#else
        T* pal_lattices = (double *)&data[0];
 
        lbm_lattices_read_subdomain(lbm_sim, lattices, subdomain);
        
        for (plint x = subDomain.x0; x <= subDomain.x1; x++) {
            for (plint y = subDomain.y0; y <= subDomain.y1; y++) {
                for (plint z = subDomain.z0; z <= subDomain.z1; z++) {
                    size_t gi = IDX(x,y,z,nx,ny,nz);
                    
                    size_t px = x - subDomain.x0;
                    size_t py = y - subDomain.y0;
                    size_t pz = z - subDomain.z0;
                    size_t pbi = PBIDX(px,py,pz,snx,sny,snz); // palabos data base index

                    pal_lattices[pbi +  0] = lattices->c [gi] - 1./3 ;
                    pal_lattices[pbi +  1] = lattices->w [gi] - 1./18;
                    pal_lattices[pbi +  2] = lattices->s [gi] - 1./18;
                    pal_lattices[pbi +  3] = lattices->bc[gi] - 1./18;
                    pal_lattices[pbi +  4] = lattices->sw[gi] - 1./36;
                    pal_lattices[pbi +  5] = lattices->nw[gi] - 1./36;
                    pal_lattices[pbi +  6] = lattices->bw[gi] - 1./36;
                    pal_lattices[pbi +  7] = lattices->tw[gi] - 1./36;
                    pal_lattices[pbi +  8] = lattices->bs[gi] - 1./36;
                    pal_lattices[pbi +  9] = lattices->ts[gi] - 1./36;
                    pal_lattices[pbi + 10] = lattices->e [gi] - 1./18;
                    pal_lattices[pbi + 11] = lattices->n [gi] - 1./18;
                    pal_lattices[pbi + 12] = lattices->tc[gi] - 1./18;
                    pal_lattices[pbi + 13] = lattices->ne[gi] - 1./36;
                    pal_lattices[pbi + 14] = lattices->se[gi] - 1./36;
                    pal_lattices[pbi + 15] = lattices->te[gi] - 1./36;
                    pal_lattices[pbi + 16] = lattices->be[gi] - 1./36;
                    pal_lattices[pbi + 17] = lattices->tn[gi] - 1./36;
                    pal_lattices[pbi + 18] = lattices->bn[gi] - 1./36;
                }
            }
        }
#endif

        return 1;
    }

    template<typename T>
    int D3Q19CudaCoProcessor3D<T>::collideAndStream(int domainHandle)
    {
        lbm_simulation_update(lbm_sim);
        return 1;
    }
#endif
    
}  // namespace plb

#endif  // CO_PROCESSOR_3D_HH

