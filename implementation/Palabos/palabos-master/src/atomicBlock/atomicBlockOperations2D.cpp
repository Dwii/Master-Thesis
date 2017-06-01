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
 * Operations on the 2D multiblock -- implementation.
 */

#include "atomicBlock/atomicBlockOperations2D.h"
#include "core/plbDebug.h"

namespace plb {

void executeDataProcessor( DataProcessorGenerator2D const& generator,
                           std::vector<AtomicBlock2D*> objects )
{
    DataProcessor2D* processor = generator.generate(objects);
    processor -> process();
    delete processor;
}

void executeDataProcessor( DataProcessorGenerator2D const& generator,
                           AtomicBlock2D& object )
{
    std::vector<AtomicBlock2D*> objects(1);
    objects[0] = &object;
    executeDataProcessor(generator, objects);
}

void executeDataProcessor( DataProcessorGenerator2D const& generator,
                           AtomicBlock2D& object1, AtomicBlock2D& object2 )
{
    std::vector<AtomicBlock2D*> objects(2);
    objects[0] = &object1;
    objects[1] = &object2;
    executeDataProcessor(generator, objects);
}


void executeDataProcessor( ReductiveDataProcessorGenerator2D& generator,
                           std::vector<AtomicBlock2D*> objects )
{
    DataProcessor2D* processor = generator.generate(objects);
    processor -> process();
    delete processor;
}

void executeDataProcessor( ReductiveDataProcessorGenerator2D& generator,
                           AtomicBlock2D& object )
{
    std::vector<AtomicBlock2D*> objects(1);
    objects[0] = &object;
    executeDataProcessor(generator, objects);
}

void executeDataProcessor( ReductiveDataProcessorGenerator2D& generator,
                           AtomicBlock2D& object1, AtomicBlock2D& object2 )
{
    std::vector<AtomicBlock2D*> objects(2);
    objects[0] = &object1;
    objects[1] = &object2;
    executeDataProcessor(generator, objects);
}


void addInternalProcessor( DataProcessorGenerator2D const& generator,
                           std::vector<AtomicBlock2D*> objects, plint level )
{
    PLB_PRECONDITION( !objects.empty() );
    objects[0] -> integrateDataProcessor(generator.generate(objects), level);
}

void addInternalProcessor( DataProcessorGenerator2D const& generator,
                           AtomicBlock2D& object, plint level )
{
    std::vector<AtomicBlock2D*> objects(1);
    objects[0] = &object;
    addInternalProcessor(generator, objects, level);
}

void addInternalProcessor( DataProcessorGenerator2D const& generator,
                           AtomicBlock2D& object1, AtomicBlock2D& object2,
                           plint level )
{
    std::vector<AtomicBlock2D*> objects(2);
    objects[0] = &object1;
    objects[1] = &object2;
    addInternalProcessor(generator, objects, level);
}

} // namespace plb
