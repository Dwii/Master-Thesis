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
 * Groups all the include files for the 2D multiBlock.
 */

#include "multiGrid/multiScale.h"
#include "multiGrid/multiGridOperations2D.h"
#include "multiGrid/multiGridDataField2D.h"
#include "multiGrid/multiGrid2D.h"
#include "multiGrid/gridRefinementDynamics.h"
#include "multiGrid/multiGridLattice2D.h"
#include "multiGrid/defaultMultiGridPolicy2D.h"
#include "multiGrid/gridRefinement.h"
#include "multiGrid/coarseGridProcessors2D.h"
#include "multiGrid/fineGridProcessors2D.h"
#include "multiGrid/multiGridParameterManager.h"
#include "multiGrid/multiGridGenerator2D.h"
#include "multiGrid/multiGridManagement2D.h"
#include "multiGrid/dynamicsGenerators.h"
#include "multiGrid/multiGridDataAnalysisWrapper2D.h"
#include "multiGrid/multiGridDataProcessorWrapper2D.h"
#include "multiGrid/svgWriter.h"
#include "multiGrid/gridConversion2D.h"
#include "multiGrid/parallelizer2D.h"

