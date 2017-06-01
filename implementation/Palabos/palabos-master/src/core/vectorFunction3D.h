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


#ifndef VECTOR_FUNCTION_3D_H
#define VECTOR_FUNCTION_3D_H

#include "core/array.h"
#include "core/functions.h"
#include "core/globalDefs.h"
#include "core/util.h"
#include "latticeBoltzmann/geometricOperationTemplates.h"
#include "multiBlock/multiBlockLattice3D.h"
#include "multiBlock/multiBlockLattice3D.hh"

#include <cmath>

namespace plb {

template<typename T>
class VectorFunction3D {
public:
    virtual ~VectorFunction3D() { }
    virtual Array<T,3> operator()(Array<T,3> const& position) const = 0;
    virtual VectorFunction3D<T>* clone() const = 0;
};

template<typename T>
class ConstantVectorFunction3D : public VectorFunction3D<T> {
public:
    ConstantVectorFunction3D(Array<T,3> const& constantVector_)
        : constantVector(constantVector_)
    { }

    virtual Array<T,3> operator()(Array<T,3> const& position) const
    {
        return constantVector;
    }

    virtual ConstantVectorFunction3D<T>* clone() const
    {
        return new ConstantVectorFunction3D<T>(*this);
    }
private:
    Array<T,3> constantVector;
};

template<typename T>
class IdentityVectorFunction3D : public VectorFunction3D<T> {
public:
    virtual Array<T,3> operator()(Array<T,3> const& position) const
    {
        return position;
    }

    virtual IdentityVectorFunction3D<T>* clone() const
    {
        return new IdentityVectorFunction3D<T>(*this);
    }
};

template<typename T>
class DiscreteRotationalVelocityFunction3D : public VectorFunction3D<T> {
public:
    DiscreteRotationalVelocityFunction3D(Array<T,3> const& angularVelocity_, Array<T,3> const& pointOnRotationAxis_)
        : angularVelocity(angularVelocity_),
          pointOnRotationAxis(pointOnRotationAxis_)
    {
        normAngularVelocity = norm(angularVelocity);
        if (!util::isZero(normAngularVelocity)) {
            rotationAxisUnitVector = angularVelocity / normAngularVelocity;
        } else {
            angularVelocity = Array<T,3>::zero();
            rotationAxisUnitVector = Array<T,3>((T) 1, (T) 0, (T) 0); // Array<T,3>::zero();
            normAngularVelocity = (T) 0;
        }
    }

    Array<T,3> getAngularVelocity() const
    {
        return angularVelocity;
    }

    // The rotation angle is defined as the norm of the current angular velocity.
    // In other words it is the angle of a discrete rotation for a time step equal to 1.
    T getRotationAngle() const
    {
        return normAngularVelocity;
    }

    virtual Array<T,3> operator()(Array<T,3> const& position) const
    {
        return getDiscreteRotationalVelocity(position, angularVelocity, rotationAxisUnitVector, pointOnRotationAxis);
    }

    virtual DiscreteRotationalVelocityFunction3D<T>* clone() const
    {
        return new DiscreteRotationalVelocityFunction3D<T>(*this);
    }
private:
    Array<T,3> angularVelocity, rotationAxisUnitVector, pointOnRotationAxis;
    T normAngularVelocity;
};

template<typename T>
class DiscreteTranslationalPositionFunction3D : public VectorFunction3D<T> {
public:
    DiscreteTranslationalPositionFunction3D(Array<T,3> const& velocity_)
        : velocity(velocity_)
    { }
    Array<T,3> getVelocity() const { return velocity; }
    virtual Array<T,3> operator()(Array<T,3> const& position) const
    {
        return position + velocity;
    }
    DiscreteTranslationalPositionFunction3D<T>* clone() const {
        return new DiscreteTranslationalPositionFunction3D<T>(*this);
    }
private:
    Array<T,3> velocity;
};

template<typename T>
class DiscreteRotationalPositionFunction3D : public VectorFunction3D<T> {
public:
    DiscreteRotationalPositionFunction3D(Array<T,3> const& angularVelocity_, Array<T,3> const& pointOnRotationAxis_)
        : angularVelocity(angularVelocity_),
          pointOnRotationAxis(pointOnRotationAxis_)
    {
        normAngularVelocity = norm(angularVelocity);
        if (!util::isZero(normAngularVelocity)) {
            rotationAxisUnitVector = angularVelocity / normAngularVelocity;
        } else {
            angularVelocity = Array<T,3>::zero();
            rotationAxisUnitVector = Array<T,3>((T) 1, (T) 0, (T) 0); // Array<T,3>::zero();
            normAngularVelocity = (T) 0;
        }
    }

    Array<T,3> getAngularVelocity() const
    {
        return angularVelocity;
    }

    // The rotation angle is defined as the norm of the current angular velocity.
    // In other words it is the angle of a discrete rotation for a time step equal to 1.
    T getRotationAngle() const
    {
        return normAngularVelocity;
    }

    virtual Array<T,3> operator()(Array<T,3> const& position) const
    {
        return getRotatedPosition(position, angularVelocity, rotationAxisUnitVector, pointOnRotationAxis);
    }

    virtual DiscreteRotationalPositionFunction3D<T>* clone() const
    {
        return new DiscreteRotationalPositionFunction3D<T>(*this);
    }
private:
    Array<T,3> angularVelocity, rotationAxisUnitVector, pointOnRotationAxis;
    T normAngularVelocity;
};

template<typename T>
class ExactRotationalVelocityFunction3D : public VectorFunction3D<T> {
public:
    ExactRotationalVelocityFunction3D(Array<T,3> const& angularVelocity_, Array<T,3> const& pointOnRotationAxis_)
        : angularVelocity(angularVelocity_),
          pointOnRotationAxis(pointOnRotationAxis_)
    {
        normAngularVelocity = norm(angularVelocity);
        if (!util::isZero(normAngularVelocity)) {
            rotationAxisUnitVector = angularVelocity / normAngularVelocity;
        } else {
            angularVelocity = Array<T,3>::zero();
            rotationAxisUnitVector = Array<T,3>((T) 1, (T) 0, (T) 0); // Array<T,3>::zero();
            normAngularVelocity = (T) 0;
        }
    }

    Array<T,3> getAngularVelocity() const
    {
        return angularVelocity;
    }

    // The rotation angle is defined as the norm of the current angular velocity.
    // In other words it is the angle of a discrete rotation for a time step equal to 1.
    T getRotationAngle() const
    {
        return normAngularVelocity;
    }

    virtual Array<T,3> operator()(Array<T,3> const& position) const
    {
        return getExactRotationalVelocity(position, angularVelocity, pointOnRotationAxis);
    }

    virtual ExactRotationalVelocityFunction3D<T>* clone() const
    {
        return new ExactRotationalVelocityFunction3D<T>(*this);
    }
private:
    Array<T,3> angularVelocity, rotationAxisUnitVector, pointOnRotationAxis;
    T normAngularVelocity;
};

template<typename T, template<typename U> class Descriptor> 
class IncreasingDiscreteRotationalVelocityFunction3D : public VectorFunction3D<T> {
public:
    IncreasingDiscreteRotationalVelocityFunction3D(Array<T,3> const& maxAngularVelocity_,
            Array<T,3> const& pointOnRotationAxis_, MultiBlockLattice3D<T,Descriptor> const& lattice_,
            plint tOffset_, plint maxT_)
        : maxAngularVelocity(maxAngularVelocity_),
          pointOnRotationAxis(pointOnRotationAxis_),
          lattice(lattice_),
          tOffset(tOffset_),
          maxT(maxT_)
    {
        T normMaxAngularVelocity = norm(maxAngularVelocity);
        if (!util::isZero(normMaxAngularVelocity)) {
            rotationAxisUnitVector = maxAngularVelocity / normMaxAngularVelocity;
        } else {
            maxAngularVelocity = Array<T,3>::zero();
            rotationAxisUnitVector = Array<T,3>((T) 1, (T) 0, (T) 0); // Array<T,3>::zero();
        }
    }

    Array<T,3> getAngularVelocity() const
    {
        plint t = lattice.getTimeCounter().getTime() + tOffset;
        Array<T,3> angularVelocity = util::sinIncreasingFunction<T>((T) t, (T) maxT) * maxAngularVelocity;
        return angularVelocity;
    }

    // The rotation angle is defined as the norm of the current angular velocity.
    // In other words it is the angle of a discrete rotation for a time step equal to 1.
    T getRotationAngle() const
    {
        T theta = dot(getAngularVelocity(), rotationAxisUnitVector);
        return theta;
    }

    virtual Array<T,3> operator()(Array<T,3> const& position) const
    {
        return getDiscreteRotationalVelocity(position, getAngularVelocity(), rotationAxisUnitVector, pointOnRotationAxis);
    }

    virtual IncreasingDiscreteRotationalVelocityFunction3D<T,Descriptor>* clone() const
    {
        return new IncreasingDiscreteRotationalVelocityFunction3D<T,Descriptor>(*this);
    }
private:
    Array<T,3> maxAngularVelocity, rotationAxisUnitVector, pointOnRotationAxis;
    MultiBlockLattice3D<T,Descriptor> const& lattice;
    plint tOffset, maxT;
};

template<typename T, template<typename U> class Descriptor> 
class IncreasingDiscreteRotationalPositionFunction3D : public VectorFunction3D<T> {
public:
    IncreasingDiscreteRotationalPositionFunction3D(Array<T,3> const& maxAngularVelocity_,
            Array<T,3> const& pointOnRotationAxis_, MultiBlockLattice3D<T,Descriptor> const& lattice_,
            plint tOffset_, plint maxT_)
        : maxAngularVelocity(maxAngularVelocity_),
          pointOnRotationAxis(pointOnRotationAxis_),
          lattice(lattice_),
          tOffset(tOffset_),
          maxT(maxT_)
    {
        T normMaxAngularVelocity = norm(maxAngularVelocity);
        if (!util::isZero(normMaxAngularVelocity)) {
            rotationAxisUnitVector = maxAngularVelocity / normMaxAngularVelocity;
        } else {
            maxAngularVelocity = Array<T,3>::zero();
            rotationAxisUnitVector = Array<T,3>((T) 1, (T) 0, (T) 0); // Array<T,3>::zero();
        }
    }

    Array<T,3> getAngularVelocity() const
    {
        plint t = lattice.getTimeCounter().getTime() + tOffset;
        Array<T,3> angularVelocity = util::sinIncreasingFunction<T>((T) t, (T) maxT) * maxAngularVelocity;
        return angularVelocity;
    }

    // The rotation angle is defined as the norm of the current angular velocity.
    // In other words it is the angle of a discrete rotation for a time step equal to 1.
    T getRotationAngle() const
    {
        T theta = dot(getAngularVelocity(), rotationAxisUnitVector);
        return theta;
    }

    virtual Array<T,3> operator()(Array<T,3> const& position) const
    {
        return getRotatedPosition(position, getAngularVelocity(), rotationAxisUnitVector, pointOnRotationAxis);
    }

    virtual IncreasingDiscreteRotationalPositionFunction3D<T,Descriptor>* clone() const
    {
        return new IncreasingDiscreteRotationalPositionFunction3D<T,Descriptor>(*this);
    }
private:
    Array<T,3> maxAngularVelocity, rotationAxisUnitVector, pointOnRotationAxis;
    MultiBlockLattice3D<T,Descriptor> const& lattice;
    plint tOffset, maxT;
};

template<typename T, template<typename U> class Descriptor> 
class IncreasingExactRotationalVelocityFunction3D : public VectorFunction3D<T> {
public:
    IncreasingExactRotationalVelocityFunction3D(Array<T,3> const& maxAngularVelocity_,
            Array<T,3> const& pointOnRotationAxis_, MultiBlockLattice3D<T,Descriptor> const& lattice_,
            plint tOffset_, plint maxT_)
        : maxAngularVelocity(maxAngularVelocity_),
          pointOnRotationAxis(pointOnRotationAxis_),
          lattice(lattice_),
          tOffset(tOffset_),
          maxT(maxT_)
    {
        T normMaxAngularVelocity = norm(maxAngularVelocity);
        if (!util::isZero(normMaxAngularVelocity)) {
            rotationAxisUnitVector = maxAngularVelocity / normMaxAngularVelocity;
        } else {
            maxAngularVelocity = Array<T,3>::zero();
            rotationAxisUnitVector = Array<T,3>((T) 1, (T) 0, (T) 0); // Array<T,3>::zero();
        }
    }

    Array<T,3> getAngularVelocity() const
    {
        plint t = lattice.getTimeCounter().getTime() + tOffset;
        Array<T,3> angularVelocity = util::sinIncreasingFunction<T>((T) t, (T) maxT) * maxAngularVelocity;
        return angularVelocity;
    }

    // The rotation angle is defined as the norm of the current angular velocity.
    // In other words it is the angle of a discrete rotation for a time step equal to 1.
    T getRotationAngle() const
    {
        T theta = dot(getAngularVelocity(), rotationAxisUnitVector);
        return theta;
    }

    virtual Array<T,3> operator()(Array<T,3> const& position) const
    {
        return getExactRotationalVelocity(position, getAngularVelocity(), pointOnRotationAxis);
    }

    virtual IncreasingExactRotationalVelocityFunction3D<T,Descriptor>* clone() const
    {
        return new IncreasingExactRotationalVelocityFunction3D<T,Descriptor>(*this);
    }
private:
    Array<T,3> maxAngularVelocity, rotationAxisUnitVector, pointOnRotationAxis;
    MultiBlockLattice3D<T,Descriptor> const& lattice;
    plint tOffset, maxT;
};

template<typename T, template<typename U> class Descriptor> 
class HarmonicVectorFunction3D : public VectorFunction3D<T> {
public:
    HarmonicVectorFunction3D(Array<T,3> const& vectorAmplitude_, T angularFrequency_, T phase_,
            MultiBlockLattice3D<T,Descriptor> const& lattice_, plint tOffset_)
        : vectorAmplitude(vectorAmplitude_),
          angularFrequency(angularFrequency_),
          phase(phase_),
          lattice(lattice_),
          tOffset(tOffset_)
    { }

    virtual Array<T,3> operator()(Array<T,3> const& position) const
    {
        plint t = lattice.getTimeCounter().getTime() + tOffset;
        return std::cos(angularFrequency * (T) t + phase) * vectorAmplitude;
    }

    virtual HarmonicVectorFunction3D<T,Descriptor>* clone() const
    {
        return new HarmonicVectorFunction3D<T,Descriptor>(*this);
    }
private:
    Array<T,3> vectorAmplitude;
    T angularFrequency, phase;
    MultiBlockLattice3D<T,Descriptor> const& lattice;
    plint tOffset;
};

} // namespace plb

#endif  // VECTOR_FUNCTION_3D_H
