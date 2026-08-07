/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_NoiseTraits_H
#define model_NoiseTraits_H

#include <cmath>
#include <vector>
#include <Eigen/Dense>

namespace model
{
    struct NoiseTraitsBase
    {
        typedef double REAL_SCALAR;
        typedef std::complex<double> COMPLEX;
        typedef Eigen::Array<int,3,1> GridSizeType;
        typedef Eigen::Array<double,3,1> GridSpacingType;
        
        static int LittleEndian();
        static float ReverseFloat(const float& inFloat);
        static double ReverseDouble(const double& inDouble);
    };

    template <int N>
    struct NoiseTraits
    {
        typedef typename NoiseTraitsBase::REAL_SCALAR REAL_SCALAR;
        typedef typename NoiseTraitsBase::COMPLEX COMPLEX;
        typedef typename NoiseTraitsBase::GridSizeType GridSizeType;
        typedef Eigen::Matrix<REAL_SCALAR,1,N> NoiseType;
        typedef std::vector<NoiseType> NoiseContainerType;
        static const NoiseType& fromMatrix(const Eigen::Matrix<REAL_SCALAR,1,N>& m);
        static NoiseType Zero();
        static NoiseType squared(const NoiseType& m);
    };

    template<>
    struct NoiseTraits<1>
    {
        typedef typename NoiseTraitsBase::REAL_SCALAR REAL_SCALAR;
        typedef typename NoiseTraitsBase::COMPLEX COMPLEX;
        typedef typename NoiseTraitsBase::GridSizeType GridSizeType;
        typedef REAL_SCALAR NoiseType;
        typedef std::vector<NoiseType> NoiseContainerType;
        static const NoiseType& fromMatrix(const Eigen::Matrix<REAL_SCALAR,1,1>& m){return m(0,0);};
        static NoiseType Zero(){return REAL_SCALAR(0.0);};
        static NoiseType squared(const NoiseType& m){return m*m;};
    };
}
#endif
