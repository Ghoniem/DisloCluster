/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_NoiseTraits_cpp
#define model_NoiseTraits_cpp

#include <NoiseTraits.h>

namespace model
{

    int NoiseTraitsBase::LittleEndian()
    {
        int num = 1;
        if(*(char *)&num == 1)
        {
            return 1;       //little endian
        }
        else
        {
            return 0;       // big endian
        }
    }

    float NoiseTraitsBase::ReverseFloat(const float& inFloat)
    {
        float retVal;
        char *FloatToConvert = ( char* ) & inFloat;
        char *returnFloat = ( char* ) & retVal;
        
        // swap the bytes into a temporary buffer
        returnFloat[0] = FloatToConvert[3];
        returnFloat[1] = FloatToConvert[2];
        returnFloat[2] = FloatToConvert[1];
        returnFloat[3] = FloatToConvert[0];
        
        return retVal;
    }

    double NoiseTraitsBase::ReverseDouble(const double& inDouble)
    {
        double retVal;
        char *DoubleToConvert = ( char* ) & inDouble;
        char *returnDouble = ( char* ) & retVal;
        
        // swap the bytes into a temporary buffer
        returnDouble[0] = DoubleToConvert[7];
        returnDouble[1] = DoubleToConvert[6];
        returnDouble[2] = DoubleToConvert[5];
        returnDouble[3] = DoubleToConvert[4];
        returnDouble[4] = DoubleToConvert[3];
        returnDouble[5] = DoubleToConvert[2];
        returnDouble[6] = DoubleToConvert[1];
        returnDouble[7] = DoubleToConvert[0];
        
        return retVal;
    }

template <int N>
const typename NoiseTraits<N>::NoiseType& NoiseTraits<N>::fromMatrix(const Eigen::Matrix<REAL_SCALAR,1,N>& m)
{
    return m;
}

template <int N>
typename NoiseTraits<N>::NoiseType NoiseTraits<N>::Zero()
{
    return NoiseType::Zero();
}

template <int N>
typename NoiseTraits<N>::NoiseType NoiseTraits<N>::squared(const NoiseType& m)
{
    return (m.array()*m.array()).matrix();
}


//template struct NoiseTraits<1>;
template struct NoiseTraits<2>;

}
#endif

