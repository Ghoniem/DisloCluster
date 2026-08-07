/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_AnalyticalSolidSolutionNoise_h
#define model_AnalyticalSolidSolutionNoise_h

#include <cmath>
#include <random>

#include <Eigen/Dense>

#include <PolycrystallineMaterialBase.h>
#include <NoiseTraits.h>
#include <GlidePlaneNoiseBase.h>

namespace model
{

    struct AnalyticalSolidSolutionNoise: public GlidePlaneNoiseBase<2>
    {
        typedef typename NoiseTraits<2>::REAL_SCALAR REAL_SCALAR;
        typedef typename NoiseTraits<2>::COMPLEX COMPLEX;
        typedef typename NoiseTraits<2>::GridSizeType GridSizeType;
        typedef typename NoiseTraitsBase::GridSpacingType GridSpacingType;
        typedef typename NoiseTraits<2>::NoiseType NoiseType;
        typedef typename NoiseTraits<2>::NoiseContainerType NoiseContainerType;
                
        const REAL_SCALAR a;
        const REAL_SCALAR a_cai;
        const double stressPrefactor;
        
//        static REAL_SCALAR Wk_Cai(REAL_SCALAR kx, REAL_SCALAR ky, REAL_SCALAR kz, REAL_SCALAR a) ;
//        static REAL_SCALAR Wk_Cai_squared(REAL_SCALAR kx, REAL_SCALAR ky, REAL_SCALAR kz, REAL_SCALAR a) ;
//
//        // Cai spreading function
//        static REAL_SCALAR W_Cai(REAL_SCALAR r2, REAL_SCALAR a) ;
// 
//        static REAL_SCALAR W_t_Cai(REAL_SCALAR r2, REAL_SCALAR a) ;
        // normalized auto-correlation function in Fourier space for sigma_xy
        REAL_SCALAR S_xy_k(REAL_SCALAR kx, REAL_SCALAR ky, REAL_SCALAR kz) const;

        // normalized auto-correlation function in Fourier space for sigma_xz
        REAL_SCALAR S_xz_k(REAL_SCALAR kx, REAL_SCALAR ky, REAL_SCALAR kz) const;

        // normalized auto-correlation function in Fourier space for sigma_yz
        REAL_SCALAR S_yz_k(REAL_SCALAR kx, REAL_SCALAR ky, REAL_SCALAR kz) const;

        AnalyticalSolidSolutionNoise(const std::string& tag, const int& seed,
                                    const GridSizeType& gridSize, const GridSpacingType& gridSpacing,
                                     const double& a_in,const double& a_Cai_in,const double& MSSS);
        
        std::array<COMPLEX,2> kCorrelations(const Eigen::Matrix<double,3,1>& kv,const Eigen::Matrix<int,3,1>& kvID) const override;

    };

}
#endif

