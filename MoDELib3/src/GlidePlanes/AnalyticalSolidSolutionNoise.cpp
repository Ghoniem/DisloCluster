/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_AnalyticalSolidSolutionNoise_cpp
#define model_AnalyticalSolidSolutionNoise_cpp

#include <AnalyticalSolidSolutionNoise.h>
//#include <complex.h>

namespace model
{

//    // Cai's doubly-convoluted spreading function in Fourier space
//    typename AnalyticalSolidSolutionNoise::REAL_SCALAR AnalyticalSolidSolutionNoise::Wk_Cai(REAL_SCALAR kx, REAL_SCALAR ky, REAL_SCALAR kz, REAL_SCALAR a)
//    {
//        REAL_SCALAR k = sqrt(kx*kx + ky*ky + kz*kz);
//        if(k>0.0)
//        {
//            return a*k*sqrt(0.5*boost::math::cyl_bessel_k(2,a*k));
//        }
//        else
//        {
//            return 1.;
//        }
//    }
//
//    // Square of Cai's doubly-convoluted spreading function in Fourier space
//    typename AnalyticalSolidSolutionNoise::REAL_SCALAR AnalyticalSolidSolutionNoise::Wk_Cai_squared(REAL_SCALAR kx, REAL_SCALAR ky, REAL_SCALAR kz, REAL_SCALAR a)
//    {
//        const REAL_SCALAR k2(kx*kx + ky*ky + kz*kz);
//        const REAL_SCALAR k(sqrt(k2));
//        if(k2>0.0)
//        {
//            return a*a*k2*0.5*boost::math::cyl_bessel_k(2,a*k);
//        }
//        else
//        {
//            return 1.;
//        }
//    }
//
//    // Cai spreading function
//    typename AnalyticalSolidSolutionNoise::REAL_SCALAR AnalyticalSolidSolutionNoise::W_Cai(REAL_SCALAR r2, REAL_SCALAR a)
//    {
//        return 15.*a*a*a*a/(8.*M_PI*pow(r2+a*a,7./2.));
//    }
//
//    typename AnalyticalSolidSolutionNoise::REAL_SCALAR AnalyticalSolidSolutionNoise::W_t_Cai(REAL_SCALAR r2, REAL_SCALAR a)
//    {
//        return 0.3425*W_Cai(r2,0.9038*a) + 0.6575*W_Cai(r2,0.5451*a);
//    }

    // normalized auto-correlation function in Fourier space for sigma_xy
    typename AnalyticalSolidSolutionNoise::REAL_SCALAR AnalyticalSolidSolutionNoise::S_xy_k(REAL_SCALAR kx, REAL_SCALAR ky, REAL_SCALAR kz) const
    {
        REAL_SCALAR k2 = kx*kx + ky*ky + kz*kz;
        return stressPrefactor*(kx*kx*ky*ky)/(k2*k2)*exp(-a*a*k2);
    }

    // normalized auto-correlation function in Fourier space for sigma_xz
    typename AnalyticalSolidSolutionNoise::REAL_SCALAR AnalyticalSolidSolutionNoise::S_xz_k(REAL_SCALAR kx, REAL_SCALAR ky, REAL_SCALAR kz) const
    {
        REAL_SCALAR k2 = kx*kx + ky*ky + kz*kz;
        return stressPrefactor*(kx*kx*kz*kz)/(k2*k2)*exp(-a*a*k2);
    }

    // normalized auto-correlation function in Fourier space for sigma_yz
    typename AnalyticalSolidSolutionNoise::REAL_SCALAR AnalyticalSolidSolutionNoise::S_yz_k(REAL_SCALAR kx, REAL_SCALAR ky, REAL_SCALAR kz) const
    {
        REAL_SCALAR k2 = kx*kx + ky*ky + kz*kz;
        return stressPrefactor*(ky*ky*kz*kz)/(k2*k2)*exp(-a*a*k2);
    }

    AnalyticalSolidSolutionNoise::AnalyticalSolidSolutionNoise(const std::string& tag,
                                                               const int& seed,
                                                               const GridSizeType& gridSize,
                                                               const GridSpacingType& gridSpacing,
                                                               const double& a_in,
                                                               const double& a_Cai_in,
                                                               const double& MSSS) :
    /* init */ GlidePlaneNoiseBase<2>("AnalyticalSolidSolutionNoise"+tag,seed,gridSize,gridSpacing)
    /* init */,a(a_in)
    /* init */,a_cai(a_Cai_in)
    /* init */,stressPrefactor(MSSS*120.0*M_PI*sqrt(M_PI)*a*a*a/(gridLength.prod()))
    {
    }

    std::array<AnalyticalSolidSolutionNoise::COMPLEX,2> AnalyticalSolidSolutionNoise::kCorrelations(const Eigen::Matrix<double,3,1>& kv,const Eigen::Matrix<int,3,1>&) const
    {
        std::array<AnalyticalSolidSolutionNoise::COMPLEX,2> temp{this->S_xz_k(kv(0),kv(1),kv(2)),this->S_yz_k(kv(0),kv(1),kv(2))};
        if(a_cai>0.0)
        {
            const double wkc2(this->Wk_Cai_squared(kv(0),kv(1),kv(2), a_cai)); // using the square because this is before the square root
            temp[0]*=wkc2;
            temp[1]*=wkc2;
        }
        return temp;
    }

}
#endif

