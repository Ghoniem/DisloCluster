/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_GlidePlaneNoiseBase_cpp
#define model_GlidePlaneNoiseBase_cpp

#include <chrono>
#include <GlidePlaneNoiseBase.h>
#include <TerminalColors.h>

namespace model
{

    // Cai's doubly-convoluted spreading function in Fourier space
    template <int N>
    typename GlidePlaneNoiseBase<N>::REAL_SCALAR GlidePlaneNoiseBase<N>::Wk_Cai(REAL_SCALAR kx, REAL_SCALAR ky, REAL_SCALAR kz, REAL_SCALAR a)
    {
        REAL_SCALAR k = sqrt(kx*kx + ky*ky + kz*kz);
        if(k>0.0)
        {
            return a*k*sqrt(0.5*boost::math::cyl_bessel_k(2,a*k));
        }
        else
        {
            return 1.;
        }
    }

    // Square of Cai's doubly-convoluted spreading function in Fourier space
    template <int N>
    typename GlidePlaneNoiseBase<N>::REAL_SCALAR GlidePlaneNoiseBase<N>::Wk_Cai_squared(REAL_SCALAR kx, REAL_SCALAR ky, REAL_SCALAR kz, REAL_SCALAR a)
    {
        const REAL_SCALAR k2(kx*kx + ky*ky + kz*kz);
        const REAL_SCALAR k(sqrt(k2));
        if(k2>0.0)
        {
            return a*a*k2*0.5*boost::math::cyl_bessel_k(2,a*k);
        }
        else
        {
            return 1.;
        }
    }

    // Cai spreading function
    template <int N>
    typename GlidePlaneNoiseBase<N>::REAL_SCALAR GlidePlaneNoiseBase<N>::W_Cai(REAL_SCALAR r2, REAL_SCALAR a)
    {
        return 15.*a*a*a*a/(8.*M_PI*pow(r2+a*a,7./2.));
    }

    template <int N>
    typename GlidePlaneNoiseBase<N>::REAL_SCALAR GlidePlaneNoiseBase<N>::W_t_Cai(REAL_SCALAR r2, REAL_SCALAR a)
    {
        return 0.3425*W_Cai(r2,0.9038*a) + 0.6575*W_Cai(r2,0.5451*a);
    }

    template <int N>
    GlidePlaneNoiseBase<N>::GlidePlaneNoiseBase(const std::string& tag_in,
                                                const int& seed_in,
                                                const NoiseTraitsBase::GridSizeType& gridSize_in,
                                                const NoiseTraitsBase::GridSpacingType& gridSpacing_in):
    /*init*/ UniformPeriodicGrid<2>(gridSize_in.template segment<2>(0),gridSpacing_in.template segment<2>(0))
    /*init*/,tag(tag_in)
    /*init*/,seed(seed_in)
    /*init*/,gridSize(gridSize_in)
    /*init*/,gridSpacing(gridSpacing_in)
    /*init*/,gridLength(gridSize.template cast<double>()*gridSpacing)
    /*init*/,NX(gridSize(0))
    /*init*/,NY(gridSize(1))
    /*init*/,NZ(gridSize(2))
    /*init*/,NK(NZ>1 ? NX*NY*(NZ/2+1) : NX*(NY/2+1))
    /*init*/,NR(NX*NY*NZ)
    /*init*/,LX(gridLength(0))
    /*init*/,LY(gridLength(1))
    /*init*/,LZ(gridLength(2))
    {
    }

    template <int N>
    const typename GlidePlaneNoiseBase<N>::NoiseContainerType& GlidePlaneNoiseBase<N>::noiseVector() const
    {
        return *this;
    }

    template <int N>
    typename GlidePlaneNoiseBase<N>::NoiseContainerType& GlidePlaneNoiseBase<N>::noiseVector()
    {
        return *this;
    }

    template <int N>
    int GlidePlaneNoiseBase<N>::storageIndex(const int& i,const int& j) const
    {/*!\param[in] localPos the  position vector on the grid
      * \returns The grid index periodically wrapped within the gridSize bounds
      */
        return gridSize(1)*i+j;
    }

    template <int N>
    void GlidePlaneNoiseBase<N>::computeRealNoise()
    {
        std::cout<<greenBoldColor<<"Computing "+tag<<defaultColor<<std::flush;
        const auto t0= std::chrono::system_clock::now();

#ifdef _MODEL_GLIDE_PLANE_NOISE_GENERATOR_
        
    // Allocate k-space and r-space vectors
    std::vector<COMPLEX*> kNoisyCorrelations;
    std::vector<REAL_SCALAR*> rNoisyCorrelations;
    for(int n=0;n<N;++n)
    {
        kNoisyCorrelations.push_back((COMPLEX*) fftw_malloc(sizeof(COMPLEX)*NK));
        rNoisyCorrelations.push_back((REAL_SCALAR*) fftw_malloc(sizeof(REAL_SCALAR)*NR));
    }
    
    // Retreive k-correlations and randomize them
        // Compute loop bounds conditionally
    const int J_MAX = (NZ > 1) ? NY : (NY/2 + 1);
    const int K_MAX = (NZ > 1) ? (NZ/2 + 1) : 1 ;

    std::default_random_engine generator(seed);
    std::normal_distribution<REAL_SCALAR> distribution(0.0,1.0);
    for(int i=0; i<NX; i++)
    {
        for(int j=0; j<J_MAX; j++)
        {
            for(int k=0; k<K_MAX; k++)
            {
//                const int ind = NY*(NZ/2+1)*i + j*(NZ/2+1) + k;
                const int ind = J_MAX*K_MAX*i + j*K_MAX + k;

                REAL_SCALAR kx = 2.*M_PI/LX*REAL_SCALAR(i);
                if(i>NX/2)
                {
                    kx = 2.*M_PI/LX*REAL_SCALAR(i-NX);
                }
                
                REAL_SCALAR ky = 2*M_PI/LY*REAL_SCALAR(j);
                if(j>NY/2)
                {
                    ky = 2.*M_PI/LY*REAL_SCALAR(j-NY);
                }
                
                REAL_SCALAR kz = 2.*M_PI/LZ*REAL_SCALAR(k);
                
                const Eigen::Matrix<double,3,1> kv((Eigen::Matrix<double,3,1>()<<kx,ky,kz).finished());
                const Eigen::Matrix<int,3,1> kvID((Eigen::Matrix<int,3,1>()<<i,j,k).finished());
                
                // random numbers
                REAL_SCALAR Nk_yz = distribution(generator);
                REAL_SCALAR Mk_yz = distribution(generator);
                REAL_SCALAR Nk_xz, Mk_xz;
                if(kx*ky>=0)
                {
                    Nk_xz = Nk_yz;
                    Mk_xz = Mk_yz;
                }
                else
                {
                    Nk_xz = -Nk_yz;
                    Mk_xz = -Mk_yz;
                }
                
                const double kCorrFactor(NZ>1 ? ((k==0 || k==NZ/2)? 1.0 : 2.0) : ((j==0 || j==NY/2)? 1.0 : 2.0)); // /!\ special case for k=0 and k==NZ/2 because of folding of C2R Fourier transform
                const auto kCorr(kCorrelations(kv,kvID));
                for(int n=0;n<N;++n)
                {
                    kNoisyCorrelations[n][ind]=sqrt(kCorr[n]/kCorrFactor)*(Nk_yz+Mk_yz*COMPLEX(0.0,1.0));
                }
            }
        }
    }
            
    // Compute Real noise
    for(int n=0;n<N;++n)
    {
        kNoisyCorrelations[n][0]=0;
        fftw_plan nPlan = (NZ>1 ? fftw_plan_dft_c2r_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(kNoisyCorrelations[n]), rNoisyCorrelations[n], FFTW_ESTIMATE)
                                : fftw_plan_dft_c2r_2d(NX, NY, reinterpret_cast<fftw_complex*>(kNoisyCorrelations[n]), rNoisyCorrelations[n], FFTW_ESTIMATE));
        fftw_execute(nPlan);
    }
    
    // Store Real noise
    this->reserve(NX*NY);
    const int k=0;
    for(int i=0;i<NX;i++)
    {
        for(int j=0;j<NY;j++)
        {
            const int ind = NY*NZ*i + j*NZ + k;
            Eigen::Matrix<REAL_SCALAR,1,N> temp;
            for(int n=0;n<N;++n)
            {
                temp(n)=rNoisyCorrelations[n][ind];
            }
            this->push_back(NoiseTraits<N>::fromMatrix(temp));
        }
    }

#else
    this->resize(NX*NY,NoiseTraits<N>::Zero());
#endif
        std::cout<<greenColor<<" ["<<(std::chrono::duration<double>(std::chrono::system_clock::now()-t0)).count()<<" sec]"<<defaultColor<<std::endl;
    }

    template <int N>
    void GlidePlaneNoiseBase<N>::computeRealNoiseStatistics(const PolycrystallineMaterialBase& mat) const
    {
        // Compute Statistics
        NoiseType ave(NoiseTraits<N>::Zero());
        for(const auto& valArr: noiseVector())
        {
            ave+=valArr;
        }
        ave/=noiseVector().size();
        
        NoiseType var(NoiseTraits<N>::Zero());
        for(const auto& valArr: noiseVector())
        {
            var+=NoiseTraits<N>::squared(valArr-ave);
        }
        var/=noiseVector().size();
        
        std::cout<<"gridSize= "<<gridSize.transpose()<<std::endl;
        std::cout<<"gridSpacing= "<<gridSpacing.transpose()<<std::endl;
        std::cout<<"noiseAverage="<<ave*mat.mu_SI<<" [Pa]"<<std::endl;
        std::cout<<"noiseVariance="<<var*std::pow(mat.mu_SI,2)<<" [Pa^2]"<<std::endl;
    }
        
        //    void AnalyticalSolidSolutionNoise::Write_field_slice(REAL_SCALAR *F, const char *fname)
        //    {
        //        FILE *OutFile=fopen(fname,"w");
        //
        //        fprintf(OutFile,"# vtk DataFile Version 2.0\n");
        //        fprintf(OutFile,"iter %d\n",0);
        //        fprintf(OutFile,"BINARY\n");
        //        fprintf(OutFile,"DATASET STRUCTURED_POINTS\n");
        //        fprintf(OutFile,"ORIGIN \t %f %f %f\n",0.,0.,0.);
        //        fprintf(OutFile,"SPACING \t %f %f %f\n", DX, DY, DZ);
        //        fprintf(OutFile,"DIMENSIONS \t %d %d %d\n", NX, NY, 1);
        //        fprintf(OutFile,"POINT_DATA \t %d\n",NX*NY);
        //        fprintf(OutFile,"SCALARS \t volume_scalars double 1\n");
        //        fprintf(OutFile,"LOOKUP_TABLE \t default\n");
        //
        //        for(int i=0;i<NX;i++)
        //        {
        //            for(int j=0;j<NY;j++)
        //            {
        //                const int k=0;
        //                const int ind = NY*NZ*i + j*NZ + k;
        //                const double temp=NoiseTraitsBase::ReverseDouble(double(F[ind]));
        //                fwrite(&temp, sizeof(double), 1, OutFile);
        //            }
        //        }
        //
        //        fclose(OutFile);
        //    }

    template struct GlidePlaneNoiseBase<1>;
    template struct GlidePlaneNoiseBase<2>;

}
#endif

