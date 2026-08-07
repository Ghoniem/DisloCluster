/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_MDSolidSolutionNoise_cpp
#define model_MDSolidSolutionNoise_cpp

#include <MDSolidSolutionNoise.h>

namespace model
{

    MDSolidSolutionNoise::MDSolidSolutionNoise(const PolycrystallineMaterialBase& mat,
                                               const std::string& tag,
                                               const std::string& correlationFile_L,
                                               const std::string& correlationFile_T,
                                               const int& seed,
                                               const GridSizeType& gridSize,
                                               const GridSpacingType& gridSpacing,
                                               const double& a_Cai_in
                                               ) :
    /*init*/ GlidePlaneNoiseBase<2>("MDSolidSolutionNoise"+tag,seed,gridSize,gridSpacing)
    /*init*/,a_cai(a_Cai_in)
    {
        // allocate
        REAL_SCALAR *Rr_xz = (REAL_SCALAR*) fftw_malloc(sizeof(REAL_SCALAR)*this->NR); //correlation in real space
        REAL_SCALAR *Rr_yz = (REAL_SCALAR*) fftw_malloc(sizeof(REAL_SCALAR)*this->NR); //correlation in real space
                
        Rk_xz = (COMPLEX*) fftw_malloc(sizeof(COMPLEX)*this->NK); //correlation in fourier space
        Rk_yz = (COMPLEX*) fftw_malloc(sizeof(COMPLEX)*this->NK); //correlation in fourier space
        
        // Initialize FFTW plans as member variables
        fftw_plan plan_R_xz_r2c = fftw_plan_dft_r2c_2d(this->NY, this->NX, Rr_xz, reinterpret_cast<fftw_complex*>(Rk_xz), FFTW_ESTIMATE);
        fftw_plan plan_R_yz_r2c = fftw_plan_dft_r2c_2d(this->NY, this->NX, Rr_yz, reinterpret_cast<fftw_complex*>(Rk_yz), FFTW_ESTIMATE);
        
        const auto originalDimensionsL(readVTKfileDimension(correlationFile_L.c_str()));
        const auto originalDimensionsT(readVTKfileDimension(correlationFile_T.c_str()));
        if((originalDimensionsL-originalDimensionsT).matrix().squaredNorm()>0)
        {
            throw std::runtime_error("correlationFile_L and correlationFile_T have different grid sizes.");
        }
        const int originalNX = originalDimensionsL(0);
        const int originalNY = originalDimensionsL(1);
        if(originalDimensionsL(2)!=1)
        {
            throw std::runtime_error("vtk correlationFiles 'DIMENSIONS' should have 3rd component == 1.");
        }
        
        REAL_SCALAR *Rr_xz_original = (REAL_SCALAR*) fftw_malloc(sizeof(REAL_SCALAR)*originalNX*originalNY); //correlation in real space
        REAL_SCALAR *Rr_yz_original = (REAL_SCALAR*) fftw_malloc(sizeof(REAL_SCALAR)*originalNX*originalNY); //correlation in real space
                
        // populate Rr_xy_original with the correlation data
        SolidSolutionCorrelationReader(correlationFile_L, Rr_xz_original);
        SolidSolutionCorrelationReader(correlationFile_T, Rr_yz_original);
                
        // Divide the values in Rr_xz_original and Rr_yz_original by mat.mu^2
        for (int i = 0; i < originalNX * originalNY; ++i)
        {
            Rr_xz_original[i] /= (mat.mu_SI*mat.mu_SI); // divide by mu^2
            Rr_yz_original[i] /= (mat.mu_SI*mat.mu_SI); // divide by mu^2
        }
        
        int start_y = (this->NY - originalNY) / 2;
        int start_x = (this->NX - originalNX) / 2;
        
        // // initialize with zeros
        for (int i = 0; i < this->NY; ++i)
        {
            for (int j = 0; j < this->NX; ++j)
            {
                Rr_xz[i*this->NX + j] = 0;
                Rr_yz[i*this->NX + j] = 0;
            }
        }
        
        // 0-pading from centere
        for (int i = 0; i < originalNY; ++i)
        {
            for (int j = 0; j < originalNX; ++j)
            {
                Rr_xz[(start_y + i) * this->NX + (start_x + j)] = Rr_xz_original[i * originalNX + j];
                Rr_yz[(start_y + i) * this->NX + (start_x + j)] = Rr_yz_original[i * originalNX + j];
            }
        }
                
        plan_R_xz_r2c = fftw_plan_dft_r2c_2d(this->NY, this->NX, Rr_xz_original, reinterpret_cast<fftw_complex*>(Rk_xz), FFTW_ESTIMATE);
        plan_R_yz_r2c = fftw_plan_dft_r2c_2d(this->NY, this->NX, Rr_yz_original, reinterpret_cast<fftw_complex*>(Rk_yz), FFTW_ESTIMATE);
                
        // Execute FFTW plans to populate Rk_xz and Rk_yz
        fftw_execute(plan_R_xz_r2c);
        fftw_execute(plan_R_yz_r2c);
        
        // Normalize the FFT output
        for (int i = 0; i < this->NX; ++i)
        {
            for (int j = 0; j < (this->NY/2 + 1); ++j)
            {
                Rk_xz[i * (this->NY/2 + 1) + j] /= (this->NX * this->NY);
                Rk_yz[i * (this->NY/2 + 1) + j] /= (this->NX * this->NY);
            }
        }
        
        // Destroy FFTW plans
        fftw_destroy_plan(plan_R_xz_r2c);
        fftw_destroy_plan(plan_R_yz_r2c);
        
        // Free allocated memory
        fftw_free(Rr_xz_original);
        fftw_free(Rr_yz_original);
        
        // Free allocated memory
        fftw_free(Rr_xz);
        fftw_free(Rr_yz);
    }

    std::array<MDSolidSolutionNoise::COMPLEX, 2> MDSolidSolutionNoise::kCorrelations(const Eigen::Matrix<double, 3, 1> &kv, const Eigen::Matrix<int, 3, 1> &index) const
    {
        std::array<MDSolidSolutionNoise::COMPLEX, 2> temp;
        int idx=(this->NY/2+1)*NZ*index(0) + index(1)*NZ + index(2);
        temp[0] = Rk_xz[idx];
        temp[1] = Rk_yz[idx];
        if(a_cai>0.0)
        {
            const double wkc2(this->Wk_Cai_squared(kv(0),kv(1),kv(2), a_cai)); // using the square because this is before the square root
            temp[0]*=wkc2;
            temp[1]*=wkc2;
        }
        return temp;
    }

    void MDSolidSolutionNoise::SolidSolutionCorrelationReader(const std::string& correlationFile, REAL_SCALAR *Rr_xz)
    {
        std::cout << "Reading solid solution correlation (xz)" << std::endl;
        
        std::ifstream vtkFile(correlationFile); //access vtk file
        // error check
        if (!vtkFile.is_open())
        {
            throw std::runtime_error("Error opening solid solution VTK Sxz correlation file!");
        }
        
        std::string line;
        while (std::getline(vtkFile, line))
        {
            //if the "POINT_DATA" string is read, read the following data
            if(line.find("POINT_DATA")!=std::string::npos)
            {
                const size_t numOfHeaders = 2;
                // get the number of points in the file
                const size_t firstSpace(line.find(' '));
                const size_t numOfPoints = std::atoi(line.substr(firstSpace+1).c_str());
                std::cout << "Number of points: " << numOfPoints << std::endl;
                
                // read the point coordinates
                for(size_t n=0; n<numOfPoints+numOfHeaders; ++n)
                {
                    std::getline(vtkFile, line);
                    // ignore the headers that come right after point_data
                    if(n<numOfHeaders)
                        continue;
                    const int ind = n-numOfHeaders;
                    //correlationCoeffs.push_back(std::atoi(line.c_str()));
                    Rr_xz[ind] = std::atof(line.c_str());
                }
            }
        }
        vtkFile.close();
    }

    typename MDSolidSolutionNoise::GridSizeType MDSolidSolutionNoise::readVTKfileDimension(const char *fname)
    {
        FILE *InFile=fopen(fname,"r");
        
        if (InFile == NULL)
        {
            fprintf(stderr, "Can't open solid solution correlation VTK file %s\n",fname);
            exit(1);
        }
        // return the 5th line of the vtk file
        char line[200];
        for(int i=0;i<4;i++)
        {
            fgets(line, 200, InFile);
        }
        // scan the returned line
        int NXX, NYY, NZZ;
        fscanf(InFile, "%s %d %d %d\n", line, &(NXX), &(NYY), &(NZZ));
        return (GridSizeType()<<NXX,NYY,NZZ).finished();
    }

}
#endif
