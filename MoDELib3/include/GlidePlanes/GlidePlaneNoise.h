/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_GlidePlaneNoise_H
#define model_GlidePlaneNoise_H

#include <cmath>
#include <random>
#include <Eigen/Dense>

#include <NoiseTraits.h>
#include <GlidePlaneNoiseBase.h>
#include <PolycrystallineMaterialBase.h>
#include <AnalyticalSolidSolutionNoise.h>
#include <MDSolidSolutionNoise.h>

namespace model
{

    struct GlidePlaneNoise : public std::map<std::string,std::shared_ptr<GlidePlaneNoiseBase<2>>> // solid-solution
    /*                    */,public std::map<std::string,std::shared_ptr<GlidePlaneNoiseBase<1>>> // stacking-fault
    {
        typedef typename Eigen::Matrix<double,3,1> VectorDim;
        typedef typename NoiseTraitsBase::GridSizeType GridSizeType;
        typedef std::map<std::string,std::shared_ptr<GlidePlaneNoiseBase<2>>> SolidSolutionNoiseContainer;
        typedef std::map<std::string,std::shared_ptr<GlidePlaneNoiseBase<1>>> StackingFaultNoiseContainer;
        
        GlidePlaneNoise(const PolycrystallineMaterialBase& mat);
        std::tuple<double,double,double> gridInterp(const Eigen::Matrix<double,2,1>& localPos) const;
        std::tuple<double,double,double> gridVal(const Eigen::Array<int,2,1>& idx) const;
        const SolidSolutionNoiseContainer& solidSolutionNoise() const;
        SolidSolutionNoiseContainer& solidSolutionNoise();
        const StackingFaultNoiseContainer& stackingFaultNoise() const;
        StackingFaultNoiseContainer& stackingFaultNoise();
    };

}
#endif

