/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_PolyhedronInclusionIndividualSpecification_H_
#define model_PolyhedronInclusionIndividualSpecification_H_


#include <string>
#include <vector>
#include <Eigen/Dense>

#include <MicrostructureSpecificationBase.h>

namespace model
{
    struct PolyhedronInclusionIndividualSpecification : public MicrostructureSpecificationBase
    {
        
        static constexpr int dim=3;
        
        std::string mshFile;
        Eigen::Matrix<double,Eigen::Dynamic,3> X0;
        Eigen::Matrix<double,3,3> F;
        Eigen::Matrix<double,1,3*3> eigenDistortion;
        double velocityReductionFactor;
        int phaseID;


        PolyhedronInclusionIndividualSpecification();
        PolyhedronInclusionIndividualSpecification(const std::string& fileName);
    };
}
#endif
