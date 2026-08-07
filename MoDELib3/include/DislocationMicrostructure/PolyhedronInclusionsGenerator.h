/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_PolyhedronInclusionsGenerator_H_
#define model_PolyhedronInclusionsGenerator_H_


#include <chrono>
#include <random>
#include <cmath>
#include <list>
#include <assert.h>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <limits>

//#include <Simplex.h>
#include <SimplicialMesh.h>
#include <Polycrystal.h>
#include <PolycrystallineMaterialBase.h>
#include <LatticeModule.h>
//#include <PlaneMeshIntersection.h>
#include <DislocationNodeIO.h>
#include <DislocationLoopIO.h>
#include <DislocationLoopLinkIO.h>
#include <DislocationLoopNodeIO.h>
#include <DDconfigIO.h>
#include <DDauxIO.h>

#include <DislocationLinkingNumber.h>
#include <TextFileParser.h>
#include <DislocationInjector.h>
#include <MeshBoundarySegment.h>
#include <GlidePlaneModule.h>
#include <MeshModule.h>
#include <Plane.h>
//#include <PolyhedronInclusionDensitySpecification.h>
#include <PolyhedronInclusionIndividualSpecification.h>


namespace model
{

    class PolyhedronInclusionsGenerator
    {
        
        static constexpr int dim=3;

        
        bool generateSingle(MicrostructureGenerator& mg,const std::map<size_t,Eigen::Vector3d>& polyNodes,const std::map<size_t,std::vector<size_t>>& faceMap, const Eigen::Matrix<double,1,dim*dim>& eTrow, const double& vrc,const int&type);
        
    public:
        
        const bool allowOverlap;
        
//        PolyhedronInclusionsGenerator(const PolyhedronInclusionDensitySpecification& spec,MicrostructureGenerator& mg);
        PolyhedronInclusionsGenerator(const PolyhedronInclusionIndividualSpecification& spec,MicrostructureGenerator& mg);
        
    };

}
#endif
