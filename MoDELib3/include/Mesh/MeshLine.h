/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_MeshLine_H_
#define model_MeshLine_H_


#include <cfloat>
#include <tuple>
#include <vector>
#include <Eigen/Dense>
#include <SimplicialMesh.h>
#include <PlanarMeshFace.h>
#include <FiniteLineSegment.h>
#include <StaticID.h>
#include <PlaneSegmentIntersection.h>
#include <PlanePlaneIntersection.h>
#include <MeshBoundarySegment.h>
#include <LineLineIntersection.h>
#include <Line.h>

//#include <EmbeddedPolygonTriangulation.h>

namespace model
{
    
    
    template <int dim>
    struct MeshLine : public FiniteLineSegment<dim>
    {
        
        typedef Eigen::Matrix<double,dim,1> VectorDim;
        typedef Eigen::Matrix<double,dim,dim> MatrixDim;
        
        static FiniteLineSegment<dim> getFiniteLineSegment(const SimplicialMesh<dim>& mesh,
                                                           const VectorDim& p,
                                                           const VectorDim& d);
        
        MeshLine(const SimplicialMesh<dim>& mesh,
                  const VectorDim& p,
                  const VectorDim& d);
        
        MeshLine(const SimplicialMesh<dim>& mesh,
                  const Line<dim>& line);
    };

}
#endif
