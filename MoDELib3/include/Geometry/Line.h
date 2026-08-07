/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_Line_H_
#define model_Line_H_

#include <cfloat>
#include <tuple>
//#include <map>
#include <Eigen/Dense>
#include <iostream>


namespace model
{
    
    template <int dim>
    struct Line
    {
        
        typedef Eigen::Matrix<double,dim,1> VectorDim;

        const VectorDim P;
        const VectorDim d;

        Line(const VectorDim& P_in,const VectorDim& d_in);
        bool contains(const VectorDim& P0,const double& tol=std::numeric_limits<float>::epsilon()) const;
        VectorDim snap(const VectorDim& P0) const;
        double distanceToPoint(const VectorDim& P0) const;
        

    };
    
}
#endif
