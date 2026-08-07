/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_Line_cpp_
#define model_Line_cpp_

#include <cfloat>
#include <tuple>
#include <limits>
//#include <map>
#include <Eigen/Dense>
#include <iostream>

#include <Line.h>

namespace model
{

    template <int dim>
    Line<dim>::Line(const VectorDim& P_in,const VectorDim& d_in):
    /* init */ P(P_in)
    /* init */,d(d_in.normalized())
    {
        const double dNorm(d.norm());
        if((dNorm-1.0)>std::numeric_limits<double>::epsilon())
        {
            std::cout<<"dNorm="<<dNorm<<std::endl;
            throw std::runtime_error("Line has non-unit direction");
        }
    }

    template <int dim>
    typename Line<dim>::VectorDim Line<dim>::snap(const VectorDim& P0) const
    {
        return P+(P0-P).dot(d)*d;
    }


    template <int dim>
    bool Line<dim>::contains(const VectorDim& P0,const double& tol) const
    {
        return distanceToPoint(P0)<tol;
    }


    template <int dim>
    double Line<dim>::distanceToPoint(const VectorDim& P0) const
    {
        return (P0-snap(P0)).norm();
    }

    template struct Line<1>;
    template struct Line<2>;
    template struct Line<3>;

}
#endif
