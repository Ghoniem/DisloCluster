/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MODEL is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_UniformPeriodicGrid_h_
#define model_UniformPeriodicGrid_h_

#include <Eigen/Dense>
#include <CTM.h>


namespace model
{

    template <int dim>
    struct UniformPeriodicGrid
    {
        
        typedef Eigen::Array<int   ,dim,1> ArrayDimI;
        typedef Eigen::Array<double,dim,1> ArrayDimD;
        
        const ArrayDimI gridSize;
        const ArrayDimD gridSpacing;

        UniformPeriodicGrid(const ArrayDimI& gridSize_in,const ArrayDimD& gridSpacing_in);
        static std::array<ArrayDimI,CTM::pow(2,dim)> cornerIdx(const std::pair<ArrayDimI,ArrayDimI>& idx);
        std::array<ArrayDimI,CTM::pow(2,dim)> posToCornerIdx(const ArrayDimD& localPos) const;
        std::array<ArrayDimI,CTM::pow(2,dim)> posToPeriodicCornerIdx(const ArrayDimD& localPos) const;
        ArrayDimD idxToPos(const ArrayDimI& idx) const;
        std::array<double,CTM::pow(2,dim)> posToWeights(const ArrayDimD& localPos) const;
        std::pair<std::array<ArrayDimI,CTM::pow(2,dim)>,std::array<double,CTM::pow(2,dim)>> posToPeriodicCornerIdxAndWeights(const ArrayDimD& localPos) const;
        std::pair<ArrayDimI,ArrayDimI> posToIdx(const ArrayDimD& localPos) const;
        std::pair<ArrayDimI,ArrayDimI> posToPeriodicIdx(const ArrayDimD& localPos) const;
        ArrayDimI idxToPeriodicIdx(const ArrayDimI& gi) const;
        std::pair<ArrayDimI,ArrayDimI> idxToPeriodicIdx(const std::pair<ArrayDimI,ArrayDimI>& idx) const;
    };
}
#endif
