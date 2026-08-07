/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_PolyhedronInclusionIndividualSpecification_cpp_
#define model_PolyhedronInclusionIndividualSpecification_cpp_

#include <PolyhedronInclusionIndividualSpecification.h>

namespace model
{
    PolyhedronInclusionIndividualSpecification::PolyhedronInclusionIndividualSpecification():
    /* init */ MicrostructureSpecificationBase("PolyhedronInclusion","Individual")
    {
        
    }

    PolyhedronInclusionIndividualSpecification::PolyhedronInclusionIndividualSpecification(const std::string& fileName):
    /* init */ MicrostructureSpecificationBase("PolyhedronInclusion","Individual",fileName)
    /* init */,mshFile(this->parser->readString("mshFile",true))
    /* init */,X0(this->parser->readMatrixCols<double>("x0",3,true))
    /* init */,F(this->parser->readMatrix<double>("F",3,3,true))
    /* init */,eigenDistortion(this->parser->readMatrix<double>("eigenDistortion",1,3*3,true))
    /* init */,velocityReductionFactor(this->parser->readScalar<double>("velocityReductionFactor",true))
    /* init */,phaseID(this->parser->readScalar<int>("phaseID",true))
    {

    }
}
#endif
