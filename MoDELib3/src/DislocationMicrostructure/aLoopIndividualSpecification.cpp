/* This file is part of MODEL, the Mechanics Of Defect Evolution Library.
 *
 * Copyright (C) 2015 by Giacomo Po <gpo@ucla.edu>.
 *
 * model is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_aLoopIndividualSpecification_cpp_
#define model_aLoopIndividualSpecification_cpp_

#include <aLoopIndividualSpecification.h>

namespace model
{
    aLoopIndividualSpecification::aLoopIndividualSpecification():
    /* init */ MicrostructureSpecificationBase("aLoop","Individual")
    {
        
    }

    aLoopIndividualSpecification::aLoopIndividualSpecification(const std::string& fileName):
    /* init */ MicrostructureSpecificationBase("aLoop","Individual",fileName)
    {
        planeIDs=this->parser->readArray<int>("planeIDs",true);
        if(planeIDs.size())
        {
            loopRadii=this->parser->readArray<double>("loopRadii_SI",true);
            loopSides=this->parser->readArray<int>("loopSides",true);
            loopCenters=this->parser->readMatrix<double>("loopCenters",planeIDs.size(),3,true);
            isVacancyLoop=this->parser->readArray<int>("isVacancyLoop",true);
        }
    }
}
#endif
