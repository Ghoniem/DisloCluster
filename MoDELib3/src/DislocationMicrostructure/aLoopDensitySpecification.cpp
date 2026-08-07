/* This file is part of MODEL, the Mechanics Of Defect Evolution Library.
 *
 * Copyright (C) 2015 by Giacomo Po <gpo@ucla.edu>.
 *
 * model is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_aLoopDensitySpecification_cpp_
#define model_aLoopDensitySpecification_cpp_

#include <aLoopDensitySpecification.h>

namespace model
{

    aLoopDensitySpecification::aLoopDensitySpecification():
    /* init */ MicrostructureSpecificationBase("aLoop","Density")
    {
        
    }

    aLoopDensitySpecification::aLoopDensitySpecification(const std::string& fileName):
    /* init */ MicrostructureSpecificationBase("aLoop","Density",fileName)
    /* init */,slipSystemIDs(this->parser->readArray<int>("slipSystemIDs",true))
    /* init */,targetDensity(this->parser->readArray<double>("targetDensity",true))
    /* init */,loopRadiusMean(this->parser->readArray<double>("loopRadiusMean",true))
    /* init */,loopRadiusStd(this->parser->readArray<double>("loopRadiusStd",true))
    /* init */,numberOfSides(this->parser->readArray<int>("numberOfSides",true))
    /* init */,areVacancyLoops(this->parser->readArray<bool>("areVacancyLoops",true))
    /* init */,ellipticityFactor(this->parser->readArray<double>("ellipticityFactor",true))

    {
        
    }

}
#endif
