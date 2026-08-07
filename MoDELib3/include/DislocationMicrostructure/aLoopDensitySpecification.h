/* This file is part of MODEL, the Mechanics Of Defect Evolution Library.
 *
 * Copyright (C) 2015 by Giacomo Po <gpo@ucla.edu>.
 *
 * model is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_aLoopDensitySpecification_H_
#define model_aLoopDensitySpecification_H_

#include <string>
#include <vector>
#include <Eigen/Dense>

#include <MicrostructureSpecificationBase.h>

namespace model
{
    struct aLoopDensitySpecification : public MicrostructureSpecificationBase
    {
        std::vector<int> slipSystemIDs;
        std::vector<double> targetDensity;
        std::vector<double> loopRadiusMean;
        std::vector<double> loopRadiusStd;
        std::vector<int> numberOfSides;
        std::vector<bool> areVacancyLoops;
        std::vector<double> ellipticityFactor;

        aLoopDensitySpecification();
        aLoopDensitySpecification(const std::string& fileName);
    };
}
#endif
