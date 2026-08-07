/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_FrankLoopsGenerator_cpp_
#define model_FrankLoopsGenerator_cpp_

#include <numbers>
#include <chrono>
#include <random>
#include <cmath>
#include <list>
#include <assert.h>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <limits>
#include <set>

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
#include <MicrostructureGenerator.h>
#include <FrankLoopsGenerator.h>
#include <PlaneLineIntersection.h>

namespace model
{

    FrankLoopsGenerator::FrankLoopsGenerator(const FrankLoopsDensitySpecification& spec,MicrostructureGenerator& mg)
    {
        std::cout<<magentaBoldColor<<"Generating Frank loop density"<<defaultColor<<std::endl;
        if(spec.targetDensity>0.0)
        {
            std::random_device rd;
            std::mt19937 generator(rd());
            std::normal_distribution<double> radiusDistribution(spec.radiusDistributionMean/mg.ddBase.poly.b_SI,spec.radiusDistributionStd/mg.ddBase.poly.b_SI);
            std::set<int> planeIDs(spec.planeIDs.begin(), spec.planeIDs.end());
            std::uniform_int_distribution<size_t> distrib(0, planeIDs.size() - 1);
//            for (int pID : planeIDs)
//            {
//                if (pID >  static_cast<int>(planeIDs.size()) )
//                {
//                    throw std::runtime_error("planeID " + std::to_string(pID) + " not found, skipping.");
//                }
//            }
            
            double density=0.0;
            while(density<spec.targetDensity)
            {
                const std::pair<LatticeVector<3>, int> rp(mg.ddBase.poly.randomLatticePointInMesh());
                const LatticeVector<3> L0=rp.first;
                const size_t grainID=rp.second;
                
                auto iter = planeIDs.begin();
                std::uniform_int_distribution<> pDist(0, mg.ddBase.poly.grain(grainID).singleCrystal->planeNormals().size() - 1);
                std::advance(iter, distrib(generator));
                int pID(*iter==-1? pDist(generator) : *iter );
                const double radius(radiusDistribution(generator));
                
//                std::cout<< "Plane ID: " << pID << "  isVacancyLoop?  " << spec.areVacancyLoops << "  Radius: " << radius << std::endl;

                
                try
                {
//                    const bool isVL(spec.areVacancyLoops>0);
                    bool success = generateSingle(mg, pID, L0.cartesian(),grainID, radius, spec.numberOfSides, spec.burgersFactor, spec.areVacancyLoops);
                    if (success)  // Only update density if the loop was inserted
                    {
                        density += 2.0 * std::numbers::pi * radius / mg.ddBase.mesh.volume() / std::pow(mg.ddBase.poly.b_SI, 2);
                        std::cout << "Frank loop density=" << density << std::endl;
                    }
                    else
                    {
                        density += 0.0;
                    }
                }
                catch(const std::exception& e)
                {
                    std::cerr << "Exception in loop generation: " << e.what() << std::endl;
                }
            }
        }
    }

FrankLoopsGenerator::FrankLoopsGenerator(const FrankLoopsIndividualSpecification& spec,MicrostructureGenerator& mg)
    {
        std::cout<<magentaBoldColor<<"Generating individual Frank loops"<<defaultColor<<std::endl;
        if(spec.planeIDs.size())
        {
            if(spec.planeIDs.size()!=spec.loopRadii.size())
            {
                throw std::runtime_error("spec.planeIDs.size()="+std::to_string(spec.planeIDs.size())+" NOT EQUAL TO spec.spec.spec.loopRadii.size()="+std::to_string(spec.loopRadii.size()));
            }
            if(int(spec.planeIDs.size())!=spec.loopCenters.rows())
            {
                throw std::runtime_error("spec.planeIDs.size()="+std::to_string(spec.planeIDs.size())+" NOT EQUAL TO spec.loopCenters.rows()="+std::to_string(spec.loopCenters.rows()));
            }
            if(spec.planeIDs.size()!=spec.loopSides.size())
            {
                throw std::runtime_error("spec.planeIDs.size()="+std::to_string(spec.planeIDs.size())+" NOT EQUAL TO spec.loopSides.size()="+std::to_string(spec.loopSides.size()));
            }
            if(spec.planeIDs.size()!=spec.burgersFactor.size())
            {
                throw std::runtime_error("spec.planeIDs.size()="+std::to_string(spec.planeIDs.size())+" NOT EQUAL TO spec.burgersFactor.size()="+std::to_string(spec.burgersFactor.size()));
            }
            if(spec.planeIDs.size()!=spec.isVacancyLoop.size())
            {
                throw std::runtime_error("spec.planeIDs.size()="+std::to_string(spec.planeIDs.size())+" NOT EQUAL TO spec.isVacancyLoop.size()="+std::to_string(spec.isVacancyLoop.size()));
            }
            for(size_t k=0;k<spec.planeIDs.size();++k)
            {
                std::pair<bool, const Simplex<3,3>*> found(mg.ddBase.mesh.search(spec.loopCenters.row(k)));
                if (!found.first)
                {
                    std::cout << "Point " << spec.loopCenters.row(k) << " is outside mesh. Skipping loop insertion." << std::endl;
                }
                else
                {
                    const int grainID(found.second->region->regionID);
                    const bool isVL(spec.isVacancyLoop[k]>0);
                    generateSingle(mg,spec.planeIDs[k],spec.loopCenters.row(k),grainID,spec.loopRadii[k]/mg.ddBase.poly.b_SI,spec.loopSides[k],spec.burgersFactor[k],isVL);
                }
               
            }
        }
        
    }

    bool FrankLoopsGenerator::generateSingle(MicrostructureGenerator& mg, const int& pID, const VectorDimD& center,const int& grainID, const double& radius, const size_t& sides, const double& burgersFactor, const bool& isVacancyLoop)
    {
        const auto& grain(mg.ddBase.poly.grain(grainID));
        if (pID >= 0 && pID < int(grain.singleCrystal->planeNormals().size()))
        {
            const auto& planeBase(*grain.singleCrystal->planeNormals()[pID]);
            const long int planeIndex(planeBase.closestPlaneIndexOfPoint(center));
            GlidePlaneKey<3> glidePlaneKey(planeIndex, planeBase);
            std::shared_ptr<PeriodicGlidePlane<3>> glidePlane(mg.ddBase.periodicGlidePlaneFactory.get(glidePlaneKey));
            const VectorDimD P0(glidePlane->referencePlane->snapToPlane(center));

            std::vector<VectorDimD> loopNodePos;
            for (size_t k = 0; k < sides; ++k)
            {
                loopNodePos.push_back(P0 + Eigen::AngleAxisd(k * 2.0 * std::numbers::pi / sides, planeBase.G2L.row(2).transpose()) *
                                      planeBase.G2L.row(0).transpose() * radius);
            }

            if (!mg.ddBase.isPeriodicDomain)
            {
                if (!mg.allPointsInGrain(loopNodePos, grainID))
                {
                    std::cout << "Some loop nodes are outside the grain. Skipping loop insertion." << std::endl;
                    return false;
                }
            }

            ReciprocalLatticeDirection<3> rp(planeBase);
            const VectorDimD b(isVacancyLoop ? (burgersFactor * rp.cartesian().normalized() * rp.planeSpacing()).eval()
                                             : (-1.0 * burgersFactor * rp.cartesian().normalized() * rp.planeSpacing()).eval());
            mg.insertJunctionLoop(loopNodePos, glidePlane, b, glidePlane->referencePlane->unitNormal, P0, grainID, DislocationLoopIO<3>::SESSILELOOP);
            return true;
        }
        else
        {
            if (pID < 0)
            {
                std::cout << "Skipping planeID " << pID << std::endl;
            }
            else
            {
                throw std::runtime_error("planeID " + std::to_string(pID) + " not found, skipping.");
            }
            return false;
        }
    }
}

#endif
