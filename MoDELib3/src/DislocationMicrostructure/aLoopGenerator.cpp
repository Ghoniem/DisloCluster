/* This file is part of MODEL, the Mechanics Of Defect Evolution Library.
 *
 * Copyright (C) 2015 by Giacomo Po <gpo@ucla.edu>.
 *
 * model is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_aLoopGenerator_cpp_
#define model_aLoopGenerator_cpp_

#include <numbers>
#include <chrono>
#include <random>
#include <cmath>
#include <list>
#include <assert.h>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <limits>
#include <cmath>
#include <numbers> // std::numbers

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
//#include <ConfinedDislocationObject.h>
#include <GlidePlaneModule.h>
#include <MeshModule.h>
#include <Plane.h>
#include <MicrostructureGenerator.h>
#include <PlanesIntersection.h>
#include <LatticePlane.h>
#include <aLoopGenerator.h>

namespace model
{

aLoopGenerator::aLoopGenerator(const aLoopDensitySpecification& spec, MicrostructureGenerator& mg)
{
    std::cout<<magentaBoldColor<<"Generating HCP a loop density"<<defaultColor<<std::endl;
    for (size_t k=0; k<spec.slipSystemIDs.size(); ++k)
    {
        int ssID(spec.slipSystemIDs[k]);
        double targetDensity(spec.targetDensity[k]);
        double radiusMean(spec.loopRadiusMean[k]);
        double STD(spec.loopRadiusStd[k]);
        int loopSides(spec.numberOfSides[k]);
        bool isVL(spec.areVacancyLoops[k]);
        double ef(spec.ellipticityFactor[k]);
        
        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<double> radiusDistribution(radiusMean/mg.ddBase.poly.b_SI,STD/mg.ddBase.poly.b_SI);
        
        std::cout<<"Generating Loops on Slip System "<<ssID<<std::endl;
        int numLoops=0;
        double density=0.0;
        while(density<targetDensity)
        {
            const std::pair<LatticeVector<3>, int> rp(mg.ddBase.poly.randomLatticePointInMesh());
            const LatticeVector<3> L0=rp.first;
            const size_t grainID=rp.second;
            const auto& grain(mg.ddBase.poly.grain(grainID));
            const double radius(radiusDistribution(generator));
            
            const VectorDimD snappedCenter(grain.singleCrystal->snapToLattice(L0.cartesian()).cartesian());
            const VectorDimD b(isVL ? grain.singleCrystal->slipSystems()[ssID]->s.cartesian() : -1*grain.singleCrystal->slipSystems()[ssID]->s.cartesian() );
            const VectorDimD loopNormal(isVL? b.normalized() : -1*b.normalized());
            const ReciprocalLatticeDirection<3> reciprocalDir(grain.singleCrystal->reciprocalLatticeDirection(loopNormal));
            const int sessilePlaneHeight(reciprocalDir.planeIndexOfPoint(snappedCenter));
            const LatticePlaneKey sessilePlaneKey(reciprocalDir,sessilePlaneHeight,grain.grainID);
            const auto sessilePlane(mg.ddBase.periodicGlidePlaneFactory.getFromKey(sessilePlaneKey));
            
            const VectorDimD R(grain.singleCrystal->slipSystems()[ssID]->unitNormal*radius);
            
            
            std::vector<VectorDimD> loopNodePos;
            double R_minor = radius;  // Minor axis is the input radius
            double R_major = radius * ef;  // Major axis is scaled

            const VectorDimD majorDir(0.0, 0.0, 1.0);
            const VectorDimD adjustedMajorDir = (majorDir - loopNormal * (majorDir.dot(loopNormal))).normalized();
            const VectorDimD minorDir = loopNormal.cross(adjustedMajorDir).normalized();

            for (int k = 0; k < loopSides; ++k)
            {
                double angle = k * 2.0 * M_PI / loopSides;
                Eigen::Vector3d point = snappedCenter
                                      + R_major * std::cos(angle) * adjustedMajorDir
                                      + R_minor * std::sin(angle) * minorDir; //parametric equation ellipse
                loopNodePos.push_back(point);
            }
//            for (int k = 0; k < loopSides; ++k)
//            {
//                loopNodePos.push_back(snappedCenter+Eigen::AngleAxisd(k*2.0*M_PI/loopSides, loopNormal)*R);
//            }
            
            if (!mg.ddBase.isPeriodicDomain)
            {
                if (!mg.allPointsInGrain(loopNodePos, grainID))
                {
                    std::cout << "Some loop nodes are outside the grain. Skipping loop insertion." << std::endl;
                    continue;
                }
            }
            
            try
            {
                mg.insertJunctionLoop(loopNodePos, sessilePlane, b, loopNormal, snappedCenter, grainID, DislocationLoopIO<3>::SESSILELOOP);
                numLoops++;
                density=numLoops/mg.ddBase.mesh.volume()/std::pow(mg.ddBase.poly.b_SI,3);
                std::cout<<"loops density with current ssID="<<density<<std::endl;
            }
            catch(const std::exception& e)
            {
                
            }
            
        }
    }
}


aLoopGenerator::aLoopGenerator(const aLoopIndividualSpecification& spec, MicrostructureGenerator& mg)
{
    std::cout<<magentaBoldColor<<"Generating HCP <a> loops"<<defaultColor<<std::endl;
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
        if(spec.planeIDs.size()!=spec.isVacancyLoop.size())
        {
            throw std::runtime_error("spec.planeIDs.size()="+std::to_string(spec.planeIDs.size())+" NOT EQUAL TO spec.isVacancyLoop.size()="+std::to_string(spec.isVacancyLoop.size()));
        }
        for(size_t k=0;k<spec.planeIDs.size();++k)
        {
            
            const bool isVL(spec.isVacancyLoop[k]>0);
            generateSingle(mg,spec.planeIDs[k],spec.loopCenters.row(k),spec.loopRadii[k],spec.loopSides[k],isVL);

        }
    }
    
}


void aLoopGenerator::generateSingle(MicrostructureGenerator& mg,const int& pID,const VectorDimD& center,const double& radius,const size_t& sides,const bool& isVacancyLoop)
{
    std::pair<bool,const Simplex<3,3>*> found(mg.ddBase.mesh.search(center));
    if(!found.first)
    {
        std::cout<<"Point "<<center.transpose()<<" is outside mesh. EXITING."<<std::endl;
        exit(EXIT_FAILURE);
    }
    
    const int grainID(found.second->region->regionID);
    assert(mg.ddBase.poly.grains.size()==1 && "Periodic dislocations only supported for single crystals");
    const auto& grain(mg.ddBase.poly.grain(grainID));
    const VectorDimD P0(grain.singleCrystal->snapToLattice(center).cartesian()); // WARNING: this may shift the point compared to the input.
    const auto& slipSystem(*grain.singleCrystal->slipSystems()[pID]);
    const double planeSpacing(slipSystem.n.planeSpacing());
    const auto& planeBase(*grain.singleCrystal->planeNormals()[pID]);

        if(fabs(planeSpacing-sqrt(3.0)/2.0)<FLT_EPSILON)
        { // prismatic plane spacing
            
            const double loopRadius(radius/mg.ddBase.poly.b_SI);
            VectorDimD b(slipSystem.s.cartesian());   // Prism axis
            const VectorDimD loopNorm(slipSystem.s.cartesian().normalized());   // Prism axis
            const VectorDimD R(slipSystem.unitNormal*loopRadius);
            const ReciprocalLatticeDirection<3> r(grain.singleCrystal->reciprocalLatticeDirection(loopNorm));
            const long int planeIndex(r.closestPlaneIndexOfPoint(P0));
            GlidePlaneKey<3> glidePlaneKey(planeIndex, r);
            std::shared_ptr<PeriodicGlidePlane<3>> glidePlane(mg.ddBase.periodicGlidePlaneFactory.get(glidePlaneKey));

            std::vector<VectorDimD> loopNodePos;
            for(size_t k=0;k<sides;++k)
            {
                loopNodePos.push_back(P0+Eigen::AngleAxisd(k*2.0*M_PI/sides, loopNorm)*R);
            }
            
            const VectorDimD Cycle_plane=(loopNodePos[1]-loopNodePos[0]).cross(loopNodePos[2]-loopNodePos[1]);
            if (b.dot(Cycle_plane)>0)
            {
                if (!isVacancyLoop) // 0 for interstitial
                {
                    b*=-1.0;
                }
            }
            else
            {
                if (isVacancyLoop) // 1 for vacancy
                {
                    b*=-1.0;
                }
            }
            std::cout << "Creating Individual <a> Loop" << std::endl;
            mg.insertJunctionLoop(loopNodePos,glidePlane,b,loopNorm,P0,grainID,DislocationLoopIO<3>::SESSILELOOP);
        }
        else if(fabs(planeSpacing-sqrt(8.0/3.0))<FLT_EPSILON)
        {// basal plane spacing
            
            VectorDimD b(0.5*slipSystem.n.planeSpacing()*slipSystem.n.cartesian().normalized()); // 1/2 c-type loop
            const VectorDimD loopNorm(slipSystem.unitNormal);
            const ReciprocalLatticeDirection<3> r(grain.singleCrystal->reciprocalLatticeDirection(loopNorm));
            const long int planeIndex(r.closestPlaneIndexOfPoint(center));
            GlidePlaneKey<3> glidePlaneKey(planeIndex, r);
            std::shared_ptr<PeriodicGlidePlane<3>> glidePlane(mg.ddBase.periodicGlidePlaneFactory.get(glidePlaneKey));
            const double loopRadius(radius/mg.ddBase.poly.b_SI);
            const VectorDimD R(slipSystem.s.cartesian().normalized()*loopRadius);
            
            std::vector<VectorDimD> loopNodePos;
            for(size_t k=0;k<sides;++k)
            {
                loopNodePos.push_back(P0+Eigen::AngleAxisd(k*2.0*M_PI/sides, loopNorm)*R);
            }
            
            VectorDimD Cycle_plane=(loopNodePos[1]-loopNodePos[0]).cross(loopNodePos[2]-loopNodePos[1]);
            if (b.dot(Cycle_plane)<0)
            {
                if (isVacancyLoop)
                {
                    b*=-1.0;
                }
            }
            else
            {
                if (!isVacancyLoop)
                {
                    b*=-1.0;
                }
            }
            mg.insertJunctionLoop(loopNodePos,glidePlane,b,loopNorm,P0,grainID,DislocationLoopIO<3>::SESSILELOOP);
            std::cout << "Created Individual Basal Loop" << std::endl;
        }

}


}
#endif
