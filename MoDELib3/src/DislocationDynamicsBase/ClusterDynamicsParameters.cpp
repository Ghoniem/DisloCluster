/* This file is part of GreatWhite, a library for integrating
 * MOOSE and MoDeLib.
 *
 * (c) 2018 Multiscale Materials Solutions, LLC
 * ALL RIGTHS REVERVED
 *   Prepared by 2018 Multiscale Materials Solutions
 *     under contract N. 126180
 *   with the NAVAL NUCLEAR LABORATORY
 *
 *  See COPYRIGHT for full restriction
 */


#ifndef model_ClusterDynamicsParameters_cpp_
#define model_ClusterDynamicsParameters_cpp_

#include <ClusterDynamicsParameters.h>

namespace model
{

    /**********************************************************************/
    /* Atomic volume used by the cluster-dynamics equations, in units of b^3.
     *
     * Polycrystal::Omega is det(latticeBasis), i.e. the volume of the lattice
     * CELL. For HEX that cell contains two atoms (det = sqrt(2) for c/a=1.633),
     * so it is twice the atomic volume the CD equations assume, and every
     * content <-> radius <-> density conversion inherits the error. Rather than
     * change Polycrystal (which the elastic/DD side also uses), let the material
     * file state the atomic volume explicitly. The key is OPTIONAL: material
     * files that do not define it keep the previous behaviour.
     */
    template<int dim>
    double ClusterDynamicsParameters<dim>::getClusterAtomicVolume(const DislocationDynamicsBase<dim>& ddBase)
    {
        if(ddBase.simulationParameters.useClusterDynamics)
        {
            try
            {
                const double omegaSI(TextFileParser(ddBase.poly.materialFile).template readScalar<double>("atomicVolume_SI",true));
                return omegaSI/std::pow(ddBase.poly.b_SI,3);
            }
            catch(const std::runtime_error&)
            {// key absent -- fall through to the lattice-cell volume
            }
        }
        return ddBase.poly.Omega;
    }

    /**********************************************************************/
    /* Positivity floor on all concentrations, mirroring ZrMicro's C_floor.
     * Optional material key; defaults to ZrMicro's own default of 1e-20. */
    template<int dim>
    double ClusterDynamicsParameters<dim>::getConcentrationFloor(const DislocationDynamicsBase<dim>& ddBase)
    {
        if(ddBase.simulationParameters.useClusterDynamics)
        {
            try
            {
                return TextFileParser(ddBase.poly.materialFile).template readScalar<double>("concentrationFloor",true);
            }
            catch(const std::runtime_error&)
            {// key absent -- ZrMicro's default
            }
        }
        return 1.0e-20;
    }

    template<int dim>
    ClusterDynamicsParameters<dim>::ClusterDynamicsParameters(const DislocationDynamicsBase<dim>& ddBase) :
    /* init */ kB(ddBase.poly.kB),
    /* init */ T(ddBase.poly.T),
    /* init */ omega(getClusterAtomicVolume(ddBase)),
    /* init */ b(ddBase.poly.b),
    /* init */ G0(ddBase.simulationParameters.useClusterDynamics? TextFileParser(ddBase.poly.materialFile).readScalar<double>("doseRate_dpaPerSec",true)*(ddBase.poly.b_SI/ddBase.poly.cs_SI) : 0.0),
    /* MOBILE SPECIES */
    /* init */ msVector(ddBase.simulationParameters.useClusterDynamics? TextFileParser(ddBase.poly.materialFile).readMatrix<int>("mobileSpeciesVector",1,mSize,true).array().template cast<double>().eval() : Eigen::Array<double,1,mSize>::Zero()),
    /* init */ msRelRelaxVol(ddBase.simulationParameters.useClusterDynamics? TextFileParser(ddBase.poly.materialFile).readMatrix<double>("mobileSpeciesRelRelaxVol",1,mSize,true).array().eval() : Eigen::Array<double,1,mSize>::Zero()),
    /* init */ msEf(ddBase.simulationParameters.useClusterDynamics? (TextFileParser(ddBase.poly.materialFile).readMatrix<double>("mobileSpeciesEnergyFormation_eV",1,mSize,true).array()*ddBase.poly.eV2J/ddBase.poly.mu_SI/pow(ddBase.poly.b_SI,3)).eval() : Eigen::Array<double,1,mSize>::Zero()),
    /* init */ msEm(ddBase.simulationParameters.useClusterDynamics? (TextFileParser(ddBase.poly.materialFile).readMatrix<double>("mobileSpeciesEnergyMigration_eV",mSize,dim*(dim+1)/2,true)*ddBase.poly.eV2J/ddBase.poly.mu_SI/pow(ddBase.poly.b_SI,3)).eval() : Eigen::Matrix<double,mSize,dim*(dim+1)/2>::Zero()),
    /* init */ msD0(ddBase.simulationParameters.useClusterDynamics? (TextFileParser(ddBase.poly.materialFile).readMatrix<double>("mobileSpeciesD0_SI",mSize,dim*(dim+1)/2,true)/ddBase.poly.b_SI/ddBase.poly.cs_SI).eval() : Eigen::Matrix<double,mSize,dim*(dim+1)/2>::Zero()),
    /* init */ D(getD(ddBase.poly.grains)),
    /* init */ invD(getInvD()),
    /* init */ detD(getDetD()),
    /* init */ msCascadeFractions((mSize>1 && G0>0.0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<double>("mobileSpeciesCascadeFractions",1,mSize,true) : Eigen::Matrix<double,1,mSize>::Ones()),
    /* init */ msSurvivingEfficiency(G0>0.0 ? TextFileParser(ddBase.poly.materialFile).readScalar<double>("mobileSpeciesSurvivingEfficiency",true) : 1.0),
    /* init */ G(G0*msSurvivingEfficiency*msCascadeFractions),
    /* init */ Eb((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,mSize>("Eb_eV",true).array()*ddBase.poly.eV2J/ddBase.poly.mu_SI/pow(ddBase.poly.b_SI,3) : Eigen::Array<double,1,mSize>::Zero().eval()),
    /* init */ otherSinks(ddBase.simulationParameters.useClusterDynamics? (TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,mSize>("otherSinks_SI",true)*ddBase.poly.b_SI*ddBase.poly.b_SI).eval() : Eigen::Array<double,1,mSize>::Zero()),
    //    /* init */ dislocationSinks(TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,iSize/2>("dislocationSinks_SI",true)*ddBase.poly.b_SI*ddBase.poly.b_SI),
    //    /* init */ initloopSinks(getInitLoopSinks(TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,iSize>("initloopSinks_SI",true),ddBase.poly.b_SI)),
    /* init */ reactionMap((ddBase.simulationParameters.useClusterDynamics && mSize>1) ? getMap(TextFileParser(ddBase.poly.materialFile).readMatrix<double,mSize*(mSize+1)/2,3>("reactionPrefactorMap",true)) : std::map<std::pair<int,int>,double>()),
    /* init */ R1(mSize>1 ? getR1() : Eigen::Matrix<double,mSize,mSize>::Zero()),
    /* init */ R1cd(iSize>0 ? msVector.abs().matrix().asDiagonal()*R1*(1.0/msVector.abs()).matrix().asDiagonal() : Eigen::Matrix<double,mSize,mSize>::Zero().eval()),
    /* init */ R2(mSize>1 ? getR2() : std::vector<Eigen::Matrix<double,mSize,mSize>>(2,Eigen::Matrix<double,mSize,mSize>::Zero())),
    /* init */ loopNucChannels((mSize>1 && iSize>0) ? getLoopNucChannels() : std::map<std::pair<int,int>,double>()),
    /* init */ discreteDislocationBias(ddBase.simulationParameters.useClusterDynamics? TextFileParser(ddBase.poly.materialFile).readMatrix<double,2,mSize>("discreteDislocationBias",true).eval() : Eigen::Array<double,2,mSize>::Zero()),
    /* IMMOBILE SPECIES */
    /* init */ immobileSpeciesVector((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<int>("immobileSpeciesVector",1,iSize/2,true).array().template cast<double>() : Eigen::Array<double,1,iSize/2>::Zero().eval()),
    /* init */ immobileSpeciesRelRelaxVol((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,iSize/2>("immobileSpeciesRelRelaxVol",true).array() : Eigen::Array<double,1,iSize/2>::Zero().eval()),
    /* init */ immobileSpeciesBurgers((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<double,dim,iSize/2>("immobileSpeciesBurgers",true) : Eigen::Matrix<double,dim,iSize/2>::Zero()),
    /* init */ immobileSpeciesBurgersMagnitude((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? getImmobileSpeciesBurgersMagnitude(ddBase.poly.grains) : Eigen::Array<double,1,iSize/2>::Zero().eval()),
    /* init */ a_bp((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readScalar<double>("alpha_bp",true) : 0.0),
    /* init */ delVPyramid((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readScalar<double>("delVPyramid",true)/pow(ddBase.poly.b_SI,3) : 0.0),
    /* init */ w0((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readScalar<double>("w0",true) : 0.0),
    /* init */ n_s((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readScalar<double>("n_s",true) : 0.0 ),
    /* init */ evc((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readScalar<double>("evc",true) : 0.0 ),
    /* init */ Nvmax((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readScalar<double>("Nvmax",true)*pow(ddBase.poly.b_SI,3) : 0.0 ),
    /* init */ nmin((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,iSize/2>("nmin",true) : Eigen::Array<double,1,iSize/2>::Zero()),
    /* init */ nmax((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,iSize/2>("nmax",true) : Eigen::Array<double,1,iSize/2>::Zero()),
    // r_min is given in [m] in the material file. Convert to b, so that it can be
    // compared directly against the radii returned by rloop()/rpyr(), which are in
    // units of b. The shrinking gate of Eq. (91) is its only consumer.
    /* init */ r_min((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? (TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,iSize/2>("r_min",true).array()/ddBase.poly.b_SI).eval() : Eigen::Array<double,1,iSize/2>::Zero().eval()),
    /* IMMOBILE KINETICS -- Deliverable D1/M1 Sec. 2.2 */
    /* init */ loopCascadeFractions((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,iSize/2>("loopCascadeFractions",true).array().eval() : Eigen::Array<double,1,iSize/2>::Zero().eval()),
    /* init */ nNuc((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,iSize/2>("nNuc",true).array().eval() : Eigen::Array<double,1,iSize/2>::Ones().eval()),
    /* init */ cLL((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,iSize/2>("cLL",true).array().eval() : Eigen::Array<double,1,iSize/2>::Zero().eval()),
    /* init */ cLN((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,iSize/2>("cLN",true).array().eval() : Eigen::Array<double,1,iSize/2>::Zero().eval()),
    /* init */ kappaLL((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readScalar<double>("kappaLL",true) : 0.0),
    /* init */ kappaLN((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readScalar<double>("kappaLN",true) : 0.0),
    // tau_vL(T)=tau0*exp(Ea/kB T), Eq. (95); seconds -> MoDELib time units (b/cs).
    /* init */ tauVac((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readScalar<double>("tau0_vLoop_SI",true)*exp(TextFileParser(ddBase.poly.materialFile).readScalar<double>("Ea_vLoop_eV",true)/(8.617333262e-5*T))*(ddBase.poly.cs_SI/ddBase.poly.b_SI) : 1.0),
    // network dislocation density: [1/m^2] -> [1/b^2]
    /* init */ rhoNetwork((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readScalar<double>("rhoNetwork_SI",true)*ddBase.poly.b_SI*ddBase.poly.b_SI : 0.0),
    /* init */ dadAnisotropy((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,mSize>("dadAnisotropy",true).array().eval() : Eigen::Array<double,1,mSize>::Ones().eval()),
    /* init */ dadZ0((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,mSize>("dadZ0",true).array().eval() : Eigen::Array<double,1,mSize>::Ones().eval()),
    /* init */ loopSinkScale((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,iSize/2>("loopSinkScale",true).array().eval() : Eigen::Array<double,1,iSize/2>::Ones().eval()),
    /* init */ concentrationFloor(getConcentrationFloor(ddBase)),
    /* init */ computeReactions((ddBase.simulationParameters.useClusterDynamics && mSize>0 && mSize+iSize>1)? TextFileParser(ddBase.poly.materialFile).readScalar<int>("computeReactions",true) : 0),
//    /* init */ use0DsinkStrength((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readScalar<int>("use0DsinkStrength",true) : 0),
//    /* init */ Zv((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,dim>("Zv",true) : Eigen::Array<double,1,dim>::Zero()),
//    /* init */ Zi((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,dim>("Zi",true) : Eigen::Array<double,1,dim>::Zero()),
//    /* init */ ZVec((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,mSize>("ZVec",true) : Eigen::Array<double,1,mSize>::Zero()),
//    /* init */ rc_il((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readMatrix<double,1,dim>("rc_il",true)/ddBase.poly.b_SI : Eigen::Array<double,1,dim>::Zero().eval()),
    /* init */ discreteDistanceFactor((ddBase.simulationParameters.useClusterDynamics && iSize>0) ? TextFileParser(ddBase.poly.materialFile).readScalar<double>("distanceFactor",true) : 0.0)
    {
        
        double vacancySum(0.0);
        double interstitialSum(0.0);
        for(int k=0;k<mSize;++k)
        {
            if(msVector(k)<0)
            {// vacancy cluster
                vacancySum+=msCascadeFractions(k);
            }
            else if(msVector(k)>0)
            {// interstitial cluster
                interstitialSum+=msCascadeFractions(k);
            }
        }
        
        if(fabs(vacancySum-1.0)>FLT_EPSILON)
        {
            //            throw std::runtime_error("vacancy cluster fraction does not sum to one.");
            std::cout<<redBoldColor<<"Warning: vacancy cluster fraction does not sum to one."<<defaultColor<<std::endl;
        }
        if(fabs(interstitialSum-1.0)>FLT_EPSILON)
        {
            //            throw std::runtime_error("interstitial cluster fraction does not sum to one.");
            std::cout<<redBoldColor<<"Warning: interstitial cluster fraction does not sum to one."<<defaultColor<<std::endl;
        }
        
        for(const auto& pair: reactionMap)
        {
            const auto key=pair.first;
            std::cout<<"find the interaction between "<<key.first<<" - "<<key.second<<" and parameter: "<<pair.second<<std::endl;
        }
        
        std::cout<<"first-order interaction matrix (in 1/s) is: "<<std::endl;
        std::cout<<R1*ddBase.poly.cs_SI/ddBase.poly.b_SI<<std::endl;
        
        for(size_t k=0; k<mSize; k++)
        {
            std::cout<<"second-order interaction matrix (in 1/s) for "<<static_cast<int>(msVector(k))<<"-species is: "<<std::endl;
            std::cout<<R2[k]*ddBase.poly.cs_SI/ddBase.poly.b_SI<<std::endl;
        }

        for(const auto& ch : loopNucChannels)
        {// homogeneous-clustering channels feeding loop nucleation
            std::cout<<"loop-nucleation channel "<<static_cast<int>(msVector(ch.first.first))
                     <<" + "<<static_cast<int>(msVector(ch.first.second))
                     <<" -> "<<static_cast<int>(msVector(ch.first.first)+msVector(ch.first.second))
                     <<" : K = "<<ch.second*ddBase.poly.cs_SI/ddBase.poly.b_SI
                     <<" 1/s (0-D: omega_a+omega_b, or 4*omega_a when a==b)"<<std::endl;
        }
    }

    template<int dim>
    std::map<std::pair<int,int>,double> ClusterDynamicsParameters<dim>::getMap(const Eigen::Array<double,mSize*(mSize+1)/2,3> matrix_in) const
    {
        std::map<std::pair<int,int>,double> tempMap;
        for(int k=0; k<mSize*(mSize+1)/2; k++)
        {
            const int key0= matrix_in(k,0)<0? static_cast<int>(matrix_in(k,0)-msVector(0)): static_cast<int>(matrix_in(k,0)-msVector(0)-1);
            const int key1= matrix_in(k,1)<0? static_cast<int>(matrix_in(k,1)-msVector(0)): static_cast<int>(matrix_in(k,1)-msVector(0)-1);
            if(key0>key1)
            {
                throw std::runtime_error("Please keep the reaction map orders, key0<=key1.");
            }
            const std::pair<int,int> key(key0,key1);
            tempMap.insert(std::pair<std::pair<int,int>,double>(key,matrix_in(k,2)));
        }
        return tempMap;
    }

    template<int dim>
    Eigen::Matrix<double,ClusterDynamicsParameters<dim>::mSize,ClusterDynamicsParameters<dim>::mSize> ClusterDynamicsParameters<dim>::getR1() const
    {
        int vIndex, iIndex;
        for(int k=0;k<mSize;k++)
        {
            if(fabs(msVector(k)+1)<FLT_EPSILON)
            {
                vIndex=k;
            }
            if(fabs(msVector(k)-1)<FLT_EPSILON)
            {
                iIndex=k;
            }
        }
        if(iIndex-vIndex!=1 && msVector.matrix().squaredNorm()>0)
        { // Expected sequence should look like ... -2,-1,+1,+2,+3 ...
            throw std::runtime_error("intersitital Index must be after vacancy Index.");
        }
        
        Eigen::Array<double,1,mSize> aveD(Eigen::Array<double,1,mSize>::Zero());
        for(int k=0;k<mSize;k++)
        {
            aveD(k)=pow(detD.begin()->second(k),1.0/3.0);
        }
        
        Eigen::Array<double,1,mSize> rn(Eigen::Array<double,1,mSize>::Zero());
        for(int k=0;k<mSize;k++)
        {
            if(fabs(msVector(k)-1)<FLT_EPSILON || fabs(msVector(k)+1)<FLT_EPSILON)
            {
                rn(k)=pow(3.0*this->omega/4.0/M_PI,1.0/3.0);
            }
            else
            {
                rn(k)=pow(fabs(msVector(k)*this->omega/this->b/M_PI),1.0/2.0);
            }
        }
                
        Eigen::Array<double,1,mSize> alpha(Eigen::Array<double,1,mSize>::Zero());
        if(vIndex>0)
        { // Dissociation rate for v-clusters [Christien and Barbu 2005 JNM 346]
            for(int k=0;k<vIndex;k++)
            {
                auto it=reactionMap.find(std::pair<int,int>(k+1,vIndex));
                if(it!=reactionMap.end())
                {
                    alpha(k)=it->second*2.0*M_PI*rn(k)*aveD(vIndex)/omega*exp(-Eb(k)/kB/T);
                }
            }
        }
                
        if(mSize-1-iIndex>1)
        { // Dissociation rate for i-clusters
            for(int k=iIndex+1;k<mSize;k++)
            {
                auto it=reactionMap.find(std::pair<int,int>(iIndex,k-1));
                if(it!=reactionMap.end())
                {
                    // alpha(k)=it->second*2.0*M_PI*rn(k)*aveD(iIndex)/omega*exp(-Eb(k)/kB/T);
                    alpha(k)=it->second*8.0*M_PI*rn(iIndex)*aveD(iIndex)/omega*exp(-Eb(k)/kB/T);
                }
            }
        }
        
        Eigen::Matrix<double,mSize,mSize> tempR1(Eigen::Matrix<double,mSize,mSize>::Zero());
        for(int k=0;k<mSize;k++)
        {
            // Add other sinks
            tempR1(k,k) += (-otherSinks(k)*aveD(k));
            // Add dissociations
            tempR1(k,k) += (-alpha(k));
            if(k>0 && k<=vIndex)
            {
                tempR1(k,k-1) += alpha(k-1);
                tempR1(vIndex,k-1) += alpha(k-1);
            }
            else if(k>=iIndex && k<mSize-1)
            {
                tempR1(k,k+1) += alpha(k+1);
                tempR1(iIndex,k+1) += alpha(k+1);
            }
            
        }
        
        return tempR1;
    }

    template<int dim>
    std::vector<Eigen::Matrix<double,ClusterDynamicsParameters<dim>::mSize,ClusterDynamicsParameters<dim>::mSize>> ClusterDynamicsParameters<dim>::getR2() const
    {

        Eigen::Array<double,1,mSize> aveD(Eigen::Array<double,1,mSize>::Zero());
        for(int k=0;k<mSize;k++)
        {
            aveD(k)=pow(detD.begin()->second(k),1.0/3.0);
        }
        
        Eigen::Array<double,1,mSize> rn(Eigen::Array<double,1,mSize>::Zero());
        for(int k=0;k<mSize;k++)
        {
            if(fabs(msVector(k)-1)<FLT_EPSILON || fabs(msVector(k)+1)<FLT_EPSILON)
            {
                rn(k)=pow(3.0*this->omega/4.0/M_PI,1.0/3.0);
            }
            else
            {
                rn(k)=pow(fabs(msVector(k)*this->omega/this->b/M_PI),1.0/2.0);
            }
        }
        
        std::vector<Eigen::Matrix<double,mSize,mSize>> tempR2;
        Eigen::Matrix<double,mSize,mSize> R2sum(Eigen::Matrix<double,mSize,mSize>::Zero());
        double maxCoeff(0.0);
        
        for(size_t k=0;k<mSize;++k)
        {
            Eigen::Matrix<double,mSize,mSize> tempR2_k(Eigen::Matrix<double,mSize,mSize>::Zero());
            for(const auto& pair : reactionMap)
            {
                const auto key=pair.first;

                if(int(k)==key.first || int(k)==key.second)
                {// (key.first, key.second) consume k_specie
//                    if(msVector(key.second)>1.0 && use0DsinkStrength)
//                    { // 3D mobile + 1D immobile
//                        tempR2_k(key.first ,key.second) -= pair.second*2.0*M_PI*(rn(key.second))*(aveD(key.first))/omega;
//                        tempR2_k(key.second,key.first)  -= pair.second*2.0*M_PI*(rn(key.second))*(aveD(key.first))/omega;
//                    }
//                    else
//                    { // 3D mobile + 3D mobile
//                        tempR2_k(key.first ,key.second) -= pair.second*4.0*M_PI*(rn(key.first)+rn(key.second))*(aveD(key.first)+aveD(key.second))/omega;
//                        tempR2_k(key.second ,key.first) -= pair.second*4.0*M_PI*(rn(key.first)+rn(key.second))*(aveD(key.first)+aveD(key.second))/omega;
//                    }
                    tempR2_k(key.first ,key.second) -= pair.second*4.0*M_PI*(rn(key.first)+rn(key.second))*(aveD(key.first)+aveD(key.second))/omega;
                    tempR2_k(key.second ,key.first) -= pair.second*4.0*M_PI*(rn(key.first)+rn(key.second))*(aveD(key.first)+aveD(key.second))/omega;
                }
                else if( fabs(msVector(key.first)+msVector(key.second)-msVector(k))<FLT_EPSILON )
                {// (key.first, key.second) generate k_specie
                    const double same= (key.first==key.second) ? 0.5 : 1.0;
//                    if(msVector(key.second)>1.0 && use0DsinkStrength)
//                    { // 3D mobile + 1D immobile
//                        tempR2_k(key.first ,key.second) += same*pair.second*2.0*M_PI*(rn(key.second))*(aveD(key.first))/omega;
//                        tempR2_k(key.second ,key.first) += same*pair.second*2.0*M_PI*(rn(key.second))*(aveD(key.first))/omega;
//                    }
//                    else
//                    { // 3D mobile + 3D mobile
//                        tempR2_k(key.first ,key.second) += same*pair.second*4.0*M_PI*(rn(key.first)+rn(key.second))*(aveD(key.first)+aveD(key.second))/omega;
//                        tempR2_k(key.second ,key.first) += same*pair.second*4.0*M_PI*(rn(key.first)+rn(key.second))*(aveD(key.first)+aveD(key.second))/omega;
//                    }
                    tempR2_k(key.first ,key.second) += same*pair.second*4.0*M_PI*(rn(key.first)+rn(key.second))*(aveD(key.first)+aveD(key.second))/omega;
                    tempR2_k(key.second ,key.first) += same*pair.second*4.0*M_PI*(rn(key.first)+rn(key.second))*(aveD(key.first)+aveD(key.second))/omega;
                }
            }
            
            tempR2.push_back(tempR2_k);
            const double maxCoeffLocal(std::max(std::abs(tempR2.back().maxCoeff()),std::abs(tempR2.back().minCoeff())));
            
            if((tempR2.back()-tempR2.back().transpose()).norm()/maxCoeffLocal>FLT_EPSILON)
            {
                throw std::runtime_error("ClusterDynamicsParameters: R2_"+std::to_string(k)+" is not symmetric.");
            }
            maxCoeff=std::max(maxCoeff,maxCoeffLocal);
            R2sum+=tempR2.back()*msVector(k);
        }
        
        if(R2sum.norm()/maxCoeff>FLT_EPSILON)
        {
            //            throw std::runtime_error("Sum of R2's must be zero.");
            std::cout<<redBoldColor<<"Warning: Sum of R2 is not zero."<<defaultColor<<std::endl;
        }
        
        return tempR2;
    }

    template<int dim>
    std::map<std::pair<int,int>,double> ClusterDynamicsParameters<dim>::getLoopNucChannels() const
    {/*! Homogeneous-clustering (loop-nucleating) channels, with the SAME
      *  coefficient getR2() assembles into R2:
      *
      *      K_ab = p_ab * 4*pi*(r_a+r_b)*(D_a+D_b)/Omega ,
      *
      *  so that the mobile solve's per-species loss on this channel, K_ab*c_a*c_b
      *  (the residual assembles 0.5*c^T R2 c, and R2 carries -K_ab twice), is
      *  exactly what the loops gain in solveImmobileClusters(). K_ab equals the
      *  0-D coefficient: omega_a+omega_b for a!=b, 4*omega_a for a==b.
      *  A pair (a,b) is a
      *  nucleation channel when
      *
      *    (i)  m_a and m_b have the SAME sign -- a mixed pair is recombination or
      *         partial annihilation, not clustering; and
      *    (ii) NO mobile species carries the product size m_a+m_b, so the product
      *         is off the end of the mobile ladder and must become a loop embryo.
      *
      *  For the Zr3d_ghoniem network {v,i,2i,3i} this selects i+3i, 2i+2i and
      *  2i+3i and rejects i+i (->2i), i+2i (->3i) and every v-bearing pair. The
      *  0-D counts the first two in nucleation_rate_iL/aiL and all three in
      *  nucleation_content_i; the third contributes ~2e-7 of the total here, so
      *  including it in both is numerically immaterial and mass-consistent.
      */
        std::map<std::pair<int,int>,double> temp;
        if(detD.empty())
        {
            return temp;
        }

        // Same radii and orientation-averaged diffusivities getR2() uses.
        Eigen::Array<double,1,mSize> aveD(Eigen::Array<double,1,mSize>::Zero());
        Eigen::Array<double,1,mSize> rn(Eigen::Array<double,1,mSize>::Zero());
        for(int k=0;k<mSize;++k)
        {
            aveD(k)=pow(detD.begin()->second(k),1.0/3.0);
            if(fabs(msVector(k)-1)<FLT_EPSILON || fabs(msVector(k)+1)<FLT_EPSILON)
            {
                rn(k)=pow(3.0*this->omega/4.0/M_PI,1.0/3.0);
            }
            else
            {
                rn(k)=pow(fabs(msVector(k)*this->omega/this->b/M_PI),1.0/2.0);
            }
        }

        for(const auto& pair : reactionMap)
        {
            const int a(pair.first.first);
            const int c(pair.first.second);
            if(pair.second<=0.0)
            {// channel switched off in the material file
                continue;
            }
            if(msVector(a)*msVector(c)<=0.0)
            {// opposite polarity (or a zero-size species): not clustering
                continue;
            }
            const double product(msVector(a)+msVector(c));
            bool productIsMobile(false);
            for(int k=0;k<mSize;++k)
            {
                if(fabs(msVector(k)-product)<FLT_EPSILON)
                {
                    productIsMobile=true;
                    break;
                }
            }
            if(productIsMobile)
            {// the ladder absorbs it; not a loop embryo
                continue;
            }
            temp.emplace(pair.first,pair.second*4.0*M_PI*(rn(a)+rn(c))*(aveD(a)+aveD(c))/omega);
        }
        return temp;
    }

    template<int dim>
    Eigen::Array<double,1,ClusterDynamicsParameters<dim>::iSize/2> ClusterDynamicsParameters<dim>::getImmobileSpeciesBurgersMagnitude(const std::map<size_t,Grain<dim>>& grains) const
    {
        Eigen::Array<double,1,iSize/2> temp(Eigen::Array<double,1,iSize/2>::Zero());
        const Eigen::Matrix<double,dim,dim> lat(grains.begin()->second.singleCrystal->latticeBasis);
        const Eigen::Matrix<double,dim,iSize/2> localBurgers(lat*immobileSpeciesBurgers);
        
        for(size_t k=0; k<iSize/2; k++)
        {
            temp(k) = (localBurgers.col(k)).norm();
        }
        
        std::cout<< "immobileSpeciesBurgersMagnitude: "<<temp<<std::endl;
        return temp;
    }


    template<int dim>
    std::map<size_t,std::vector<Eigen::Matrix<double,dim,dim>>> ClusterDynamicsParameters<dim>::getD(const std::map<size_t,Grain<dim>>& grains) const
    {
        std::map<size_t,std::vector<Eigen::Matrix<double,dim,dim>>> temp;
        
        for(const auto& grainPair : grains)
        {
            auto grainIter(temp.emplace(grainPair.first,std::vector<Eigen::Matrix<double,dim,dim>>()));
            
            for(size_t k=0; k<mSize; k++)
            {
                Eigen::Matrix<double,dim,dim> Dlocal;
                Dlocal<<   msD0(k,0)*exp(-msEm(k,0)/kB/T), msD0(k,1)*exp(-msEm(k,1)/kB/T), msD0(k,2)*exp(-msEm(k,2)/kB/T),
                msD0(k,1)*exp(-msEm(k,1)/kB/T), msD0(k,3)*exp(-msEm(k,3)/kB/T), msD0(k,4)*exp(-msEm(k,4)/kB/T),
                msD0(k,2)*exp(-msEm(k,2)/kB/T), msD0(k,4)*exp(-msEm(k,4)/kB/T), msD0(k,5)*exp(-msEm(k,5)/kB/T);
                
                // g^T inv(Dg)*g = (C2G*c)^T*inv(Dg)*C2G*c = c^T*C2G^T*inv(Dg)*C2G*c= c^T*inv(inv(C2G)*Dg*inv(C2G^T))*c
                // c^T*inv(C2G^T*Dg*C2G)*c
                // Dc=C2G^T*Dg*C2G
                // Dg = C2G*Dc*C2G^T
                const Eigen::Matrix<double,dim,dim> Dglobal(grainPair.second.singleCrystal->C2G*Dlocal*grainPair.second.singleCrystal->C2G.transpose());
                grainIter.first->second.emplace_back(Dglobal);
                
                if( fabs(Dglobal.eigenvalues().imag()(0)*Dglobal.eigenvalues().imag()(1)*Dglobal.eigenvalues().imag()(2)) > FLT_EPSILON)
                {
                    throw std::runtime_error("Diffusion tensor does not have proper eigen values.");
                }
                std::cout<<greenColor<<"  Diffusion coefficient for "<<k<<":\n"<<Dglobal<<std::endl;
            }
        }
        
        return temp;
    }


    template<int dim>
    std::vector<Eigen::Matrix<double,dim,dim>> ClusterDynamicsParameters<dim>::getDlocal() const
    {
        std::vector<Eigen::Matrix<double,dim,dim>> temp;
        
        for(size_t k=0; k<mSize; k++)
        {
            Eigen::Matrix<double,dim,dim> Dlocal;
            Dlocal<<   msD0(k,0)*exp(-msEm(k,0)/kB/T), msD0(k,1)*exp(-msEm(k,1)/kB/T), msD0(k,2)*exp(-msEm(k,2)/kB/T),
            msD0(k,1)*exp(-msEm(k,1)/kB/T), msD0(k,3)*exp(-msEm(k,3)/kB/T), msD0(k,4)*exp(-msEm(k,4)/kB/T),
            msD0(k,2)*exp(-msEm(k,2)/kB/T), msD0(k,4)*exp(-msEm(k,4)/kB/T), msD0(k,5)*exp(-msEm(k,5)/kB/T);
            
            temp.emplace_back(Dlocal);
        }
        
        return temp;
    }

    template<int dim>
    std::map<size_t,std::vector<Eigen::Matrix<double,dim,dim>>> ClusterDynamicsParameters<dim>::getInvD() const
    {
        std::map<size_t,std::vector<Eigen::Matrix<double,dim,dim>>> temp;
        for(const auto& pair : D)
        {
            auto mapIter(temp.emplace(pair.first,std::vector<Eigen::Matrix<double,dim,dim>>()));
            for(const auto& d : pair.second)
            {
                mapIter.first->second.push_back(d.inverse());
            }
        }
        return temp;
    }

    template<int dim>
    std::map<size_t,Eigen::Array<double,ClusterDynamicsParameters<dim>::mSize,1>> ClusterDynamicsParameters<dim>::getDetD() const
    {
        std::map<size_t,Eigen::Array<double,mSize,1>> temp;// (1,mSize);
        for(const auto& pair : D)
        {
            auto mapIter(temp.emplace(pair.first,Eigen::Array<double,mSize,1>()));
            for(int k=0;k<mSize;++k)
            {
                mapIter.first->second(k)=pair.second[k].determinant();
            }
        }
        return temp;
    }

    template<int dim>
    Eigen::Array<double,1,ClusterDynamicsParameters<dim>::iSize> ClusterDynamicsParameters<dim>::getInitLoopSinks(const Eigen::Array<double,1,iSize> initloopSinks_SI, const double b_SI) const
    {
        const Eigen::Array<double,1,iSize/2> initDen = initloopSinks_SI.template block<1,iSize/2>(0,0)*b_SI*b_SI*b_SI;
        // const Eigen::Array<double,1,iSize/2> initRad = initloopSinks_SI.template block<1,iSize/2>(0,iSize/2)/b_SI;
        const Eigen::Array<double,1,iSize/2> initRad = initloopSinks_SI.template block<1,iSize/2>(0,iSize/2);
        
        Eigen::Array<double,1,iSize> temp;
        temp<< initDen,initRad;
        
        return temp;
    }

    template<int dim>
    Eigen::Array<double,1,ClusterDynamicsParameters<dim>::mSize> ClusterDynamicsParameters<dim>::equilibriumMobileConcentration(const double& stressTrace) const
    {
        
        return msVector.abs()*exp(-(msEf-stressTrace*msVector*msRelRelaxVol*omega/3.0)/kB/T);
    }

    template<int dim>
    Eigen::Array<double,1,ClusterDynamicsParameters<dim>::mSize> ClusterDynamicsParameters<dim>::boundaryMobileConcentration(const double& stressTrace,const double& normalTraction) const
    {
        return equilibriumMobileConcentration(stressTrace)*exp(-normalTraction*msVector*omega/kB/T);
    }

    template<int dim>
    Eigen::Array<double,1,ClusterDynamicsParameters<dim>::mSize> ClusterDynamicsParameters<dim>::dislocationMobileConcentration(const VectorDim& b,
                                                                                                                                const VectorDim& t,
                                                                                                                                const VectorDim& fPK,
                                                                                                                                const MatrixDim& stress) const
    {
        const VectorDim bxt(b.cross(t));
        const double bxtNorm2(bxt.squaredNorm());
        if(bxtNorm2>FLT_EPSILON)
        {
            const double fc(fPK.dot(bxt));
            return equilibriumMobileConcentration(stress.trace())*exp(-(msVector*omega*fc)/(kB*T*(bxtNorm2+0.05*b.squaredNorm())));
        }
        else
        {// screw direction
            return Eigen::Array<double,1,ClusterDynamicsParameters<dim>::mSize>::Zero();
        }
    }


    // sigmoid function of number of vacancies FOR ALL SPECIES
    template<int dim>
    Eigen::Array<double,1,ClusterDynamicsParameters<dim>::iSize/2> ClusterDynamicsParameters<dim>::sigmoid(const Eigen::Array<double,1,iSize/2>& n) const
    {
        // parameters
        const Eigen::Array<double,1,iSize/2> n0 = ((nmin+nmax)*(0.5) - n_s);
        const Eigen::Array<double,1,iSize/2> w = w0*(nmax-nmin);
        
        return 1.0/(1.0+exp(-(n - n0)/w));
    }

    // pyramid radius function of V
    template<int dim>
    Eigen::Array<double,1,ClusterDynamicsParameters<dim>::iSize/2> ClusterDynamicsParameters<dim>::rpyr(const Eigen::Array<double,1,iSize/2>& n) const
    {
        return pow(n*omega/sqrt(8),1.0/3.0); // Vp = sqrt(8)*r^3;
    }

    // loop radius function of V
    template<int dim>
    Eigen::Array<double,1,ClusterDynamicsParameters<dim>::iSize/2> ClusterDynamicsParameters<dim>::rloop(const Eigen::Array<double,1,iSize/2>& n) const
    {
        return sqrt(n*omega/(M_PI*b*immobileSpeciesBurgersMagnitude)); // Vl = pi*b*r^2;
    }

    template<int dim>
    Eigen::Array<double,1,ClusterDynamicsParameters<dim>::iSize/2> ClusterDynamicsParameters<dim>::sigmoidalVectorInterpolation(const Eigen::Array<double,1,iSize/2>& CI, const Eigen::Array<double,1,iSize/2>& N, const Eigen::Array<double,1,iSize/2>& lowValue, const Eigen::Array<double,1,iSize/2>& highValue) const
    {
        const Eigen::Array<double,1,iSize/2> n = CI/N/omega;
        
        return highValue*sigmoid(n) + lowValue*(1.0 - sigmoid(n));
    }

    template<int dim>
    Eigen::Array<double,1,ClusterDynamicsParameters<dim>::iSize/2> ClusterDynamicsParameters<dim>::clusterRadius(const Eigen::Array<double,1,iSize/2>& CI, const Eigen::Array<double,1,iSize/2>& N) const
    {
        const Eigen::Array<double,1,iSize/2> n = CI/N/omega;
        
        return sigmoidalVectorInterpolation(CI,N,rpyr(n),rloop(n));
    }

    template<int dim>
    Eigen::Array<double,1,ClusterDynamicsParameters<dim>::iSize/2> ClusterDynamicsParameters<dim>::clusterDensity(const Eigen::Array<double,1,iSize/2>& CI, const Eigen::Array<double,1,iSize/2>& N) const
    {
        const Eigen::Array<double,1,iSize/2> n = CI/N/omega;
        const Eigen::Array<double,1,iSize/2> LoopS = 2.0*M_PI*rloop(n)*N;
        const Eigen::Array<double,1,iSize/2> PyrS = a_bp*4.0*M_PI*rpyr(n)*N;
        
        return sigmoidalVectorInterpolation(CI,N,PyrS,LoopS);
    }

    template<int dim>
    Eigen::Array<double,dim,dim> ClusterDynamicsParameters<dim>::sigmoidalMatrixInterpolation(const Eigen::Array<double,1,iSize/2>& CI, const Eigen::Array<double,1,iSize/2>& N, const Eigen::Array<double,dim,dim>& lowValue, const Eigen::Array<double,dim,dim>& highValue, const int& index) const
    {
        const Eigen::Array<double,1,iSize/2> n = CI/N/omega;
        
        return highValue*sigmoid(n)(index) + lowValue*(1.0 - sigmoid(n)(index));
    }

    template<int dim>
    Eigen::Array<double,1,ClusterDynamicsParameters<dim>::iSize/2> ClusterDynamicsParameters<dim>::sigmoidalPlotVectorInterpolation(const Eigen::Array<double,1,iSize/2>& CI, const Eigen::Array<double,1,iSize/2>& N, const Eigen::Array<double,1,iSize/2>& lowValue, const Eigen::Array<double,1,iSize/2>& highValue) const
    {
        const Eigen::Array<double,1,iSize/2> n = CI/N/omega;
        
        return lowValue*sigmoid(n) + highValue*(1.0 - sigmoid(n));
    }

    template<int dim>
    Eigen::Array<double,1,ClusterDynamicsParameters<dim>::iSize/2> ClusterDynamicsParameters<dim>::clusterPlotRadius(const Eigen::Array<double,1,iSize/2>& CI, const Eigen::Array<double,1,iSize/2>& N) const
    {
        const Eigen::Array<double,1,iSize/2> n = CI/N/omega;
        
        return sigmoidalPlotVectorInterpolation(CI,N,rpyr(n),rloop(n));
    }

    /**********************************************************************/
    /* Diffusional-anisotropy-difference (DAD) capture efficiencies.
     * Deliverable D1/M1 Eq. (15):
     *      p_m = (D_c^m / D_a^m)^(1/6)
     *      Z_aL,m = Z0_m (p_m + p_m^-2)/2      (prismatic <a> loops)
     *      Z_cL,m = Z0_m  p_m                  (basal <c> loops)
     * CALIBRATION (adopted). p_m and Z0_m are NOT derived from the diffusion
     * tensors here. They are supplied through `dadAnisotropy` and `dadZ0` in the
     * material file, fitted so that Eq. (15) reproduces the 0-D empirical
     * SYMMETRIC forms of Eqs. (63)-(64) as closely as possible:
     *
     *     Z^a_i = 1+delta_i    Z^a_v = 1-delta_v
     *     Z^c_i = 1-delta_i    Z^c_v = 1+delta_v
     *
     * Because Eq. (15) carries two free parameters per species (Z0_m, p_m) and
     * the 0-D forms impose two targets (Z_a, Z_c), the system inverts EXACTLY:
     *
     *     2 Z_a / Z_c = 1 + p^-3     ->  p  = (2 Z_a/Z_c - 1)^(-1/3)
     *                                    Z0 = Z_c / p
     *
     * so the fit is exact to machine precision, not a least-squares compromise.
     * The fitted values are also physically sensible for h.c.p. Zr: p_v>1
     * (vacancies faster along c) and p_i<1 (interstitials faster in the basal
     * plane).
     *
     * NOTE the consequence: the 0-D symmetric forms impose Z^a_m + Z^c_m = 2,
     * a normalization Eq. (15) does not naturally satisfy, so p_m no longer
     * equals (D_c/D_a)^(1/6) of the material file's migration tensors. The bias
     * is therefore CALIBRATED to the reduced model rather than generated from
     * the tensors, and it is tied to the temperature at which delta_i(T) was
     * evaluated (delta_i is T-interpolated between 300 K and 600 K anchors in
     * the 0-D model). Refit dadAnisotropy/dadZ0 if T changes.
     *
     * Returned: row 0 = <c> (vacancy) loops, row 1 = <a> (interstitial) loops.
     */
    template<int dim>
    Eigen::Array<double,2,ClusterDynamicsParameters<dim>::mSize> ClusterDynamicsParameters<dim>::loopDADbias() const
    {
        Eigen::Array<double,2,mSize> Z(Eigen::Array<double,2,mSize>::Ones());
        for(int k=0;k<mSize;++k)
        {
            const double pm(std::max(dadAnisotropy(k),double(FLT_EPSILON)));
            Z(0,k)=dadZ0(k)*pm;                             // <c> loops,  Eq. (15)
            Z(1,k)=dadZ0(k)*0.5*(pm+std::pow(pm,-2.0));     // <a> loops,  Eq. (15)
        }
        return Z;
    }

    template struct ClusterDynamicsParameters<3>;

}
#endif
