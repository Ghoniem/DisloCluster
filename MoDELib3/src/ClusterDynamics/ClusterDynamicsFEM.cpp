/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_ClusterDynamicsFEM_cpp_
#define model_ClusterDynamicsFEM_cpp_

#include <cmath>
#include <limits>
#include <vector>
#include <ClusterDynamicsFEM.h>
#include <ExternalAndInternalBoundary.h>
#include <Fix.h>
#include <ImmobileSinks.h>
#include <TextFileParser.h>

namespace model
{

    template<int dim>
    FluxMatrix<dim>::FluxMatrix(const ClusterDynamicsParameters<dim>& cdp_in) :
    /* init */cdp(cdp_in)
    {
        
    }

    template<int dim>
    const typename FluxMatrix<dim>::MatrixType FluxMatrix<dim>::operator() (const ElementType& elem, const BaryType&) const
    {
        const size_t grainID(elem.simplex.region->regionID);
        MatrixType D(MatrixType::Zero());
        for(size_t k=0;k<mSize;++k)
        {
            D.template block<dim,dim>(k*dim,k*dim)=-cdp.D.at(grainID)[k]/cdp.omega;
        }
        return D;
    }

    template struct FluxMatrix<3>;

template<int dim>
InvDscaling<dim>::InvDscaling(const ClusterDynamicsParameters<dim>& cdp_in) :
/* init */cdp(cdp_in)
{
    
}

template<int dim>
const typename InvDscaling<dim>::MatrixType InvDscaling<dim>::operator() (const ElementType& elem, const BaryType&) const
{
    const size_t grainID(elem.simplex.region->regionID);
    MatrixType InvDscaling(MatrixType::Zero());
    for(size_t k=0;k<mSize;++k)
    {
        InvDscaling(k,k)=3.0/(cdp.D.at(grainID)[k].trace()/cdp.omega);
//        InvDscaling(k,k)=(cdp.D.at(grainID)[k].trace()/cdp.omega)/3.0;
//        InvDscaling(k,k)=1.0;

    }
    return InvDscaling;
}

template struct InvDscaling<3>;


    template<int dim>
    bool ClusterDynamicsFEM<dim>::getUseImmobileSolver(const DislocationDynamicsBase<dim>& ddBase)
    {/*! `useImmobileSolver` from DD.txt, defaulting to 1 when the key is absent.
      *
      *  TextFileParser::readScalar throws on a missing key, and every DD.txt
      *  written before this flag existed lacks it. Defaulting on a throw keeps
      *  those cases running exactly as they did.
      */
        try
        {
            return bool(TextFileParser(ddBase.simulationParameters.traitsIO.ddFile).readScalar<int>("useImmobileSolver",true));
        }
        catch(const std::runtime_error&)
        {
            return true;
        }
    }

    template<int dim>
    double ClusterDynamicsFEM<dim>::ddScalar(const DislocationDynamicsBase<dim>& ddBase,const std::string& key,const double& fallback)
    {/*! An optional DD.txt scalar. TextFileParser throws on a missing key, and
      *  none of these exist in a DD.txt written before they did.
      */
        try
        {
            return TextFileParser(ddBase.simulationParameters.traitsIO.ddFile).readScalar<double>(key,true);
        }
        catch(const std::runtime_error&)
        {
            return fallback;
        }
    }

    template<int dim>
    ClusterDynamicsFEM<dim>::ClusterDynamicsFEM(const DislocationDynamicsBase<dim>& ddBase_in,const ClusterDynamicsParameters<dim>& cdp_in) :
    /* init */ ddBase(ddBase_in)
    /* init */,cdp(cdp_in)
    /* init */,iDs(cdp)
    /* init */,mobileClusters(ddBase.fe->template trial<'m',mSize>())
    /* init */,mobileGrad(grad(mobileClusters))
    /* init */,mobileFlux(FluxMatrix<dim>(cdp)*mobileGrad)
    /* init */,immobileClusters(ddBase.fe->template trial<'i',iSize>())
    /* init */,nodeListInternalExternal(ddBase.isPeriodicDomain ? -1 : ddBase.fe->template createNodeList<ExternalAndInternalBoundary>())
    /* init */,mobileClustersIncrement(ddBase.fe->template trial<'d',mSize>())
    /* init */,dV(ddBase.fe->template domain<EntireDomain,dVorder,GaussLegendre>())
//    /* init */,mBWF((test(this->mobileGrad),-cdp.omega*this->mobileFlux)*dV)
    // cdp.omega, NOT ddBase.poly.Omega. FluxMatrix divides D by cdp.omega, so the
    // two must be the same quantity for the flux to come out as -D*grad(c). They
    // were identical until the CD atomic volume became settable independently of
    // the lattice cell volume (atomicVolume_SI); using poly.Omega here would leave
    // the diffusion operator scaled by poly.Omega/cdp.omega = 3.98 relative to the
    // reaction terms.
    /* init */,mBWF((test(grad(iDs*mobileClusters)),-cdp.omega*this->mobileFlux)*dV)
    /* init */,dmBWF((test(grad(iDs*mobileClustersIncrement)),-cdp.omega*(FluxMatrix<dim>(this->cdp)*grad(mobileClustersIncrement)))*dV)
    /* init */,mSolver(true,FLT_EPSILON)
    /* init */,solverInitialized(false)
//    /* init */,cascadeGlobalProduction(((test(this->mobileClusters),make_constant(this->cdp.G))*dV).globalVector())
    /* init */,cascadeGlobalProduction(((test(iDs*this->mobileClusters),make_constant(this->cdp.G))*dV).globalVector())
    /* init */,useImmobileSolver(getUseImmobileSolver(ddBase_in))
    /* init */,mobileSolverTolerance(ddScalar(ddBase_in,"mobileSolverTolerance",1.0e-5))
    /* init */,mobileSolverMaxIterations(int(ddScalar(ddBase_in,"mobileSolverMaxIterations",200)))
    /* init */,mobileSolverRelaxation(ddScalar(ddBase_in,"mobileSolverRelaxation",1.0))
    /* init */,mobileSolverErrorMode(int(ddScalar(ddBase_in,"mobileSolverErrorMode",0)))
    /* init */,mobileSolverClampInLoop(int(ddScalar(ddBase_in,"mobileSolverClampInLoop",1)))
    {
        mobileClustersIncrement.setConstant(Eigen::Matrix<double,mSize,1>::Zero());
        mobileClusters.setConstant(cdp.equilibriumMobileConcentration(0.0).matrix().transpose());
        immobileClusters.setConstant(Eigen::Matrix<double,iSize,1>::Zero());
    }

    template<int dim>
    void ClusterDynamicsFEM<dim>::clampMobileClusters()
    {/*! Positivity floor on the mobile concentrations, mirroring ZrMicro, which
      *  applies y = max(y, C_floor) before every rate evaluation.
      *
      *  The imposed Dirichlet values are restored afterwards and are NOT floored.
      *  They must not be: the grain-boundary condition is the thermal equilibrium
      *  concentration, and for the interstitial species that is
      *  exp(-3.0 eV/kT) = 4.1e-27 at 573 K -- far BELOW the 1e-20 floor. Flooring
      *  it would raise the boundary value by seven orders and quietly disable the
      *  grain boundary as a sink for interstitials, which is the one piece of
      *  physics the 3-D model adds over the 0-D.
      */
        Eigen::VectorXd mDof(mobileClusters.dofVector());
        mDof=mDof.cwiseMax(cdp.concentrationFloor);
        for(const auto& dc : TrialBase<MobileTrialType>::dirichletConditions())
        {
            mDof(dc.first)=dc.second;
        }
        mobileClusters=mDof;
    }

    template<int dim>
    void ClusterDynamicsFEM<dim>::solveMobileClusters()
    {
        std::cout<<", mobile solver "<<std::flush;
        mobileClusters=mSolver.solve(cascadeGlobalProduction);
        clampMobileClusters();

        if(this->cdp.computeReactions)
        {/*! Fixed-point/Newton loop on the mobile reaction terms.
          *
          *  Upstream (both MoDELib2-NNL and MoDELib-fullCD) this is
          *  `while(cError>cTol)` with cTol = 1e-5, no iteration cap and no
          *  stagnation test. That is safe only while the iteration always
          *  converges. It does not: on a mobile field solved against an
          *  immobile state supplied by an external march, cError falls to
          *  ~1e-4 in nine iterations and then stagnates, oscillating around
          *  5e-5 indefinitely. With no cap the solve never returns.
          *
          *  The loop below is the same iteration with four run-time controls,
          *  all read from DD.txt and all defaulting to the historical
          *  behaviour except the iteration cap:
          *
          *    mobileSolverTolerance      cTol                        (1e-5)
          *    mobileSolverMaxIterations  cap; 0 = unlimited           (200)
          *    mobileSolverRelaxation     w in c += w*increment        (1.0)
          *    mobileSolverErrorMode      0 = all dofs (historical)
          *                               1 = exclude floored+Dirichlet dofs
          *                               2 = max nodal |dc|/(|c|+floor)
          *
          *  The best iterate seen is retained, so hitting the cap returns the
          *  most converged field rather than whatever the last oscillation
          *  happened to leave behind.
          */
            const double cTol(mobileSolverTolerance);
            const int maxIter(mobileSolverMaxIterations);
            const double relax(mobileSolverRelaxation);
            const int errMode(mobileSolverErrorMode);
            double cError(1.0);
            double bestError(std::numeric_limits<double>::max());
            Eigen::VectorXd bestDof(mobileClusters.dofVector());
            int iter(0);
            // Active set of the positivity floor, carried between iterations.
            // The count alone does not settle the question: a cycle can hold
            // the cardinality fixed while membership churns, one dof entering
            // as another leaves. `entered`/`left` measure that directly.
            std::vector<char> activePrev;
            while(cError>cTol)
            {
                if(maxIter>0 && iter>=maxIter)
                {
                    std::cout<<"    mobile solver STOPPED at "<<iter
                             <<" iterations: cError="<<cError
                             <<" > cTol="<<cTol
                             <<"; keeping best iterate (cError="<<bestError<<")"
                             <<std::endl;
                    mobileClusters=bestDof;
                    break;
                }
                ++iter;
                const auto R1((this->cdp.R1cd).eval());
                auto bWF_R1((test(iDs*mobileClustersIncrement),R1*(-1.0*mobileClustersIncrement))*dV); // THIS SHOULD BE STORED SINCE IT IS ALWAYS THE SAME
                auto lWF_R1((test(iDs*mobileClustersIncrement),eval(R1*mobileClusters))*dV);
                
                SecondOrderReaction<MobileTrialType> R2(mobileClusters,this->cdp);
                auto bWF_R2((test(iDs*mobileClustersIncrement),R2*(-1.0*mobileClustersIncrement))*dV);
                auto lWF_R2((test(iDs*mobileClustersIncrement),eval(R2*(0.5*mobileClusters)))*dV);
                
                // Immobile (loop) sinks -- 0-D loop_abs term, Eqs. (83)-(89),
                // evaluated from the LOCAL immobile state (Eqs. 11-14). The
                // network sink is already in R1 via otherSinks; the
                // grain-boundary sink is imposed spatially by the Dirichlet BC
                // on the mobile species and must not be added here.
                ImmobileSinks<ImmobileTrialType,mSize> RI(immobileClusters,this->cdp);
                auto bWF_RI((test(iDs*mobileClustersIncrement),RI*(-1.0*mobileClustersIncrement))*dV);
                auto lWF_RI((test(iDs*mobileClustersIncrement),eval(RI*mobileClusters))*dV);

                Eigen::SparseMatrix<double,Eigen::RowMajor> AcIR;
                AcIR.resize(mobileClustersIncrement.gSize(),mobileClustersIncrement.gSize());
                std::vector<Eigen::Triplet<double>> globalTripletsR((bWF_R1+bWF_R2+bWF_RI).globalTriplets());
                AcIR.setFromTriplets(globalTripletsR.begin(),globalTripletsR.end());

                // NOTE: switching this to the direct (SparseLU) branch was tried
                // and does NOT work -- compute() itself fails on the ~96k-dof 3-D
                // Newton matrix, while the LLT factorization of the pure diffusion
                // operator in mSolver succeeds. The iterative branch is retained;
                // the first-step convergence problem is an initial-guess problem,
                // not a linear-algebra one (see ZR3D_GHONIEM_CHANGES.md, issue 12).
                MobileReactionSolverType rSolver(false,FLT_EPSILON);
                rSolver.compute(dmBWF+bWF_R1+bWF_R2+bWF_RI);
                mobileClustersIncrement=rSolver.solve(cascadeGlobalProduction-mSolver.getA()*mobileClusters.dofVector()+(lWF_R1+lWF_R2+lWF_RI).globalVector());
                
                const Eigen::VectorXd dofBefore(mobileClusters.dofVector());
                Eigen::MatrixXd cOld(dofBefore);
                cOld.resize(mSize,mobileClusters.gSize()/mSize);
                mobileClusters += relax*mobileClustersIncrement.dofVector();

                // How many dofs the floor actually moves. If this is large the
                // step the error test sees is not the step Newton proposed, and
                // that is the mechanism behind the stagnation.
                const Eigen::VectorXd dofPre(mobileClusters.dofVector());
                if(mobileSolverClampInLoop)
                {
                    clampMobileClusters();
                }
                const Eigen::VectorXd dofPost(mobileClusters.dofVector());
                const long nClamped((dofPost.array()>dofPre.array()).count());

                std::vector<char> active(size_t(dofPost.size()),0);
                for(long j=0;j<dofPost.size();++j)
                {
                    active[size_t(j)]=(dofPost(j)>dofPre(j))?1:0;
                }
                long entered(0),left(0);
                if(activePrev.size()==active.size())
                {
                    for(size_t j=0;j<active.size();++j)
                    {
                        if(active[j] && !activePrev[j]) ++entered;
                        if(!active[j] && activePrev[j]) ++left;
                    }
                }
                activePrev=active;

                Eigen::MatrixXd cNew(dofPost);
                cNew.resize(mSize,mobileClusters.gSize()/mSize);

                Eigen::VectorXd relErr(Eigen::VectorXd::Zero(mSize));
                if(errMode==2)
                {// max over dofs of |dc|/(|c|+floor): a componentwise measure,
                 // so a converged decade-spanning field is not judged by its
                 // largest entry alone.
                    const Eigen::VectorXd num((dofPost-dofBefore).cwiseAbs());
                    const Eigen::VectorXd den(dofPost.cwiseAbs().array()+cdp.concentrationFloor);
                    const Eigen::VectorXd r((num.array()/den.array()).matrix());
                    for(int k=0;k<mSize;++k)
                    {
                        double m(0.0);
                        for(int j=k;j<r.size();j+=mSize)
                        {
                            m=std::max(m,r(j));
                        }
                        relErr(k)=m;
                    }
                }
                else
                {
                    Eigen::MatrixXd dC(cNew-cOld);
                    if(errMode==1)
                    {// zero out dofs the clamp touched and the imposed Dirichlet
                     // dofs, so the test measures only where the iteration is
                     // free to move.
                        Eigen::VectorXd mask(Eigen::VectorXd::Ones(dofPost.size()));
                        for(long j=0;j<dofPost.size();++j)
                        {
                            if(dofPost(j)>dofPre(j))
                            {
                                mask(j)=0.0;
                            }
                        }
                        for(const auto& dc : TrialBase<MobileTrialType>::dirichletConditions())
                        {
                            mask(dc.first)=0.0;
                        }
                        Eigen::MatrixXd maskM(mask);
                        maskM.resize(mSize,mobileClusters.gSize()/mSize);
                        dC=(dC.array()*maskM.array()).matrix();
                    }
                    const Eigen::VectorXd absErr(dC.rowwise().norm());
                    const Eigen::VectorXd cNewNorm((cNew.rowwise().norm().array()+1.e-50).matrix());
                    relErr=(absErr.array()/cNewNorm.array()).matrix();
                }

                int worst(0);
                cError=relErr.maxCoeff(&worst);
                if(cError<bestError)
                {
                    bestError=cError;
                    bestDof=mobileClusters.dofVector();
                }
                static const char* mNames[4]={"Cv","Ci","C2i","C3i"};
                std::cout<<"convergenceError="<<cError
                         <<" [it "<<iter
                         <<", worst "<<(worst<4?mNames[worst]:"?")
                         <<", clamped "<<nClamped<<"/"<<dofPost.size()
                         <<", set +"<<entered<<"/-"<<left
                         <<"]"<<std::endl;
            }
            if(cError<=cTol)
            {
                std::cout<<"    mobile solver converged in "<<iter
                         <<" iterations (cError="<<cError<<")"<<std::endl;
            }
            if(!mobileSolverClampInLoop)
            {// the floor was deferred out of the iteration; apply it once now
                clampMobileClusters();
            }
        }
        // Find immobile rate
        
        
        // Find diffusive-displacement rate
        
    }

    /**********************************************************************/
    /*! Immobile (loop) population update -- Deliverable D1/M1 Sec. 2.2.
     *
     *  The immobile fields obey spatially-resolved ORDINARY differential
     *  equations (no spatial derivatives): a density equation
     *
     *      dn_I/dt = Ndot_nuc - a_n - DeltaN                      (Eq. 34)
     *
     *  and a mass-conserving content equation
     *
     *      dc_I/dt = Gdot_c + P_G - A_c - A_c^emit - DeltaC       (Eq. 39)
     *
     *  so they are integrated node-by-node against the frozen mobile field,
     *  with no FEM assembly. DOF layout (see ClusterDynamics::output):
     *  index = node*iSize + component, components [0,nF) = densities n_k,
     *  [nF,2nF) = contents c_k, with family order {vL, a1, a2, a3}.
     *
     *  Absorption uses MoDELib's own sink-strength convention: the rate at which
     *  family k absorbs mobile species m is  D_m * Z_km * S_k * c_m, where S_k is
     *  the geometric sink strength returned by clusterDensity() -- which already
     *  interpolates between the bi-pyramid law alpha_bp*4*pi*r*N (Eq. 14) and the
     *  planar loop law 2*pi*r*N (Eqs. 12-13) through the sigmoid of Eq. (30).
     *  Z_km is the DAD bias generated from the diffusion tensors, Eq. (15).
     *
     *  A sub-stepped explicit update is used with a positivity floor; the loop
     *  kinetics are slow compared with the mobile manifold (that is the premise
     *  of the two-time-scale split), so the dose step is resolved cheaply.
     */
    template<int dim>
    void ClusterDynamicsFEM<dim>::solveImmobileClusters()
    {
        if(iSize<=0)
        {
            return;
        }
        std::cout<<", immobile solver "<<std::flush;

        constexpr int nF(iSize/2);                       // number of immobile families
        const double dtTotal(ddBase.simulationParameters.dtMax);
        if(dtTotal<=0.0)
        {
            return;
        }

        const Eigen::Array<double,2,mSize> Z(cdp.loopDADbias());   // row0 <c>, row1 <a>

        // orientationally-averaged diffusivity of each mobile species
        Eigen::Array<double,1,mSize> Dbar(Eigen::Array<double,1,mSize>::Zero());
        if(!cdp.detD.empty())
        {
            for(int m=0;m<mSize;++m)
            {
                Dbar(m)=std::pow(std::max(cdp.detD.begin()->second(m),0.0),1.0/3.0);
            }
        }

        // polarity of each immobile family from immobileSpeciesVector (<0 vacancy)
        Eigen::Array<int,1,nF> isVacancyFamily(Eigen::Array<int,1,nF>::Zero());
        for(int k=0;k<nF;++k)
        {
            isVacancyFamily(k)=(cdp.immobileSpeciesVector(k)<0.0)?1:0;
        }

        const Eigen::VectorXd& mDof(mobileClusters.dofVector());
        Eigen::VectorXd iDof(immobileClusters.dofVector());
        const size_t nNodes(mobileClusters.fe().nodes().size());

        // Positivity floors, from ZrMicro's C_floor. That floor is stated per ATOM
        // for every one of its 12 species, so the content (a volume fraction, which
        // is numerically the per-atom content) takes it directly, while the number
        // density -- carried here per b^3 -- takes C_floor/Omega.
        const double cFloor(cdp.concentrationFloor);
        const double nFloor(cdp.concentrationFloor/cdp.omega);
        const int nSub(20);                // explicit sub-steps per dose step
        const double dt(dtTotal/nSub);

        // Share of the homogeneous-clustering (SIA) nucleation flux taken by each
        // family, normalized WITHIN its own polarity. The 0-D splits that flux with
        // f_na and f_a -- the very fractions that split the cascade source G_iL into
        // G_iL/G_aiL (reaction_rates.nucleation_rate_iL/aiL) -- so using
        // loopCascadeFractions here reproduces the 0-D split exactly and generalizes
        // to any number of families. Falls back to an equal split if a polarity has
        // no cascade channel at all.
        Eigen::Array<double,1,nF> clusShare(Eigen::Array<double,1,nF>::Zero());
        {
            Eigen::Array<double,1,2> fracTot(Eigen::Array<double,1,2>::Zero());
            Eigen::Array<double,1,2> cntTot(Eigen::Array<double,1,2>::Zero());
            for(int k=0;k<nF;++k)
            {
                const int p(isVacancyFamily(k)?0:1);
                fracTot(p)+=cdp.loopCascadeFractions(k);
                cntTot(p)+=1.0;
            }
            for(int k=0;k<nF;++k)
            {
                const int p(isVacancyFamily(k)?0:1);
                clusShare(k)=(fracTot(p)>0.0)? cdp.loopCascadeFractions(k)/fracTot(p)
                                             : 1.0/cntTot(p);
            }
        }

        for(size_t node=0;node<nNodes;++node)
        {
            // ---- local mobile concentrations ------------------------------
            Eigen::Array<double,1,mSize> cM(Eigen::Array<double,1,mSize>::Zero());
            for(int m=0;m<mSize;++m)
            {
                cM(m)=std::max(mDof(node*mSize+m),0.0);
            }

            // ---- homogeneous clustering: the SIA nucleation flux ------------
            // The mobile ladder stops at 3i, so the reactions i+3i, 2i+2i and
            // 2i+3i push their product OFF it. getR2() therefore debits the
            // reactants and credits nothing -- which is what raises "Warning: Sum
            // of R2 is not zero" at start-up. Those interstitials do not vanish:
            // in the 0-D they are precisely the homogeneous loop-nucleation
            // source, entering nucleation_rate_iL/aiL as R_i_3i + R_2i_2i (loop
            // NUMBER) and nucleation_content_i as 4 R_i_3i + 2 R_2i_2i +
            // 3 R_2i_3i (loop CONTENT). Omitting them here left the <a> loops fed
            // by cascades alone: measured N_a = 0.53x the 0-D at 30 dpa, against a
            // cascade-only nucleation share of 7.04e-13/1.144e-12 = 0.62.
            //
            // For channel (a,b) getR2() writes -K_ab into BOTH R2[a](a,b) and
            // R2[a](b,a), so c^T R2[a] c = -2 K_ab c_a c_b -- but the residual is
            // assembled as R2*(0.5*mobileClusters) (lWF_R2 above; the 0.5 is the
            // usual quadratic-form factor that makes bWF_R2 = R2*c its exact
            // Jacobian). The per-species loss rate is therefore K_ab c_a c_b, and
            // K_ab is exactly the 0-D coefficient:
            //   i +3i : K = 5.0535e7 = omega_i + omega_3i   -> K c_i c_3i = R_i_3i
            //   2i+2i : K = 9.877e2  = 4 omega_2i           -> K c_2i^2   = R_2i_2i
            //   2i+3i : K = 4.938e2  = omega_2i + omega_3i  -> R_2i_3i
            // One loop embryo is born per event, so that same rate is the
            // loop-NUMBER source. The CONTENT source is the number of ATOMS that
            // left the ladder: |m_a|+|m_b| per event for a!=b, and |m_a| per event
            // when a==b (one species supplying both reactants). Taking both from
            // the same K_ab makes the split mass-conserving by construction: the
            // loops gain exactly what the mobile field loses.
            //
            // Units: cM are atom fractions, so clusN is loops per ATOM per time
            // and needs the same /omega as the cascade source to reach loops per
            // b^3; clusC is atoms per atom per time, which IS the volume-fraction
            // convention of the content DOF, so it passes through unconverted.
            // Index 0 = vacancy-type product, 1 = interstitial-type.
            Eigen::Array<double,1,2> clusN(Eigen::Array<double,1,2>::Zero());
            Eigen::Array<double,1,2> clusC(Eigen::Array<double,1,2>::Zero());
            for(const auto& ch : cdp.loopNucChannels)
            {
                const int a(ch.first.first);
                const int c(ch.first.second);
                const double loss(ch.second*cM(a)*cM(c));      // per-species loss rate
                const int p((cdp.msVector(a)<0.0)?0:1);
                clusN(p)+=loss;
                clusC(p)+=(a==c)? std::fabs(cdp.msVector(a))*loss
                                : (std::fabs(cdp.msVector(a))+std::fabs(cdp.msVector(c)))*loss;
            }

            for(int s=0;s<nSub;++s)
            {
                Eigen::Array<double,1,nF> n(Eigen::Array<double,1,nF>::Zero());
                Eigen::Array<double,1,nF> c(Eigen::Array<double,1,nF>::Zero());
                for(int k=0;k<nF;++k)
                {
                    n(k)=std::max(iDof(node*iSize+k),nFloor);
                    c(k)=std::max(iDof(node*iSize+nF+k),cFloor);
                }

                // ---- geometry (Eqs. 28-30): radius and sink strength -------
                const Eigen::Array<double,1,nF> rk(cdp.clusterRadius(c,n));
                // loopSinkScale multiplies the sink strength identically here and in
                // ImmobileSinks, so absorption by the loops and the growth it causes
                // stay consistent -- as in the 0-D, where loop_absorption() and the
                // loop_growth_rate_* methods share one prefactor.
                const Eigen::Array<double,1,nF> Sk(cdp.clusterDensity(c,n)*cdp.loopSinkScale);

                // ---- absorption fluxes, Eqs. (41)-(45) ---------------------
                // phi_v : vacancy channel ; phi_i : interstitial channel weighted
                // by cluster size |m_m| (a 2i cluster carries two interstitials).
                Eigen::Array<double,1,nF> gdot(Eigen::Array<double,1,nF>::Zero());
                Eigen::Array<double,1,nF> absV(Eigen::Array<double,1,nF>::Zero());
                for(int k=0;k<nF;++k)
                {
                    const int row(isVacancyFamily(k)?0:1);
                    double likeFlux(0.0), oppFlux(0.0);
                    for(int m=0;m<mSize;++m)
                    {
                        const double sz(std::fabs(cdp.msVector(m)));
                        const double rate(Dbar(m)*Z(row,m)*Sk(k)*cM(m)*sz);
                        const bool mobileIsVacancy(cdp.msVector(m)<0.0);
                        if(mobileIsVacancy==bool(isVacancyFamily(k)))
                        {
                            likeFlux+=rate;
                        }
                        else
                        {
                            oppFlux+=rate;
                        }
                    }
                    // minimum-stable-size gate on the shrinking channel (Eq. 91)
                    double gate(1.0);
                    if(cdp.r_min(k)>0.0)
                    {
                        const double xg((rk(k)/cdp.r_min(k)-1.0)/0.3);
                        gate=(xg<=0.0)?0.0:((xg>=1.0)?1.0:xg*xg*(3.0-2.0*xg));
                    }
                    gdot(k)=likeFlux-gate*oppFlux;
                    absV(k)=likeFlux;
                }

                // ---- geometric coalescence, Eqs. (97)-(102) ----------------
                // These are first-order LOSS COEFFICIENTS [1/time], not rates:
                // Eq. (97) removes loops in proportion to n and (for the
                // loop-network channel) content in proportion to c. They are
                // handed to the implicit update below for the same reason as the
                // vacancy-loop dissolution: nu_LN*phi_LN*dt_sub reaches ~3.6 at
                // G=1e-7, so an explicit step overshoots, clamps to the floor,
                // and the mean loop size collapses back to n_nuc every step.
                Eigen::Array<double,1,nF> coalN(Eigen::Array<double,1,nF>::Zero());
                Eigen::Array<double,1,nF> coalC(Eigen::Array<double,1,nF>::Zero());
                if(cdp.rhoNetwork>0.0)
                {
                    const double dNet(1.0/std::sqrt(cdp.rhoNetwork));
                    for(int k=0;k<nF;++k)
                    {
                        const double Nvol(n(k));                      // loops per b^3
                        if(Nvol<=0.0) continue;
                        const double dLL(std::pow(1.0/std::max(Nvol,nFloor),1.0/3.0));
                        const double rLL(std::min(rk(k),dLL));
                        const double rLN(std::min(rk(k),dNet));
                        const double phiLL(1.0-std::exp(-cdp.kappaLL*(4.0/3.0)*M_PI*rLL*rLL*rLL*Nvol));
                        const double phiLN(1.0-std::exp(-cdp.kappaLN*M_PI*rLN*rLN*cdp.rhoNetwork));
                        // absorbed-flux climb speed: stays positive at saturation
                        const double vAbs(std::max(absV(k)/std::max(2.0*M_PI*rk(k)*Nvol,double(FLT_EPSILON)),0.0));
                        const double nuLL(cdp.cLL(k)*vAbs*std::pow(Nvol,1.0/3.0));
                        const double nuLN(cdp.cLN(k)*vAbs*std::sqrt(cdp.rhoNetwork));
                        coalN(k)=nuLL*phiLL+nuLN*phiLN;
                        coalC(k)=nuLN*phiLN;          // like-loop channel conserves content
                    }
                }

                // ---- sources and sinks -------------------------------------
                for(int k=0;k<nF;++k)
                {
                    // Cascade-borne loop production, Eqs. (36)/(51).
                    //
                    // UNITS. The immobile DOFs follow MoDELib's own convention,
                    // the one clusterRadius()/clusterDensity()/sigmoid() assume:
                    //   n  = loop number density   [1/b^3]
                    //   c  = defect VOLUME FRACTION [-]   (= the 0-D per-atom
                    //        content numerically, since c = (defects/b^3)*Omega)
                    // so that n_defects/loop = c/(n*Omega), exactly the ratio
                    // those helpers form. G0*eps_k is a per-ATOM production rate,
                    // hence the content source Gk needs no conversion but the
                    // number source does: dividing by Omega turns loops-per-atom
                    // into loops-per-b^3.
                    const double Gk(cdp.G0*cdp.loopCascadeFractions(k));
                    // Cascade-borne loops are born at nNuc defects; clustering-borne
                    // embryos are born carrying only the |m_a|+|m_b| atoms of the
                    // event that made them, so the two sources are NOT divided by
                    // the same number -- exactly as in the 0-D, where the cascade
                    // term is G_iL/n_iL_nuc while the clustering term is the bare
                    // reaction rate. This dilutes the mean loop size, and it should.
                    const int pol(isVacancyFamily(k)?0:1);
                    const double nucRate(Gk/std::max(cdp.nNuc(k),1.0)/cdp.omega
                                         +clusShare(k)*clusN(pol)/cdp.omega);

                    // vacancy-loop dissolution (Eqs. 38/53) and thermal
                    // emission of single vacancies (Eqs. 54-56). Both act only
                    // on the vacancy family; emission removes CONTENT but not
                    // DENSITY, dissolution removes both.
                    // Linear (first-order) loss COEFFICIENTS [1/time], not rates:
                    //   dissolution   n/tau_vL, c/tau_vL          (Eqs. 38/53)
                    //   emission      alpha_v(mBar) * c           (Eqs. 55/56)
                    // These are integrated IMPLICITLY below. tau_vL is short
                    // compared with a dose step (tau_vL ~ 1e17 vs dt ~ 1e19 in
                    // MoDELib units at G=1e-7 dpa/s), so an explicit update
                    // would give n(1-dt/tau) < 0 and collapse the vacancy-loop
                    // family onto the positivity floor. The implicit form is
                    // unconditionally stable and returns the correct balance
                    // n* = Ndot_nuc * tau_vL as dt/tau_vL -> infinity.
                    double lossN(coalN(k)), lossC(coalC(k));
                    double contentSink(0.0);
                    if(isVacancyFamily(k) && cdp.tauVac>0.0)
                    {
                        lossN+=1.0/cdp.tauVac;
                        // Annealing releases each dissolving loop's BIRTH content,
                        // n_nuc defects, NOT its mean content -- ZrMicro
                        // annealing_content_vL(): tau_vL describes dissolution of
                        // still-small EMBRYO loops, while loops that survive and
                        // grow are stable. The release is therefore proportional to
                        // the DENSITY, n_nuc*n/tau_vL (times Omega, n being per
                        // b^3 and c a volume fraction).
                        //
                        // Using c/tau_vL instead caps the mean loop size at ~n_nuc
                        // (measured: m_c = 487 against n_nuc = 400, versus 1.7e5 in
                        // the 0-D). Self-check: at number saturation n* =
                        // Ndot_nuc*tau_vL, so the release equals n_nuc*Ndot_nuc*Omega
                        // = Gk and exactly cancels the cascade content seed, leaving
                        // net content growth flux-driven -- as the 0-D docstring says.
                        contentSink+=cdp.nNuc(k)*n(k)*cdp.omega/cdp.tauVac;
                        // Thermal emission of single vacancies, Eqs. (54)-(56).
                        // Per loop, alpha_v = 2*pi*r*D_v/Omega*exp(-Eb/kB T)
                        // (the convention of getR1()); the content lost per unit
                        // volume is alpha_v*N*Omega = S_k*D_v*exp(-Eb/kB T), the
                        // Omegas cancelling. That rate does NOT scale with c, so
                        // to keep the implicit update below it is divided by c to
                        // form an equivalent first-order coefficient.
                        if(n(k)>nFloor)
                        {
                            const double emitC(Sk(k)*Dbar(0)*std::exp(-cdp.Eb(0)/(cdp.kB*cdp.T)));
                            lossC+=emitC/std::max(c(k),cFloor);
                        }
                    }

                    // implicit for the linear losses, explicit for the sources
                    const double nNew((n(k)+dt*nucRate)/(1.0+dt*lossN));
                    const double cNew((c(k)+dt*(gdot(k)+Gk+clusShare(k)*clusC(pol)-contentSink))/(1.0+dt*lossC));
                    iDof(node*iSize+k)     = std::max(nNew,nFloor);
                    iDof(node*iSize+nF+k)  = std::max(cNew,cFloor);
                }
            }
        }
        immobileClusters=iDof;
    }

    template<int dim>
    void ClusterDynamicsFEM<dim>::solve()
    {/*! The two-time-scale split, in the order the split requires: the FAST
      *  step first -- solveMobileClusters() has no time derivative at all, it
      *  is the quasi-steady solve of the mobile field for the immobile state
      *  currently held -- then the SLOW step, which advances the immobile
      *  population over dt with that mobile field frozen.
      *
      *  With useImmobileSolver=0 only the fast step runs, and an external
      *  driver supplies the slow one.
      */
        solveMobileClusters();
        if(useImmobileSolver)
        {
            solveImmobileClusters();
        }
        else
        {
            std::cout<<", immobile solver SKIPPED (useImmobileSolver=0)"<<std::flush;
        }
    }

    template<int dim>
    void ClusterDynamicsFEM<dim>::initializeSolver()
    {
        std::cout<<" decomposing"<<std::flush;
        std::array<bool,mSize> allComps;
        for(int k=0;k<mSize;++k)
        {
            allComps[k]=true;
        }
        
        mobileClusters.addDirichletCondition(nodeListInternalExternal,Fix(),allComps); // apply to the four components of c
        mobileClustersIncrement.addDirichletCondition(nodeListInternalExternal,Fix(),allComps); // apply to the four components of c
        mSolver.compute(mBWF); // call this after assigning the BCs
        solverInitialized=true;
    }

    template<int dim>
    void ClusterDynamicsFEM<dim>::writeNodePositions() const
    {/*! Dump the finite-element node coordinates, in b, one row per node and in
      *  the SAME order as the cd columns of evl_*.txt.
      *
      *  Without this the fields cannot be plotted in space: the CD trial functions
      *  live on second-order elements (24115 nodes for a 3392-node, 15702-tet
      *  mesh), so the rows of cdMatrix do NOT correspond to the vertices in the
      *  .msh file, and the interior ordering is not reproducible from outside.
      *  Written once per run, since the mesh does not move.
      */
        std::ofstream nodeFile("evl/cdNodes.txt");
        nodeFile<<std::setprecision(15)<<std::scientific;
        for(const auto& node : mobileClusters.fe().nodes())
        {
            nodeFile<<node.P0.transpose()<<"\n";
        }
        std::cout<<"wrote evl/cdNodes.txt ("<<mobileClusters.fe().nodes().size()<<" nodes)"<<std::endl;
    }

    template<int dim>
    void ClusterDynamicsFEM<dim>::initializeConfiguration(const DDconfigIO<dim>& configIO,const std::ofstream& f_file,const std::ofstream& F_labels)
    {
        writeNodePositions();

        if(size_t(configIO.cdMatrix().size())==mobileClusters.gSize()+immobileClusters.gSize())
        {
            const size_t nNodes(mobileClusters.fe().nodes().size());
            mobileClusters=configIO.cdMatrix().block(0,0,nNodes,mSize).transpose().reshaped(mobileClusters.gSize(),1);
            immobileClusters=configIO.cdMatrix().block(0,mSize,nNodes,iSize).transpose().reshaped(immobileClusters.gSize(),1);
        }
        else
        {
            if(configIO.cdMatrix().size())
            {
                throw std::runtime_error("ClusterDynamics: TrialFunctions initializatoin size mismatch");
            }
        }
        
        
        
    }

    template<int dim>
    typename ClusterDynamicsFEM<dim>::VectorDim ClusterDynamicsFEM<dim>::inelasticDisplacementRate(const VectorDim& x, const NodeType* const node, const ElementType* const ele,const SimplexDim* const guess) const
    {
        Eigen::Matrix<double,dim*mSize,1> speciesFlux(Eigen::Matrix<double,dim*mSize,1>::Zero());
        if(node)
        {
            speciesFlux=eval(this->mobileFlux)(*node);
        }
        else
        {
            if(ele)
            {
                speciesFlux=eval(this->mobileFlux)(*ele,ele->simplex.pos2bary(x));
            }
            else
            {
                speciesFlux=eval(this->mobileFlux)(x,guess);
            }
        }
        Eigen::Matrix<double,dim,1> netFlux(Eigen::Matrix<double,dim,1>::Zero());
        for(int i=0; i<mSize; i++)
        {
            const int mSgn(this->cdp.msVector(i)/std::abs(this->cdp.msVector(i)));
            netFlux += speciesFlux.template block<dim,1>(i*dim,0)*mSgn;
        }
        return netFlux;
    }

    template struct ClusterDynamicsFEM<3>;

}
#endif



//template<int dim>
//void ClusterDynamicsFEM<dim>::applyBoundaryConditions()
//{
//
//    const auto& nodesInternalExternal(mobileClusters.fe().nodeList(nodeListInternalExternal));
//
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
//    for(size_t k=0;k<nodesInternalExternal.size();++k)
//    {
//        const auto& node(nodesInternalExternal[k]);
//        const auto outNormal(node->outNormal()); // used to compute traction
//        const MatrixDim sigma(microstructures.stress(node->P0,node,nullptr,nullptr));
//        const double normalTraction(outNormal.dot(sigma*outNormal));
//        const auto bndConcentration(this->cdp.boundaryMobileConcentration(sigma.trace(),normalTraction));
//
//        VectorMSize otherConcentration(VectorMSize::Zero());
//        for(const auto& microstructure : this->microstructures)
//        {
//            if(microstructure.get()!=static_cast<const MicrostructureBase<dim>* const>(this))
//            {// not the ClusterDynamics physics
//                otherConcentration += microstructure->mobileConcentration(node->P0,node,nullptr,nullptr);
//            }
//        }
//        for(int k=0;k<mSize;++k)
//        {
//            mobileClusters.dirichletConditions().at(mSize*node->gID+k) = bndConcentration(k) - otherConcentration(k);
//        }
//    }
//}
