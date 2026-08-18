/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_GalerkinClimbSolver_cpp_
#define model_GalerkinClimbSolver_cpp_

#include <deque>
#include <limits>       // numeric_limits, for the pair-cutoff distance test
#include <algorithm>    // min
#include <vector>
#include <Eigen/SparseLU>   // the real Eq. (50) solve, in place of row lumping


#include <ClusterDynamicsParameters.h>
#include <GalerkinClimbSolver.h>
#include <TerminalColors.h>
#include <EqualIteratorRange.h>

namespace model
{

    template <typename DislocationNetworkType>
    GalerkinClimbSolver<DislocationNetworkType>::GalerkinClimbSolver(const DislocationNetworkType& DN_in,const ClusterDynamics<dim>* const CD_in) :
    /* init */ DislocationClimbSolverBase<DislocationNetworkType>(DN_in,CD_in)
    {
        std::cout<<greenBoldColor<<"Creating GalerkinClimbSolver"<<defaultColor<<std::endl;
    }

    template <typename DislocationNetworkType>
    typename GalerkinClimbSolver<DislocationNetworkType>::ForceVectorMatrixType GalerkinClimbSolver<DislocationNetworkType>::clusterForceKernel(const int& k,const NetworkLinkType& fieldSegment) const
    {
        const Eigen::Matrix<double,1,mSize> deltaC(fieldSegment.quadraturePoint(k).cDD-fieldSegment.quadraturePoint(k).cCD);
        const double u(fieldSegment.quadraturePoint(k).abscissa);
        return  (ForceVectorMatrixType()<<(1.0-u)*deltaC,u*deltaC).finished();
    }

    template <typename DislocationNetworkType>
    typename GalerkinClimbSolver<DislocationNetworkType>::ForceVectorMatrixType GalerkinClimbSolver<DislocationNetworkType>::clusterForceVector(const NetworkLinkType& fieldSegment) const
    {
        ForceVectorMatrixType F(ForceVectorMatrixType::Zero());
        const auto bxt(fieldSegment.burgers().cross(fieldSegment.chord()));
        NetworkLinkType::QuadratureDynamicType::integrate(fieldSegment.quadraturePoints().size(),this,F,&GalerkinClimbSolver<DislocationNetworkType>::clusterForceKernel,fieldSegment);
        F.row(0)*=bxt.dot(fieldSegment.source->climbDirection());
        F.row(1)*=bxt.dot(fieldSegment.sink->climbDirection());
        return F;
    }

    template <typename DislocationNetworkType>
    typename GalerkinClimbSolver<DislocationNetworkType>::StiffnessMatrixType GalerkinClimbSolver<DislocationNetworkType>::clusterStiffnessKernel(const int& k,const NetworkLinkType& fieldSegment,const NetworkLinkType& sourceSegment) const
    {
        StiffnessMatrixType temp(StiffnessMatrixType::Zero());
        if(sourceSegment.grains().size() == 1)
        {
            const auto bxt(fieldSegment.burgers().cross(fieldSegment.chord()));
            const double u(fieldSegment.quadraturePoint(k).abscissa);
            const Eigen::Matrix<double,2,1> m(( Eigen::Matrix<double,2,1>()<<(1.0-u)*bxt.dot(fieldSegment.source->climbDirection()),
                                               /*                               */ u*bxt.dot(fieldSegment.sink->climbDirection())).finished());
            const auto sourceConcentratoinMatrices(sourceSegment.concentrationMatrices(fieldSegment.quadraturePoint(k).r,this->CD->cdp));
            for(int kc=0; kc < mSize; kc++)
            {
                temp.template block<2,2>(kc*2,0)+=m*sourceConcentratoinMatrices.row(kc);
            }
        }
        return temp;
    }

    template <typename DislocationNetworkType>
    typename GalerkinClimbSolver<DislocationNetworkType>::StiffnessMatrixType GalerkinClimbSolver<DislocationNetworkType>::clusterStiffnessMatrix(const NetworkLinkType& fieldSegment,const NetworkLinkType& sourceSegment) const
    {
        StiffnessMatrixType K(StiffnessMatrixType::Zero());
        NetworkLinkType::QuadratureDynamicType::integrate(fieldSegment.quadraturePoints().size(),this,K,&GalerkinClimbSolver<DislocationNetworkType>::clusterStiffnessKernel,fieldSegment,sourceSegment);
        return K;
    }

    template <typename DislocationNetworkType>
    Eigen::VectorXd GalerkinClimbSolver<DislocationNetworkType>::getNodeVelocitiesPipe() const
    {
        Eigen::VectorXd nodeVelocities(Eigen::VectorXd::Zero(this->DN.networkNodes().size()*dim));
        return nodeVelocities;
    }

    template <typename DislocationNetworkType>
    void GalerkinClimbSolver<DislocationNetworkType>::computeClimbScalarVelocities()
    {
        std::cout<<", climbSolver "<<std::flush;
        computeClimbScalarVelocitiesBulk();
    }

    template <typename DislocationNetworkType>
    void GalerkinClimbSolver<DislocationNetworkType>::computeClimbScalarVelocitiesBulk()
    {
        const size_t nNodes(this->DN.networkNodes().size());
        std::vector<TripletContainerType> lhsT(mSize);
        std::vector<Eigen::VectorXd> Fc(mSize,Eigen::VectorXd::Zero(nNodes));
        std::vector<Eigen::VectorXd> KKc(mSize,Eigen::VectorXd::Zero(nNodes));
        // The lumped vector KKc is assembled either way: it is cheap, it is the
        // legacy path under climbLumpedSolver=1, and it is the fallback if the
        // sparse factorization fails.
        const bool lumped(this->CD->cdp.climbLumpedSolver!=0);

#ifdef _OPENMP
        const size_t nThreads(omp_get_max_threads());
        std::vector<std::vector<TripletContainerType>> lhsTV(nThreads,lhsT);
        std::vector<std::vector<Eigen::VectorXd>> KKcT(nThreads,KKc);
        std::vector<std::vector<Eigen::VectorXd>> FcT(nThreads,Fc);
        const EqualIteratorRange<typename DislocationNetworkType::NetworkLinkContainerType::const_iterator> eir(this->DN.networkLinks().begin(),this->DN.networkLinks().end(),nThreads);
#pragma omp parallel for
        for(size_t thread=0;thread<eir.size();++thread)
        {
            auto& Fc_ref(FcT[thread]);
            auto& KKc_ref(KKcT[thread]);
            auto& lhs_ref(lhsTV[thread]);
            for(auto fieldLinkIter=eir[thread].first;fieldLinkIter!=eir[thread].second;++fieldLinkIter)
//            for(const auto& fieldLink : this->DN.networkLinks())
            {// sum line-integral part of displacement field per segment
                const auto& fieldLink(*fieldLinkIter);
#else
                auto& Fc_ref(Fc);
                auto& KKc_ref(KKc);
                auto& lhs_ref(lhsT);
                for(const auto& fieldLink : this->DN.networkLinks())
                {
#endif
////#pragma omp parallel for
//        for(size_t thread=0;thread< eir.size();++thread)
//        {
//            for(auto fieldLinkIter=eir[thread].first;fieldLinkIter!=eir[thread].second;++fieldLinkIter)
////            for(const auto& fieldLink : this->DN.networkLinks())
//            {// sum line-integral part of displacement field per segment
//                const auto& fieldLink(*fieldLinkIter);
                if(   !fieldLink.second.lock()->hasZeroBurgers()
                   && fieldLink.second.lock()->isSessile()
                   && !fieldLink.second.lock()->isBoundarySegment()
                   && !fieldLink.second.lock()->isGrainBoundarySegment()
                   &&  fieldLink.second.lock()->chordLength()>FLT_EPSILON
                   )
                {
                    const size_t i0(fieldLink.second.lock()->source->gID());
                    const size_t i1(fieldLink.second.lock()->  sink->gID());
                    
                    const ForceVectorMatrixType fc(clusterForceVector(*fieldLink.second.lock()));
                    for(int kc=0; kc<mSize; ++kc)
                    {
//#ifdef _OPENMP
//#pragma omp critical
//#endif
//                        {
                            Fc_ref[kc](i0)+=fc(0,kc);
                            Fc_ref[kc](i1)+=fc(1,kc);
//                        }
                    }
                    
                    // Midpoint and half-length of the FIELD segment, for the
                    // screened-cutoff test below. Hoisted out of the source
                    // loop because it does not depend on the source.
                    const double Rc(this->CD->cdp.climbNeighborCutoff);
                    const VectorDim fieldMid(fieldLink.second.lock()->source->get_P()
                                             +0.5*fieldLink.second.lock()->chord());
                    const double fieldHalf(0.5*fieldLink.second.lock()->chordLength());

                    for(const auto& sourceLink : this->DN.networkLinks())
                    {// sum line-integral part of displacement field per segment
                        if(   !sourceLink.second.lock()->hasZeroBurgers()
                           && !sourceLink.second.lock()->isBoundarySegment()
                           && !sourceLink.second.lock()->isGrainBoundarySegment()
                           &&  sourceLink.second.lock()->chordLength()>FLT_EPSILON
                           )
                        {
                            /* Screened cutoff on the pair assembly.
                             *
                             * The kernel is exp(-k r)/r, not bare 1/r, so this
                             * sum converges and truncating it is legitimate --
                             * see ClusterDynamicsParameters::climbNeighborCutoff.
                             *
                             * The test is deliberately CONSERVATIVE: it compares
                             * the distance between segment MIDPOINTS against
                             * Rc plus BOTH half-chords, so a pair is dropped only
                             * when no point of one segment can lie within Rc of
                             * any point of the other. It can over-include, never
                             * wrongly exclude.
                             *
                             * The minimum is taken over periodicShifts, because
                             * concentrationMatrices() sums over the images and a
                             * pair that is far in the primary cell may be near
                             * in one of them.
                             *
                             * The SELF term is never truncated: its midpoint
                             * distance is zero, so it passes any Rc >= 0.
                             */
                            if(Rc>0.0)
                            {
                                const VectorDim sourceMid(sourceLink.second.lock()->source->get_P()
                                                          +0.5*sourceLink.second.lock()->chord());
                                const double reach(Rc+fieldHalf+0.5*sourceLink.second.lock()->chordLength());
                                double d2min(std::numeric_limits<double>::max());
                                for(const auto& shift : this->DN.ddBase.periodicShifts)
                                {
                                    d2min=std::min(d2min,(sourceMid+shift-fieldMid).squaredNorm());
                                }
                                if(d2min>reach*reach)
                                {
                                    continue;
                                }
                            }
                            const size_t j0(sourceLink.second.lock()->source->gID());
                            const size_t j1(sourceLink.second.lock()->  sink->gID());
                            const StiffnessMatrixType kcc(clusterStiffnessMatrix(*fieldLink.second.lock(),*sourceLink.second.lock()));
                            
                            for(int kc=0; kc<mSize; ++kc)
                            {
                                const Eigen::Matrix<double,2,2> kccs(kcc.template block<2,2>(2*kc,0));

                                        // Legacy lumped vector: a symmetrized
                                        // row+column sum, kept for
                                        // climbLumpedSolver=1 and as the
                                        // fallback if the factorization fails.
                                        KKc_ref[kc](i0)+=0.5*kccs(0,0)+0.5*kccs(0,1);
                                        KKc_ref[kc](j0)+=0.5*kccs(0,0)+0.5*kccs(1,0);
                                        KKc_ref[kc](i1)+=0.5*kccs(1,0)+0.5*kccs(1,1);
                                        KKc_ref[kc](j1)+=0.5*kccs(0,1)+0.5*kccs(1,1);

                                        /* THE ACTUAL OPERATOR of Eq. (50).
                                         *
                                         * `clusterStiffnessKernel` forms
                                         * kccs = m * G_row, an outer product of
                                         *   m     = [(1-u) bxt.n_i0, u bxt.n_i1]
                                         * (the field segment's shape functions,
                                         * i.e. the EQUATION rows i0,i1) with
                                         *   G_row = concentrationMatrices.row(kc)
                                         * (the concentration produced at the
                                         * field point per unit climb velocity of
                                         * the source segment's nodes, i.e. the
                                         * UNKNOWN columns j0,j1).
                                         *
                                         * So the entry (row, col) mapping is
                                         * exactly kccs(a,b) -> K(i_a, j_b), and
                                         * nothing may be collapsed onto the
                                         * diagonal. F is assembled on the same
                                         * rows i0,i1 by clusterForceVector. */
                                        lhs_ref[kc].emplace_back(i0,j0,kccs(0,0));
                                        lhs_ref[kc].emplace_back(i0,j1,kccs(0,1));
                                        lhs_ref[kc].emplace_back(i1,j0,kccs(1,0));
                                        lhs_ref[kc].emplace_back(i1,j1,kccs(1,1));
//                                }
//                                else
//                                {
//                                    const bool forceSym(true);// at the moment we need to symmtrize, since numerical intgration on fieldLink is not equivalent to analytical integration on sourceLink
//                                    if(forceSym)
//                                    {
//#ifdef _OPENMP
//#pragma omp critical
//#endif
//                                        {
//                                            lhsT[kc].emplace_back(i0,j0,0.5*kccs(0,0));
//                                            lhsT[kc].emplace_back(i0,j1,0.5*kccs(0,1));
//                                            lhsT[kc].emplace_back(i1,j0,0.5*kccs(1,0));
//                                            lhsT[kc].emplace_back(i1,j1,0.5*kccs(1,1));
//                                            
//                                            lhsT[kc].emplace_back(j0,i0,0.5*kccs(0,0));
//                                            lhsT[kc].emplace_back(j1,i0,0.5*kccs(0,1));
//                                            lhsT[kc].emplace_back(j0,i1,0.5*kccs(1,0));
//                                            lhsT[kc].emplace_back(j1,i1,0.5*kccs(1,1));
//                                        }
//                                    }
//                                    else
//                                    {
//#ifdef _OPENMP
//#pragma omp critical
//#endif
//                                        {
//                                            lhsT[kc].emplace_back(i0,j0,kccs(0,0));
//                                            lhsT[kc].emplace_back(i0,j1,kccs(0,1));
//                                            lhsT[kc].emplace_back(i1,j0,kccs(1,0));
//                                            lhsT[kc].emplace_back(i1,j1,kccs(1,1));
//                                        }
//                                    }
//                                }
                            }
                        }
                    }
                }
            }
#ifdef _OPENMP
        }
            for(size_t thread=0;thread< eir.size();++thread)
            {// recombine contributions of different threads
                for(int kc=0; kc<mSize; ++kc)
                {
                    Fc[kc]+=FcT[thread][kc];
                    KKc[kc]+=KKcT[thread][kc];
                    lhsT[kc].insert(lhsT[kc].end(), lhsTV[thread][kc].begin(), lhsTV[thread][kc].end());
                }
            }
#endif
        
        std::vector<Eigen::Array<double,1,mSize>> nodeV(nNodes,Eigen::Array<double,1,mSize>::Zero());

        for(int kc=0; kc<mSize; ++kc)
        {
            // The lumped answer: computed always, used directly when
            // climbLumpedSolver=1 and as the fallback below.
            Eigen::VectorXd vLumped(Eigen::VectorXd::Zero(nNodes));
            for(size_t n=0; n<nNodes; ++n)
            {
                if(std::fabs(KKc[kc](n))>FLT_EPSILON)
                {
                    vLumped(n)=Fc[kc](n)/KKc[kc](n);
                }
            }

            if(lumped)
            {
                for(size_t n=0; n<nNodes; ++n)
                {
                    nodeV[n](kc)=vLumped(n);
                }
                continue;
            }

            /* SOLVE Eq. (50) AS WRITTEN: K w = F.
             *
             * A node that carries no climb equation -- a purely glide node, a
             * boundary or grain-boundary node, a node on a zero-Burgers or
             * non-sessile segment -- contributes no triplet and no force, so its
             * row and column are empty. Left alone that makes K singular. Those
             * rows are given a unit diagonal against a zero force, which pins
             * w=0 there: the same answer the lumped path gave them (its
             * |KKc|>eps test skipped them), and it keeps the participating block
             * untouched. */
            SparseMatrixType Kcc(nNodes,nNodes);
            Kcc.setFromTriplets(lhsT[kc].begin(),lhsT[kc].end());
            Kcc.makeCompressed();

            std::vector<bool> active(nNodes,false);
            for(int c=0; c<Kcc.outerSize(); ++c)
            {
                for(typename SparseMatrixType::InnerIterator it(Kcc,c); it; ++it)
                {
                    if(std::fabs(it.value())>0.0)
                    {
                        active[it.row()]=true;
                    }
                }
            }
            TripletContainerType pinned;
            for(size_t n=0; n<nNodes; ++n)
            {
                if(!active[n])
                {
                    pinned.emplace_back(n,n,1.0);
                }
            }
            if(pinned.size())
            {
                TripletContainerType all(lhsT[kc]);
                all.insert(all.end(),pinned.begin(),pinned.end());
                Kcc.setZero();
                Kcc.setFromTriplets(all.begin(),all.end());
                Kcc.makeCompressed();
            }

            Eigen::SparseLU<SparseMatrixType,Eigen::COLAMDOrdering<int>> solver;
            solver.analyzePattern(Kcc);
            solver.factorize(Kcc);
            bool ok(solver.info()==Eigen::Success);
            Eigen::VectorXd w;
            if(ok)
            {
                w=solver.solve(Fc[kc]);
                ok=(solver.info()==Eigen::Success) && w.allFinite();
            }
            if(!ok)
            {
                /* NEVER fall through to a wrong answer in silence. A failed
                 * factorization means the pair assembly is degenerate -- most
                 * likely a duplicated node position -- and the lumped value is
                 * the only thing left. Say so loudly, because a run that
                 * silently reverts to the lumped path is exactly the situation
                 * this change exists to end. */
                std::cout<<redBoldColor<<"GalerkinClimbSolver: SparseLU failed for"
                         <<" mobile species "<<kc<<"; falling back to the LUMPED"
                         <<" velocity for this species. The climb condition is"
                         <<" NOT enforced."<<defaultColor<<std::endl;
                w=vLumped;
            }
            for(size_t n=0; n<nNodes; ++n)
            {
                nodeV[n](kc)=w(n);
            }
        }

        this->scalarVelocities().resize(this->DN.networkNodes().size(),Eigen::Array<double,1,mSize>::Zero());
        for(size_t k=0; k<this->DN.networkNodes().size();++k)
        {
            this->scalarVelocities()[k]=nodeV[k];
        }
    }

    template <typename DislocationNetworkType>
    Eigen::VectorXd GalerkinClimbSolver<DislocationNetworkType>::getNodeVelocitiesBulk() const
    {
        Eigen::VectorXd nodeVelocities(Eigen::VectorXd::Zero(this->DN.networkNodes().size()*dim));
        size_t k=0;
        for(auto& node: this->DN.networkNodes())
        {
            const double vScalarTotal(-1.0*(this->scalarVelocities()[k]*this->CD->cdp.msVector/this->CD->cdp.msVector.abs()).matrix().sum());
            nodeVelocities.template segment<dim>(k*dim)= vScalarTotal*node.second.lock()->climbDirection();
            k++;
        }
        return nodeVelocities;
    }

    template <typename DislocationNetworkType>
    Eigen::VectorXd GalerkinClimbSolver<DislocationNetworkType>::getNodeVelocities() const
    {
        return getNodeVelocitiesBulk()+getNodeVelocitiesPipe();
    }

    template class GalerkinClimbSolver<DislocationNetwork<3,0>>;

}
#endif
