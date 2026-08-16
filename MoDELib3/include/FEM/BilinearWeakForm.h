/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_BilinearWeakForm_H_
#define model_BilinearWeakForm_H_

#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <chrono>
#include <vector>

//#ifdef _OPENMP
//#include <omp.h>
//#include <Model/Threads/EqualIteratorRange.h>
//#endif

#include <Eigen/Sparse>

#include <TrialBase.h>
#include <LinearWeakForm.h>
#include <TerminalColors.h>
#include <BilinearForm.h>

#include <BilinearWeakExpression.h>
#include <BilinearWeakSum.h>
#include <WeakProblem.h>



namespace model
{
    
    /**************************************************************************/
    /**************************************************************************/
    template <typename _BilinearFormType,typename _IntegrationDomainType>
    struct BilinearWeakForm : public BilinearWeakExpression<BilinearWeakForm<_BilinearFormType,_IntegrationDomainType> >
    {
        
        typedef _BilinearFormType BilinearFormType;
        typedef _IntegrationDomainType IntegrationDomainType;
        typedef BilinearWeakForm<BilinearFormType,IntegrationDomainType> BilinearWeakFormType;
        typedef typename BilinearFormType::TrialFunctionType TrialFunctionType;
        typedef typename BilinearFormType::TestExpressionType TestExpressionType;
        typedef typename BilinearFormType::TrialExpressionType TrialExpressionType;
        typedef typename TrialFunctionType::ElementType ElementType;
        typedef typename IntegrationDomainType::QuadratureType QuadratureType;
        typedef typename QuadratureType::VectorDim AbscissaType;
        
        
        constexpr static int dim=TypeTraits<TrialFunctionType>::dim;
        constexpr static int nodesPerElement=TypeTraits<TrialFunctionType>::nodesPerElement;
        constexpr static int dofPerNode=TypeTraits<TrialFunctionType>::dofPerNode;
        constexpr static int dofPerElement=TypeTraits<TrialFunctionType>::dofPerElement;
        
        
        typedef Eigen::Matrix<double,dofPerElement,dofPerElement> ElementMatrixType;
        
//        const BilinearFormType& bilinearForm;
        ExpressionRef<BilinearFormType> bilinearForm;

        const IntegrationDomainType& domain;
//        const TestExpressionType& testExpr;
//        const TrialExpressionType& trialExpr;
////        const TrialFunctionType& trial;
//        const size_t& gSize;
        
        
        
        /**********************************************************************/
        BilinearWeakForm(const BilinearFormType& bf,
                         const IntegrationDomainType& dom) :
        /* init list */ bilinearForm(bf), // cast testE to its base T2 type
        /* init list */ domain(dom) // cast trialE to its derived T1 type
//        /* init list */ testExpr(bf.testExpr),
//        /* init list */ trialExpr(bf.trialExpr),
////        /* init list */ trial(trialExpr.trial),
//        /* init list */ gSize(bilinearForm.gSize)
        {
//            std::cout<<greenColor<<"Creating BilinearWeakForm: gSize="<<gSize<<defaultColor<<std::endl;
        }
        
        /**********************************************************************/
        BilinearWeakForm(BilinearFormType&& bf,
                         const IntegrationDomainType& dom) :
        /* init list */ bilinearForm(std::move(bf)), // cast testE to its base T2 type
        /* init list */ domain(dom) // cast trialE to its derived T1 type
        //        /* init list */ testExpr(bf.testExpr),
        //        /* init list */ trialExpr(bf.trialExpr),
        ////        /* init list */ trial(trialExpr.trial),
        //        /* init list */ gSize(bilinearForm.gSize)
        {
//            std::cout<<greenColor<<"Creating BilinearWeakForm: gSize="<<gSize<<defaultColor<<std::endl;
        }
        
        /**********************************************************************/
        std::vector<Eigen::Triplet<double> >  globalTriplets() const
        {
            /*! Element assembly, in parallel over EXPLICIT contiguous chunks.
             *
             *  Each element's ke is computed from const state into a local
             *  matrix and appended to a per-chunk buffer, so there is nothing
             *  shared to race on. This loop is the dominant cost of the whole
             *  coupled march: the fast step is ~91% of a march, compute() is
             *  ~87% of the fast step, and this is what compute() spends it on.
             *
             *  The chunks are partitioned by hand rather than left to an OpenMP
             *  schedule, and concatenated in chunk order, so the triplet
             *  sequence is EXACTLY the serial one. That matters: setFromTriplets
             *  sums duplicates, so a different order would re-associate those
             *  sums and the result would agree only to rounding. Reproducing
             *  the order keeps the assembled matrix bit-identical, which is
             *  what makes this change verifiable rather than merely plausible.
             */
            const size_t nEle(domain.size());
#ifdef _OPENMP
            const size_t nChunk(std::min<size_t>(size_t(std::max(1,omp_get_max_threads())),
                                                 std::max<size_t>(nEle,1)));
#else
            const size_t nChunk(1);
#endif
            std::vector<std::vector<Eigen::Triplet<double> > > chunks(nChunk);
            const size_t perChunk(dofPerElement*dofPerElement*(nEle/nChunk+1));
            for(auto& c : chunks)
            {
                c.reserve(perChunk);
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(static,1)
#endif
            for (long long c=0;c<(long long)nChunk;++c)
            {
                const size_t kBeg((size_t(c)*nEle)/nChunk);
                const size_t kEnd(((size_t(c)+1)*nEle)/nChunk);
                std::vector<Eigen::Triplet<double> >& glbtrip(chunks[size_t(c)]);
            for (size_t k=kBeg;k<kEnd;++k)
            {
                const ElementType& ele(domain.element(k));
                ElementMatrixType ke(ElementMatrixType::Zero());
                QuadratureType::integrate(this,ke,&BilinearWeakFormType::elementAssemblyKernel<AbscissaType>,ele);
                
                for (int i=0;i<dofPerElement;++i) //Assemble ke into global array of triplets
                {
                    for (int j=0;j<dofPerElement;++j)
                    {
                        if(ke(i,j)!=0.0)
                        {
                            const size_t  nodeID_I(i/dofPerNode);
                            const size_t nodeDof_I(i%dofPerNode);
                            const size_t gI= ele.node(nodeID_I).gID*dofPerNode+nodeDof_I;
                            
                            const size_t  nodeID_J(j/dofPerNode);
                            const size_t nodeDof_J(j%dofPerNode);
                            const size_t gJ=ele.node(nodeID_J).gID*dofPerNode+nodeDof_J;
                            
                            glbtrip.emplace_back(gI,gJ,ke(i,j));
                        }
                    }
                }
            }
            }
            if(nChunk==1)
            {
                return chunks[0];
            }
            size_t tot(0);
            for(const auto& c : chunks)
            {
                tot+=c.size();
            }
            std::vector<Eigen::Triplet<double> > glbtrip;
            glbtrip.reserve(tot);
            for(const auto& c : chunks)
            {
                glbtrip.insert(glbtrip.end(),c.begin(),c.end());
            }
            return glbtrip;
        }
        
        /**********************************************************************/
        template <typename AbscissaType>
        ElementMatrixType elementAssemblyKernel(const AbscissaType& a, const ElementType& ele) const
        {
            const Eigen::Matrix<double,dim+1,1> bary(BarycentricTraits<dim>::x2l(a));
     //       return testExpr.sfm(ele,bary).transpose()*trialExpr.sfm(ele,bary)*ele.absJ(bary);
            return bilinearForm()(ele,bary)*ele.absJ(bary);
        }
        
//        /**********************************************************************/
//        template< typename T3>
//        WeakProblem<BilinearWeakFormType,T3> operator=(const LinearWeakExpression<T3>& lwf) const
//        {
//            return WeakProblem<BilinearWeakFormType,T3>(*this,lwf);
//        }
        
    };
    
    
    /**************************************************************************/
    /**************************************************************************/
    // Operator *
    template <typename T1, typename T2, typename FiniteElementType, int qOrder, int dimMinusDomainDim,
    /*     */ template <short unsigned int, size_t> class QuadratureRule>
    BilinearWeakForm<BilinearForm<T1,T2>,IntegrationDomain<FiniteElementType,dimMinusDomainDim,qOrder,QuadratureRule> > operator*(const BilinearForm<T1,T2>& biForm,
                                                                                                                                  const IntegrationDomain<FiniteElementType,dimMinusDomainDim,qOrder,QuadratureRule>& domain)
    {
        return BilinearWeakForm<BilinearForm<T1,T2>,IntegrationDomain<FiniteElementType,dimMinusDomainDim,qOrder,QuadratureRule> >(biForm,domain);
    }
    
    template <typename T1, typename T2, typename FiniteElementType, int qOrder, int dimMinusDomainDim,
    /*     */ template <short unsigned int, size_t> class QuadratureRule>
    BilinearWeakForm<BilinearForm<T1,T2>,IntegrationDomain<FiniteElementType,dimMinusDomainDim,qOrder,QuadratureRule> > operator*(BilinearForm<T1,T2>&& biForm,
                                                                                                                                  const IntegrationDomain<FiniteElementType,dimMinusDomainDim,qOrder,QuadratureRule>& domain)
    {
        return BilinearWeakForm<BilinearForm<T1,T2>,IntegrationDomain<FiniteElementType,dimMinusDomainDim,qOrder,QuadratureRule> >(std::move(biForm),domain);
    }
    
}	// close namespace
#endif


