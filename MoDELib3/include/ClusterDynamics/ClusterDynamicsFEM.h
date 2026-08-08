/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_ClusterDynamicsFEM_H_
#define model_ClusterDynamicsFEM_H_

#ifdef MODELIB_CHOLMOD // SuiteSparse Cholmod module
#include <Eigen/CholmodSupport>
#endif

#ifdef MODELIB_UMFPACK // SuiteSparse UMFPACK module
#include <Eigen/UmfPackSupport>
#endif

#include <ClusterDynamicsParameters.h>
#include <DislocationDynamicsBase.h>
#include <EvalFunction.h>
#include <FixedDirichletSolver.h>
#include <DDconfigIO.h>
#include <MicrostructureBase.h>
#include <SecondOrderReaction.h>
#include <MicrostructureContainer.h>

namespace model
{

    template <int dim>
    struct FluxMatrix : public EvalFunction<FluxMatrix<dim>>
    {
        typedef typename DislocationDynamicsBase<dim>::ElementType ElementType;
        typedef Eigen::Matrix<double,dim+1,1> BaryType;
        constexpr static int mSize=ClusterDynamicsParameters<dim>::mSize;
        constexpr static int rows=dim*mSize;
        constexpr static int cols=rows;
        typedef Eigen::Matrix<double,rows,cols> MatrixType;

        const ClusterDynamicsParameters<dim>& cdp;
        
        FluxMatrix(const ClusterDynamicsParameters<dim>& cdp_in);
        const MatrixType operator() (const ElementType& elem, const BaryType& bary) const;
    };

    template <int dim>
    struct InvDscaling : public EvalFunction<InvDscaling<dim>>
    {
        typedef typename DislocationDynamicsBase<dim>::ElementType ElementType;
        typedef Eigen::Matrix<double,dim+1,1> BaryType;
        constexpr static int mSize=ClusterDynamicsParameters<dim>::mSize;
        constexpr static int rows=mSize;
        constexpr static int cols=rows;
        typedef Eigen::Matrix<double,rows,cols> MatrixType;

        const ClusterDynamicsParameters<dim>& cdp;
        
        InvDscaling(const ClusterDynamicsParameters<dim>& cdp_in);
        const MatrixType operator() (const ElementType& elem, const BaryType& bary) const;
    };

    template<int dim>
    struct ClusterDynamicsFEM
    {
        typedef typename MicrostructureBase<dim>::MatrixDim MatrixDim;
        typedef typename MicrostructureBase<dim>::VectorDim VectorDim;
        typedef typename MicrostructureBase<dim>::ElementType ElementType;
        typedef typename MicrostructureBase<dim>::SimplexDim SimplexDim;
        typedef typename MicrostructureBase<dim>::NodeType NodeType;
        typedef typename MicrostructureBase<dim>::VectorMSize VectorMSize;
        typedef FiniteElement<ElementType> FiniteElementType;
        static constexpr int dVorder=4;
        typedef IntegrationDomain<FiniteElementType,0,dVorder,GaussLegendre> VolumeIntegrationDomainType;
        static constexpr int mSize=ClusterDynamicsParameters<dim>::mSize;
        static constexpr int iSize=ClusterDynamicsParameters<dim>::iSize;
        typedef TrialFunction<'m',mSize,FiniteElementType> MobileTrialType;
        typedef TrialFunction<'i',iSize,FiniteElementType> ImmobileTrialType;
//        typedef TrialFunction<'z',dim,FiniteElementType> DiffusiveTrialType;
        typedef TrialGrad<MobileTrialType> MobileGradType;
        typedef TrialProd<FluxMatrix<dim>,MobileGradType> MobileFluxType;
        
        typedef TrialProd<InvDscaling<dim>,MobileTrialType> MobileTestType;
        typedef TrialGrad<MobileTestType> MobileTestGradType;
        typedef BilinearForm<MobileTestGradType,TrialProd<Constant<double,1,1>,MobileFluxType>> MobileBilinearFormType;

//        typedef BilinearForm<MobileGradType,TrialProd<Constant<double,1,1>,MobileFluxType>> MobileBilinearFormType;
        typedef BilinearWeakForm<MobileBilinearFormType,VolumeIntegrationDomainType> MobileBilinearWeakFormType;

        typedef TrialFunction<'d',mSize,FiniteElementType> MobileIncrementTrialType;
        typedef TrialProd<InvDscaling<dim>,MobileIncrementTrialType> MobileIncrementTestType;
        typedef TrialGrad<MobileIncrementTestType> MobileIncrementTestGradType;

        typedef TrialGrad<MobileIncrementTrialType> MobileIncrementGradType;
        typedef TrialProd<FluxMatrix<dim>,MobileIncrementGradType> MobileIncrementFluxType;
        typedef BilinearForm<MobileIncrementTestGradType,TrialProd<Constant<double,1,1>,MobileIncrementFluxType>> MobileIncrementBilinearFormType;
        typedef BilinearWeakForm<MobileIncrementBilinearFormType,VolumeIntegrationDomainType> MobileIncrementBilinearWeakFormType;
        typedef Eigen::SparseMatrix<double,Eigen::RowMajor> SparseMatrixType;
#ifdef CHOLMOD_H // SuiteSparse Cholmod (LLT) module
    typedef Eigen::CholmodSupernodalLLT<SparseMatrixType> LltSolverType;
#else
    typedef Eigen::SimplicialLLT<SparseMatrixType> LltSolverType;
#endif
        
#ifdef UMFPACK_H // SuiteSparse Umfpack (LU) module
    typedef Eigen::UmfPackLU<SparseMatrixType> LuSolverType;
#else
    typedef Eigen::SparseLU<SparseMatrixType> LuSolverType;
#endif
                
        typedef FixedDirichletSolver<LltSolverType,Eigen::ConjugateGradient<SparseMatrixType>> MobileSolverType;
        typedef FixedDirichletSolver<LuSolverType,Eigen::BiCGSTAB<SparseMatrixType>> MobileReactionSolverType;

        const DislocationDynamicsBase<dim>& ddBase;
        const ClusterDynamicsParameters<dim>& cdp;
        const InvDscaling<dim> iDs;
        
        const Eigen::Matrix<double,mSize,mSize> invTrD;
        
        MobileTrialType mobileClusters;
        MobileGradType mobileGrad;
        MobileFluxType mobileFlux;
        ImmobileTrialType immobileClusters;

        const int nodeListInternalExternal;
        MobileIncrementTrialType mobileClustersIncrement;
        VolumeIntegrationDomainType dV;
        MobileBilinearWeakFormType mBWF;
        MobileIncrementBilinearWeakFormType dmBWF;

        MobileSolverType mSolver;
        bool solverInitialized;

        const Eigen::VectorXd cascadeGlobalProduction;

        /*! Whether solve() advances the immobile population after the mobile
         *  QSSA solve. Read from DD.txt as `useImmobileSolver`, default 1.
         *
         *  Setting it to 0 turns a DDomp run into the FAST STEP ALONE of the
         *  two-time-scale split: the steady mobile field C_M*(x) is solved for
         *  the immobile state currently in the CD block, and the immobile field
         *  is passed through untouched. That is what lets an external driver
         *  own the slow step -- the ZrMicro coupled march integrates the
         *  immobile ODEs at every node with CVODE and calls back here for a
         *  fresh C_M*(x) each time. The alternative, replaying a mobile field
         *  recorded by some other run, is not an operator split at all: the
         *  mobile field would never respond to the immobile state the march is
         *  building, so a seed away from quasi-steady state could never relax.
         */
        const bool useImmobileSolver;

        /*! Run-time controls on the mobile fixed-point loop. Upstream that loop
         *  is `while(cError>1e-5)` with no cap; these expose its tolerance, cap
         *  it, and allow under-relaxation and a different error measure. All
         *  are optional DD.txt scalars; the defaults reproduce the historical
         *  iteration except that it can no longer run forever.
         */
        const double mobileSolverTolerance;      // cTol,        default 1e-5
        const int    mobileSolverMaxIterations;  // 0=unlimited, default 200
        const double mobileSolverRelaxation;     // w,           default 1.0
        const int    mobileSolverErrorMode;      // 0/1/2,       default 0
        /*! Whether the positivity floor is applied INSIDE the fixed-point loop.
         *  1 (default) is MoDELib3's behaviour. 0 is upstream's: fullCD has no
         *  mobile clamp at all, and clamping inside the loop turns the
         *  iteration into a projected Newton method, which cycles when
         *  undamped. With 0 the floor is applied once, after the loop.
         */
        const int    mobileSolverClampInLoop;    // 0/1,         default 1

        static bool getUseImmobileSolver(const DislocationDynamicsBase<dim>& ddBase);
        static double ddScalar(const DislocationDynamicsBase<dim>& ddBase,
                               const std::string& key,const double& fallback);

        ClusterDynamicsFEM(const DislocationDynamicsBase<dim>& ddBase_in,const ClusterDynamicsParameters<dim>& cdp_in);
        void clampMobileClusters();   // positivity floor (ZrMicro C_floor)
        void writeNodePositions() const;  // evl/cdNodes.txt, for spatial plotting
        void solveMobileClusters();
        void solveImmobileClusters();
        void solve();
//        void applyBoundaryConditions();
        void initializeConfiguration(const DDconfigIO<dim>& configIO,const std::ofstream& f_file,const std::ofstream& F_labels);
        void initializeSolver();
        VectorDim inelasticDisplacementRate(const VectorDim&, const NodeType* const, const ElementType* const,const SimplexDim* const) const;

    };
    
}
#endif

