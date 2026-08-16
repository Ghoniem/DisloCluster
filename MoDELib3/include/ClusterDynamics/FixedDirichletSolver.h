#ifndef model_FixedDirichletSolver_H_
#define model_FixedDirichletSolver_H_

#include<Eigen/SparseCore>
#include<Eigen/SparseLU>
#include<Eigen/IterativeLinearSolvers>

#include<TrialBase.h>

namespace model
{
    template<typename DirectSolverType,typename IterativeSolverType>
    class FixedDirichletSolver
    {// solves Ax=b given x=T*x1+g. Only b, and g can change
        
        static_assert(std::is_same<typename DirectSolverType::MatrixType,typename IterativeSolverType::MatrixType>::value, "Direct and Iterative MatrixType must be the same");
        typedef typename DirectSolverType::MatrixType SparseMatrixType;

//    public:
//        typedef Eigen::SparseMatrix<double,Eigen::RowMajor> SparseMatrixType;
        
 //   private:

        SparseMatrixType A1;
        
//        Eigen::SparseLU<SparseMatrixType> directSolver;
//        Eigen::BiCGSTAB<SparseMatrixType> iterativeSolver;
        
        DirectSolverType directSolver;
        IterativeSolverType iterativeSolver;

        const std::map<size_t,double>* dirichletConditions;
        const Eigen::Matrix<double,Eigen::Dynamic,1>* dofVector;
        size_t gSize;
        size_t cSize;
        size_t tSize;
        
        SparseMatrixType A;
        SparseMatrixType T;
        // global dof -> reduced index, -1 for a Dirichlet dof. Invariant with T.
        std::vector<long> rMap;

        public:
        
        const bool use_directSolver;

        const SparseMatrixType& getA() const
        {
            return A;
        }

        const SparseMatrixType& getT() const
        {
            return T;
        }
        
        FixedDirichletSolver(const bool& use_directSolver_in,const double& tol):
        /* init */ dirichletConditions(nullptr)
        /* init */,dofVector(nullptr)
        /* init */,gSize(0)
        /* init */,cSize(0)
        /* init */,tSize(0)
        /* init */,use_directSolver(use_directSolver_in)
        {
            iterativeSolver.setTolerance(tol);
        }
        
        Eigen::VectorXd solve(const Eigen::VectorXd& b) const
        {/*!@param[in] b the rhs of A*x=b
          * @param[in] y the guess for x
          */
            
            Eigen::VectorXd g(Eigen::VectorXd::Zero(gSize));
            Eigen::VectorXd guess(Eigen::VectorXd::Zero(tSize));
            
            size_t startRow=0;
            size_t col=0;
            
            for (const auto& cIter : *dirichletConditions)
            {
                const size_t& endRow = cIter.first;
                g(endRow)= cIter.second;
                for (size_t row=startRow;row!=endRow;++row)
                {
                    guess(col)=(*dofVector)(row);
                    col++;
                }
                startRow=endRow+1;
            }
            for (size_t row=startRow;row!=gSize;++row)
            {
                guess(col)=(*dofVector)(row);
                col++;
            }
                    
            Eigen::VectorXd b1(T.transpose()*(b-A*g));
            Eigen::VectorXd x(Eigen::VectorXd::Zero(b1.rows()));
            if(use_directSolver)
            {
                x=directSolver.solve(b1);
                if(directSolver.info()!=Eigen::Success)
                {
                    throw std::runtime_error("Direct FixedDirichletSolver failed.");
                }
            }
            else
            {
                x=iterativeSolver.solveWithGuess(b1,guess);
                if(iterativeSolver.info()!=Eigen::Success)
                {
                    throw std::runtime_error("Iterative FixedDirichletSolver failed.");
                }
            }
            return T*x+g;
        }

        template<typename BilinearWeakFormType>
        void compute(const BilinearWeakFormType& bWF)
        {
            dirichletConditions=&TrialBase<typename BilinearWeakFormType::TrialFunctionType>::dirichletConditions();
            dofVector=&TrialBase<typename BilinearWeakFormType::TrialFunctionType>::dofVector();
            gSize=TrialBase<typename BilinearWeakFormType::TrialFunctionType>::gSize();
            cSize=dirichletConditions->size();
            tSize = gSize-cSize;

            A.resize(gSize,gSize);
            const auto aTriplets(bWF.globalTriplets());
            A.setFromTriplets(aTriplets.begin(),aTriplets.end());

            /*! T and the global->reduced index map depend only on gSize and on
             *  the Dirichlet set, neither of which changes once the boundary
             *  conditions are imposed. compute() is called once per Newton
             *  iteration of the mobile solve, so rebuilding them every time was
             *  pure repetition. Rebuilt only when the sizes actually move.
             */
            if(T.rows()!=long(gSize) || T.cols()!=long(tSize) || rMap.size()!=gSize)
            {
                std::vector<Eigen::Triplet<double> > tTriplets;
                tTriplets.reserve(tSize);
                size_t startRow=0;
                size_t col=0;
                for (const auto& cIter : *dirichletConditions)
                {
                    const size_t& endRow = cIter.first;
                    for (size_t row=startRow;row!=endRow;++row)
                    {
                        tTriplets.emplace_back(row,col,1.0);
                        col++;
                    }
                    startRow=endRow+1;
                }
                for (size_t row=startRow;row!=gSize;++row)
                {
                    tTriplets.emplace_back(row,col,1.0);
                    col++;
                }
                T.resize(gSize,tSize);
                T.setFromTriplets(tTriplets.begin(),tTriplets.end());

                rMap.assign(gSize,-1);
                long rc=0;
                auto dIt=dirichletConditions->begin();
                for(size_t row=0;row<gSize;++row)
                {
                    if(dIt!=dirichletConditions->end() && dIt->first==row)
                    {
                        ++dIt;
                        continue;
                    }
                    rMap[row]=rc++;
                }
            }

            /*! A1 = T^T*A*T with T a SELECTION matrix -- one 1.0 per column and
             *  nothing else -- so the triple sparse-sparse product computes, at
             *  the cost of two full sparse products on a gSize x gSize matrix,
             *  a copy of A with the Dirichlet rows and columns struck out.
             *  Doing that directly is one pass over the nonzeros of A.
             *
             *  Bit-identical rather than merely close: A is already compressed,
             *  so every surviving entry of A1 is exactly one entry of A. No
             *  sums are re-associated and no duplicate triplets are merged.
             */
            {
                std::vector<Eigen::Triplet<double> > a1Triplets;
                a1Triplets.reserve(size_t(A.nonZeros()));
                for(int k=0;k<A.outerSize();++k)
                {
                    for(typename SparseMatrixType::InnerIterator it(A,k);it;++it)
                    {
                        const long r(rMap[size_t(it.row())]);
                        const long c(rMap[size_t(it.col())]);
                        if(r>=0 && c>=0)
                        {
                            a1Triplets.emplace_back(int(r),int(c),it.value());
                        }
                    }
                }
                A1.resize(tSize,tSize);
                A1.setFromTriplets(a1Triplets.begin(),a1Triplets.end());
            }
            
            if(use_directSolver)
            {
                directSolver.compute(A1);
                if(directSolver.info()!=Eigen::Success)
                {
                    throw std::runtime_error("FixedDirichletSolver failed.");
                }
            }
            else
            {
                iterativeSolver.compute(A1);
                if(iterativeSolver.info()!=Eigen::Success)
                {
                    throw std::runtime_error("FixedDirichletSolver failed.");
                }
            }
        }
    };

}
#endif
