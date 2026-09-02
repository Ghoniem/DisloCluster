#ifndef model_FixedDirichletSolver_H_
#define model_FixedDirichletSolver_H_

#include<Eigen/SparseCore>
#include<algorithm>
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

        /*! Sparsity-pattern cache for the Newton loop of the mobile solve.
         *
         *  computeFromTriplets() is called once per iteration with a triplet
         *  list whose (row,col) PATTERN is invariant -- same mesh, same weak
         *  forms, same element loops -- and whose values alone change. Building
         *  A and A1 with setFromTriplets each time therefore re-sorts and
         *  re-allocates an unchanged structure twice per iteration, and
         *  setFromTriplets is serial. Measured on the coupled march, that
         *  rebuild ("precond" in the stage timers, though the preconditioner
         *  here is Eigen's diagonal one and costs nothing) is 46% of the fast
         *  solve at 200 nm and 51% at 1000 nm -- the single largest phase.
         *
         *  aSlot[k]    : triplet k -> index into A.valuePtr()
         *  a1FromA[j]  : A value index j -> A1 value index, -1 if struck out
         *
         *  Values are accumulated in TRIPLET ORDER, which is the order
         *  setFromTriplets sums duplicates in, so the refilled matrix is
         *  bit-identical to the rebuilt one. The cache is rebuilt whenever the
         *  triplet count or either size moves, so a changed pattern is a
         *  correctness non-event.
         */
        std::vector<int> aSlot;
        std::vector<int> a1FromA;
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
            computeFromTriplets<BilinearWeakFormType>(bWF.globalTriplets());
        }

        /*! compute() from an already-assembled triplet list.
         *
         *  The Newton loop of the mobile solve re-assembles four weak forms
         *  per iteration, three of which cannot have changed: dmBWF is a
         *  member, bWF_R1 is constant by construction, and bWF_RI depends only
         *  on the immobile field, which is frozen for the whole of a fast
         *  step. Only the second-order reaction term moves. This overload lets
         *  the caller assemble the invariant part once and splice the one
         *  varying term in, IN THE ORIGINAL ORDER so that setFromTriplets sums
         *  duplicates exactly as before and the matrix stays bit-identical.
         */
        template<typename BilinearWeakFormType>
        void computeFromTriplets(const std::vector<Eigen::Triplet<double> >& aTriplets)
        {
            dirichletConditions=&TrialBase<typename BilinearWeakFormType::TrialFunctionType>::dirichletConditions();
            dofVector=&TrialBase<typename BilinearWeakFormType::TrialFunctionType>::dofVector();
            gSize=TrialBase<typename BilinearWeakFormType::TrialFunctionType>::gSize();
            cSize=dirichletConditions->size();
            tSize = gSize-cSize;

            /*! FAST PATH: pattern unchanged, refill values in place.
             *  Skips both setFromTriplets sorts and every allocation.
             */
            const bool reuse(aSlot.size()==aTriplets.size()
                             && size_t(A.rows())==gSize
                             && size_t(A1.rows())==tSize
                             && A1.rows()>0);
            if(reuse)
            {
                std::fill(A.valuePtr(),A.valuePtr()+A.nonZeros(),0.0);
                for(size_t k=0;k<aTriplets.size();++k)
                {
                    A.valuePtr()[aSlot[k]]+=aTriplets[k].value();
                }
                std::fill(A1.valuePtr(),A1.valuePtr()+A1.nonZeros(),0.0);
                for(int j=0;j<A.nonZeros();++j)
                {
                    const int d(a1FromA[size_t(j)]);
                    if(d>=0)
                    {
                        A1.valuePtr()[d]=A.valuePtr()[j];
                    }
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
                return;
            }

            A.resize(gSize,gSize);
            A.setFromTriplets(aTriplets.begin(),aTriplets.end());
            A.makeCompressed();

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
                A1.makeCompressed();
            }

            /*! Record the pattern so every later call takes the fast path.
             *  Both lookups are a binary search inside one compressed row, so
             *  this costs O(nnz log nnz) ONCE against a setFromTriplets pair
             *  per Newton iteration thereafter.
             */
            {
                auto slotOf=[](const SparseMatrixType& M,const int r,const int c)->int
                {// M is RowMajor and compressed: row r occupies [outer[r],outer[r+1])
                    const int* out(M.outerIndexPtr());
                    const int* inn(M.innerIndexPtr());
                    const int lo(out[r]), hi(out[r+1]);
                    const int* f(std::lower_bound(inn+lo,inn+hi,c));
                    return (f!=inn+hi && *f==c) ? int(f-inn) : -1;
                };
                aSlot.assign(aTriplets.size(),-1);
                bool ok(true);
                for(size_t k=0;k<aTriplets.size();++k)
                {
                    const int sl(slotOf(A,aTriplets[k].row(),aTriplets[k].col()));
                    if(sl<0){ ok=false; break; }
                    aSlot[k]=sl;
                }
                a1FromA.assign(size_t(A.nonZeros()),-1);
                if(ok)
                {
                    for(int r=0;r<A.outerSize() && ok;++r)
                    {
                        for(typename SparseMatrixType::InnerIterator it(A,r);it;++it)
                        {
                            const long rr(rMap[size_t(it.row())]);
                            const long cc(rMap[size_t(it.col())]);
                            if(rr>=0 && cc>=0)
                            {
                                const int sl(slotOf(A1,int(rr),int(cc)));
                                if(sl<0){ ok=false; break; }
                                a1FromA[size_t(&it.value()-A.valuePtr())]=sl;
                            }
                        }
                    }
                }
                if(!ok)
                {// pattern not recoverable -- disable the cache rather than guess
                    aSlot.clear();
                    a1FromA.clear();
                }
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
