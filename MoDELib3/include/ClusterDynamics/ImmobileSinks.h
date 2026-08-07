/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_ImmobileSinks_H_
#define model_ImmobileSinks_H_

#include <Eigen/Dense>
#include <EvalFunction.h>
#include <ClusterDynamicsParameters.h>

namespace model
{

/*! \brief First-order absorption of mobile species at the immobile (loop)
 *  population -- the "loop_abs" term of the 0-D model, Deliverable D1/M1
 *  Eqs. (83)-(89), written as a spatially-resolved reaction operator.
 *
 *  The mobile master equation (Eq. 1) carries the sink block
 *
 *      D = diag{omega .* C_s} + D_d * K_L                          (Eq. 10)
 *
 *  whose two parts are handled separately here:
 *
 *   - the NETWORK part, k2_v = rho_N and k2_i = Z_N rho_N (Eq. 66), is constant
 *     in space and is supplied through `otherSinks_SI` in the material file; it
 *     is already assembled into R1 by getR1() as -k2*Dbar.
 *
 *   - the LOOP part, K_L = diag{k2_m} with k2_m = sum_IL <k2_IL,m> summed over
 *     the immobile families (Eqs. 11-14), is a function of the LOCAL immobile
 *     state and therefore varies from point to point. It is evaluated here, in
 *     the same way SecondOrderReaction evaluates the R2 block from the local
 *     mobile field.
 *
 *  The GRAIN-BOUNDARY sink is deliberately absent: unlike the 0-D reduction,
 *  which must lump it into a sink strength, the spatially-resolved model
 *  captures it through the mobile-species flux being absorbed at the boundary
 *  under the Dirichlet condition of Eq. (19). Adding it here as well would
 *  double-count it.
 *
 *  Per family k and mobile species m the absorption rate is
 *
 *      D_m * Z_km * S_k * c_m ,
 *
 *  where S_k is the geometric sink strength returned by clusterDensity() --
 *  already interpolating the bi-pyramid law alpha_bp*4*pi*r*N (Eq. 14) into the
 *  planar loop law 2*pi*r*N (Eqs. 12-13) through the sigmoid of Eq. (30) -- and
 *  Z_km is the DAD capture efficiency generated from the diffusion tensors,
 *  Eq. (15). Losses enter with a NEGATIVE diagonal, matching the sign
 *  convention getR1() uses for otherSinks.
 *
 *  The operator is diagonal in species, so the |m|-scaling that R1cd applies
 *  (diag(|m|) R1 diag(1/|m|)) leaves it unchanged and is omitted. Note the rate
 *  above is a loss of CLUSTERS; the corresponding gain of loop CONTENT carries
 *  the extra factor |m_m| (a di-interstitial deposits two interstitials), which
 *  is applied on the immobile side in solveImmobileClusters().
 */
template <typename ImmobileTrialFunctionType, int mSizeIn>
struct ImmobileSinks : public EvalFunction<ImmobileSinks<ImmobileTrialFunctionType,mSizeIn>>
{

    constexpr static int rows=mSizeIn;
    constexpr static int cols=mSizeIn;
    constexpr static int dim=ImmobileTrialFunctionType::dim;
    constexpr static int iSize=ImmobileTrialFunctionType::rows;
    constexpr static int nFamilies=iSize/2;

    typedef EvalExpression<ImmobileTrialFunctionType> EvalFunctionType;

    const ClusterDynamicsParameters<dim>& cdp;
    const ImmobileTrialFunctionType& nI;      // immobile field: [n_k ; c_k]
    const EvalFunctionType nIe;
    const Eigen::Array<double,2,mSizeIn> Z;   // row0 <c> loops, row1 <a> loops
    Eigen::Array<double,1,mSizeIn> Dbar;      // orientation-averaged diffusivity
    Eigen::Array<int,1,nFamilies> vacancyFamily;

    /**********************************************************************/
    ImmobileSinks(const ImmobileTrialFunctionType& nI_in, const ClusterDynamicsParameters<dim>& cdp_in) :
    /* init */ cdp(cdp_in),
    /* init */ nI(nI_in),
    /* init */ nIe(nI_in),
    /* init */ Z(cdp_in.loopDADbias())
    {
        Dbar.setZero();
        if(!cdp.detD.empty())
        {
            for(int m=0;m<mSizeIn;++m)
            {
                Dbar(m)=std::pow(std::max(cdp.detD.begin()->second(m),0.0),1.0/3.0);
            }
        }
        for(int k=0;k<nFamilies;++k)
        {
            vacancyFamily(k)=(cdp.immobileSpeciesVector(k)<0.0)?1:0;
        }
    }

    /**********************************************************************/
    template<typename ElementType, typename BaryType>
    const Eigen::Matrix<double,rows,cols> operator() (const ElementType& ele, const BaryType& bary) const
    {
        const Eigen::Matrix<double,iSize,1> iVal(nIe(ele,bary));

        // ZrMicro's C_floor: stated per atom, so the content (a volume fraction)
        // takes it directly and the per-b^3 number density takes C_floor/Omega.
        const double cFloor(cdp.concentrationFloor);
        const double nFloor(cdp.concentrationFloor/cdp.omega);
        Eigen::Array<double,1,nFamilies> nk,ck;
        for(int k=0;k<nFamilies;++k)
        {
            nk(k)=std::max(iVal(k),nFloor);
            ck(k)=std::max(iVal(nFamilies+k),cFloor);
        }

        // Sink strength per family, Eqs. (12)-(14) with the bi-pyramid -> loop
        // interpolation of Eq. (28)/(30), times the per-family calibration
        // multiplier loopSinkScale (1 = purely geometric). The SAME multiplier is
        // applied to the loop growth flux in solveImmobileClusters(), mirroring the
        // 0-D where absorption and growth share one prefactor.
        const Eigen::Array<double,1,nFamilies> Sk(cdp.clusterDensity(ck,nk)*cdp.loopSinkScale);

        Eigen::Matrix<double,rows,cols> temp(Eigen::Matrix<double,rows,cols>::Zero());
        for(int m=0;m<rows;++m)
        {
            double k2(0.0);
            for(int k=0;k<nFamilies;++k)
            {
                if(std::isfinite(Sk(k)) && Sk(k)>0.0)
                {
                    k2+=Z(vacancyFamily(k)?0:1,m)*Sk(k);
                }
            }
            temp(m,m)=-k2*Dbar(m);      // loss: negative diagonal (cf. getR1)
        }
        return temp;
    }

};

}
#endif
