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


#ifndef model_ClusterDynamicsParameters_H_
#define model_ClusterDynamicsParameters_H_

#include <TerminalColors.h>
//#include <DislocatedMaterialBase.h>
#include <Polycrystal.h>
#include <DislocationDynamicsBase.h>
//#include <TrialBase.h>
//#include <EvalFunction.h>
//#include <EvalExpression.h>
//#include <DislocationStress.h>

namespace model
{

template<int dim>
struct ClusterDynamicsParameters
{
    static constexpr int mSize=4;     // e.g. Cv, Ci, C2i, C3i
    static constexpr int iSize=8;  // e.g. Nc, Na1, Na2, Na3, cv, ca1, ca2, ca3

    typedef Eigen::Matrix<double,dim,1> VectorDim;
    typedef Eigen::Matrix<double,dim,dim> MatrixDim;

    // Materials parameters
    const double kB;
    const double T; // Studied temperature in K
    const double omega; // Atomic volume
    const double b; // Burgers vector
    const double G0; // dpa/s, Dose rate

    // Mobile Species
    const Eigen::Array<double,1,mSize> msVector;
    const Eigen::Array<double,1,mSize> msRelRelaxVol;
    const Eigen::Array<double,1,mSize> msEf;
    const Eigen::Matrix<double,mSize,dim*(dim+1)/2> msEm; // for each species (row) em11 em12 em13 em22 em23 em33
    const Eigen::Matrix<double,mSize,dim*(dim+1)/2> msD0; // diffusion pre-exponential coeff. for each species (row) D011 D012 D013 D022 D023 D033
    const std::map<size_t,std::vector<Eigen::Matrix<double,dim,dim>>> D;   // map<grain_ID, vector of diffusion coeff for each species>
    const std::map<size_t,std::vector<Eigen::Matrix<double,dim,dim>>> invD;
    const std::map<size_t,Eigen::Array<double,mSize,1>> detD;
    const Eigen::Matrix<double,1,mSize> msCascadeFractions;
    const double msSurvivingEfficiency; // Surviving effciency
    const Eigen::Matrix<double,mSize,1> G; // dpa/s, Effective dose rate

    // Binding energies. MUST be declared before R1: getR1() reads Eb(k) to build
    // the cluster dissociation rates, and C++ initializes members in DECLARATION
    // order, not in the order of the constructor's initializer list. Declared
    // after R1 (as it was) this is an uninitialized read -- it evaluated to
    // exp(0)=1 instead of exp(-Eb/kT), inflating the 2i/3i dissociation rate to
    // the bare attempt frequency ~5e7 1/s and suppressing C2i/C3i by ~2.4e6.
    // Harmless while every Eb_eV was 5.0 (exp(-5/kT) ~ 1e-44 either way), which
    // is why it went unnoticed.
    const Eigen::Array<double,1,mSize> Eb;

    // First-order reaction
    const Eigen::Array<double,1,mSize> otherSinks;
    const std::map<std::pair<int,int>,double> reactionMap;
    const Eigen::Matrix<double,mSize,mSize> R1;
    const Eigen::Matrix<double,mSize,mSize> R1cd;
    // Second-order reaction
    const std::vector<Eigen::Matrix<double,mSize,mSize>> R2;
    /*! Homogeneous-clustering channels: the subset of reactionMap whose product
     *  size m_a+m_b is carried by NO mobile species, so the product leaves the
     *  mobile ladder altogether. Physically these are the reactions that nucleate
     *  a new loop; in the 0-D they are R_i_3i (i+3i -> 4i) and R_2i_2i
     *  (2i+2i -> 4i), which feed nucleation_rate_iL/aiL and nucleation_content_i.
     *  Stored as (a,b) -> K_ab, the SAME coefficient getR2() writes into R2, so
     *  the loops gain exactly what the mobile solve loses. Empty when iSize<=0.
     *  Consumed by ClusterDynamicsFEM::solveImmobileClusters(). */
    const std::map<std::pair<int,int>,double> loopNucChannels;

    // Bias factors
    const Eigen::Array<double,2,mSize> discreteDislocationBias;

    // Immobile Species
    const Eigen::Array<double,1,iSize/2> immobileSpeciesVector;
    const Eigen::Array<double,1,iSize/2> immobileSpeciesRelRelaxVol;
    const Eigen::Matrix<double,dim,iSize/2> immobileSpeciesBurgers;
    const Eigen::Array<double,1,iSize/2> immobileSpeciesBurgersMagnitude;
    const double a_bp; // bi-pyramid sink strength coefficient
    const double delVPyramid;
    const double w0;
    const double n_s;
    // (Eb is declared above, before R1 -- see the note there.)

    // Irradiation Production
    const double evc; // Vacancy cluster generation efficiency
    const double Nvmax; // Satuatrion number density for vacancy loops in m^-3
    const Eigen::Array<double,1,iSize/2> nmin; // Critical size for <c> pyramid -> loop
    const Eigen::Array<double,1,iSize/2> nmax;
    const Eigen::Array<double,1,iSize/2> r_min; // minimal loop sizes

    // ---- Immobile kinetics: Deliverable D1/M1 Sec. 2.2 -------------------
    // Parameters of the loop density (Eq. 34) and content (Eq. 39) equations.
    // All are read only when iSize>0 and are converted to MoDELib units here.
    const Eigen::Array<double,1,iSize/2> loopCascadeFractions; // eps_k, cascade-borne loop fraction (Eq. 51)
    const Eigen::Array<double,1,iSize/2> nNuc;                 // defects per cascade-nucleated loop (Eq. 36)
    const Eigen::Array<double,1,iSize/2> cLL;                  // like-loop coalescence coefficient (Eq. 99)
    const Eigen::Array<double,1,iSize/2> cLN;                  // loop-network coalescence coefficient (Eq. 99)
    const double kappaLL;                                      // Avrami overlap, like-loop channel (Eq. 98)
    const double kappaLN;                                      // Avrami overlap, loop-network channel (Eq. 98)
    const double tauVac;                                       // vacancy-loop dissolution lifetime (Eq. 95) [MoDELib time]
    const double rhoNetwork;                                   // network dislocation density [1/b^2]
    // DAD calibration, Eq. (15). p_m and Z0_m are supplied explicitly and fitted
    // to reproduce the 0-D empirical symmetric forms Z^a_i=1+d_i, Z^a_v=1-d_v,
    // Z^c_i=1-d_i, Z^c_v=1+d_v (Eqs. 63-64); see loopDADbias().
    const Eigen::Array<double,1,mSize> dadAnisotropy;          // p_m = (D_c/D_a)^(1/6)
    const Eigen::Array<double,1,mSize> dadZ0;                  // Z0_m, isotropic-limit bias
    /*! Per-family multiplier on the loop sink strength S_k = 2*pi*r_k*N_k, applied
     *  identically to the mobile-species sink (ImmobileSinks) and to the loop growth
     *  flux, as in the 0-D where both come from the same prefactor. Defaults to 1
     *  (purely geometric). Zr3d_ghoniem sets it to reproduce ZrMicro's convention;
     *  see loopSinkScale in the material file. */
    const Eigen::Array<double,1,iSize/2> loopSinkScale;
    /*! Positivity floor on every concentration, mirroring ZrMicro's C_floor
     *  (`rate_equations.py`, default 1e-20). Applied to the mobile species after
     *  each Newton update and to the immobile densities/contents after each
     *  sub-step. Optional material key `concentrationFloor`. */
    const double concentrationFloor;

    // Reaction map (types: parameters)
    const bool computeReactions;
//    const int use0DsinkStrength;
//
//    // Bias factors
//    const Eigen::Array<double,1,dim> Zv;
//    const Eigen::Array<double,1,dim> Zi;
//    const Eigen::Array<double,1,mSize> ZVec;
//    const Eigen::Array<double,1,dim> rc_il;


    // Discrete Loop Generation
    const double discreteDistanceFactor;

    
    ClusterDynamicsParameters(const DislocationDynamicsBase<dim>& ddBase /*const Polycrystal<dim>& poly*/);
    /*! Atomic volume [b^3] for the CD equations. Reads the optional material-file
     *  key atomicVolume_SI; falls back to Polycrystal::Omega (the lattice CELL
     *  volume, which for HEX holds two atoms) when the key is absent. */
    static double getClusterAtomicVolume(const DislocationDynamicsBase<dim>& ddBase);
    /*! Positivity floor for all concentrations; optional material-file key
     *  `concentrationFloor`, defaulting to ZrMicro's 1e-20. */
    static double getConcentrationFloor(const DislocationDynamicsBase<dim>& ddBase);
    std::map<std::pair<int,int>,double> getMap(const Eigen::Array<double,mSize*(mSize+1)/2,3> matrix_in) const;
    Eigen::Matrix<double,mSize,mSize> getR1() const;
    std::vector<Eigen::Matrix<double,mSize,mSize>> getR2() const;
    /*! Build loopNucChannels; see its declaration. */
    std::map<std::pair<int,int>,double> getLoopNucChannels() const;
    Eigen::Array<double,1,iSize/2> getImmobileSpeciesBurgersMagnitude(const std::map<size_t,Grain<dim>>& grains) const;
    std::map<size_t,std::vector<Eigen::Matrix<double,dim,dim>>> getD(const std::map<size_t,Grain<dim>>& grains) const;
    std::vector<Eigen::Matrix<double,dim,dim>> getDlocal() const;
    std::map<size_t,std::vector<Eigen::Matrix<double,dim,dim>>> getInvD() const;
    std::map<size_t,Eigen::Array<double,mSize,1>> getDetD() const;
    Eigen::Array<double,1,iSize> getInitLoopSinks(const Eigen::Array<double,1,iSize> initloopSinks_SI, const double b_SI) const;
    Eigen::Array<double,1,mSize> equilibriumMobileConcentration(const double& stressTrace) const;
    Eigen::Array<double,1,mSize> dislocationMobileConcentration(const VectorDim& b,const VectorDim& t,const VectorDim& fPK,const MatrixDim& stress) const;
    Eigen::Array<double,1,mSize> boundaryMobileConcentration(const double& stressTrace,const double& normalTraction) const;
    Eigen::Array<double,1,iSize/2> sigmoid(const Eigen::Array<double,1,iSize/2>& n) const;
    Eigen::Array<double,1,iSize/2> rpyr(const Eigen::Array<double,1,iSize/2>& n) const;
    Eigen::Array<double,1,iSize/2> rloop(const Eigen::Array<double,1,iSize/2>& n) const;
    Eigen::Array<double,1,iSize/2> sigmoidalVectorInterpolation(const Eigen::Array<double,1,iSize/2>& CI, const Eigen::Array<double,1,iSize/2>& N, const Eigen::Array<double,1,iSize/2>& lowValue, const Eigen::Array<double,1,iSize/2>& highValue) const;
    Eigen::Array<double,1,iSize/2> clusterRadius(const Eigen::Array<double,1,iSize/2>& CI, const Eigen::Array<double,1,iSize/2>& N) const;
    /*! Diffusional-anisotropy-difference capture efficiencies, Deliverable Eq. (15).
     *  Row 0 = vacancy-type loops (basal <c>), row 1 = interstitial-type loops
     *  (prismatic <a>); columns are the mobile species. The bias is GENERATED from
     *  the anisotropy of the species diffusion tensors, p_m=(D_c/D_a)^(1/6), and not
     *  posited as a scalar delta as in the 0-D reduction.
     */
    Eigen::Array<double,2,mSize> loopDADbias() const;
    Eigen::Array<double,1,iSize/2> clusterDensity(const Eigen::Array<double,1,iSize/2>& CI, const Eigen::Array<double,1,iSize/2>& N) const;
    Eigen::Array<double,dim,dim> sigmoidalMatrixInterpolation(const Eigen::Array<double,1,iSize/2>& CI, const Eigen::Array<double,1,iSize/2>& N, const Eigen::Array<double,dim,dim>& lowValue, const Eigen::Array<double,dim,dim>& highValue, const int& index) const;
    Eigen::Array<double,1,iSize/2> sigmoidalPlotVectorInterpolation(const Eigen::Array<double,1,iSize/2>& CI, const Eigen::Array<double,1,iSize/2>& N, const Eigen::Array<double,1,iSize/2>& lowValue, const Eigen::Array<double,1,iSize/2>& highValue) const;
    Eigen::Array<double,1,iSize/2> clusterPlotRadius(const Eigen::Array<double,1,iSize/2>& CI, const Eigen::Array<double,1,iSize/2>& N) const;
};

}
#endif
