# rate_equations.py
import numpy as np

class RateEquations:
    """
    Class to define the ODE system for the rate equations
    Maps the time derivatives to the y-array for LSODA integration
    """
    
    def __init__(self, input_data, reaction_rates):
        """
        Initialize with input data and reaction rates
        """
        self.input_data = input_data
        self.reaction_rates = reaction_rates
        
        # Set the back-reference so ReactionRates can access this instance
        self.reaction_rates.rate_equations = self
        
        # Define the order of concentrations in the y array
        self.concentration_names = [
            'Cv',      # 0 - Vacancy concentration
            'Ci',      # 1 - Interstitial concentration  
            'C2i',     # 2 - Di-interstitial concentration
            'C3i',     # 3 - Tri-interstitial concentration
            'CiL',     # 4 - Interstitial loop concentration
            'CaiL',    # 5 - Aligned interstitial loop concentration
            'CvL',     # 6 - Vacancy loop concentration
            'CavL',    # 7 - Aligned vacancy loop concentration
            'CiL_i',   # 8 - Interstitials in interstitial loops
            'CaiL_i',  # 9 - Interstitials in aligned interstitial loops
            'CvL_v',   # 10 - Vacancies in vacancy loops
            'CavL_v'   # 11 - Vacancies in aligned vacancy loops
        ]
        
        self.n_physical = len(self.concentration_names)

        # Point-defect conservation accumulators (monotonic time integrals).
        # These do not feed back into the physics; they integrate the production
        # and physical loss rates so post-processing can report the relative
        # error in the interstitial and vacancy atom balances.
        # Order MUST match rate_equations.cpp indices 12..17.
        self.accumulator_names = [
            'cum_prod_i',    # 12 - integral of interstitial production  G_i + 2 G_2i + 3 G_3i
            'cum_recomb_i',  # 13 - integral of recombination  R_i_v + R_2i_v + R_3i_v
            'cum_sink_i',    # 14 - integral of i sink absorption  R_i_s + 2 R_2i_s + 3 R_3i_s
            'cum_prod_v',    # 15 - integral of vacancy production  G_v
            'cum_recomb_v',  # 16 - integral of recombination (same Frenkel events)
            'cum_sink_v',    # 17 - integral of v sink absorption  R_v_s
        ]

        # Evolving network dislocation density rho_N (state row AFTER the
        # accumulators). It starts at the grown-in, T-dependent material value
        # and grows as loops are absorbed into the network (loop-network
        # coalescence). It feeds back into the point-defect sink strengths and
        # the coalescence gates. Order MUST match rate_equations.cpp index 18.
        self.idx_rhoN = self.n_physical + len(self.accumulator_names)   # 18

        self.n_equations = self.idx_rhoN + 1

        # Concentration floor: any value below this is unphysical; clamped during integration
        self.C_floor = input_data.model_params.get('C_floor', 1e-20)

        # Operator-split QSSA (two-time-scale ZrMicro <-> MoDELib2-NNL coupling).
        # When True, the four mobile species (indices 0..3) are held fixed at
        # their input values (their derivatives are zeroed), so this RHS acts as
        # the local IMMOBILE micro-model with a frozen, externally-supplied
        # steady mobile field. Set by the coupling driver; default False leaves
        # the standalone 0-D model unchanged.
        self.freeze_mobile = False

        print(f"Initialized rate equations system with {self.n_equations} equations "
              f"({self.n_physical} physical + {len(self.accumulator_names)} conservation "
              f"accumulators + 1 evolving rho_N)")
        print(f"  C_floor = {self.C_floor:.2e}")
    
    def ode_system(self, t, y):
        """
        Main ODE system function for LSODA integration
        
        Parameters:
        t: float - current time
        y: array - current concentrations [Cv, Ci, C2i, C3i, CiL, CaiL, CvL, CavL, CiL_i, CaiL_i, CvL_v, CavL_v]
        
        Returns:
        dydt: array - time derivatives
        """
        
        # Apply concentration floor before any rate evaluation
        y = np.maximum(y, self.C_floor)

        # Update reaction rates with current state
        self.reaction_rates.update_state(y, t)
        
        # Initialize derivatives array
        dydt = np.zeros(self.n_equations)
        
        # Extract concentrations for readability
        Cv, Ci, C2i, C3i = y[0], y[1], y[2], y[3]
        CiL, CaiL, CvL, CavL = y[4], y[5], y[6], y[7]
        CiL_i, CaiL_i, CvL_v, CavL_v = y[8], y[9], y[10], y[11]
        
        # Equation 1: dCv/dt (Equations 34-35)
        dydt[0] = self.dCv_dt(y)
        
        # Equation 2: dCi/dt (Equations 36-37)
        dydt[1] = self.dCi_dt(y)
        
        # Equation 3: dC2i/dt (Equations 38-39)
        dydt[2] = self.dC2i_dt(y)
        
        # Equation 4: dC3i/dt (Equations 40-41)
        dydt[3] = self.dC3i_dt(y)
        
        # Equation 5: dCiL/dt (Equation 42)
        dydt[4] = self.dCiL_dt(y)
        
        # Equation 6: dCaiL/dt (Equation 43)
        dydt[5] = self.dCaiL_dt(y)
        
        # Equation 7: dCvL/dt (Equation 44)
        dydt[6] = self.dCvL_dt(y)
        
        # Equation 8: dCavL/dt (Equation 45)
        dydt[7] = self.dCavL_dt(y)
        
        # Equation 9: dCiL_i/dt (Equation 46)
        dydt[8] = self.dCiL_i_dt(y)
        
        # Equation 10: dCaiL_i/dt (Equation 47)
        dydt[9] = self.dCaiL_i_dt(y)
        
        # Equation 11: dCvL_v/dt (Equation 48)
        dydt[10] = self.dCvL_v_dt(y)
        
        # Equation 12: dCavL_v/dt (Equation 49)
        dydt[11] = self.dCavL_v_dt(y)

        # Equations 13-18: point-defect conservation accumulators (indices 12-17).
        # Mirrors rate_equations.cpp. These integrate the production and physical
        # loss channels; the residual vs stored atoms quantifies non-conservation.
        rr = self.reaction_rates
        _, _, loop_recomb = rr.loop_absorption(y)               # defect-loop recombination
        # Interstitial atoms produced = free monomers/clusters (G_i + 2 G_2i +
        # 3 G_3i) + cascade loop-borne (G_iL + G_aiL), the eps_iL channel now
        # seeded as a-loop content. Sums to the full dpa rate G, mirroring prod_v.
        prod_i = (rr.G_i() + 2.0 * rr.G_2i() + 3.0 * rr.G_3i()
                  + rr.G_iL() + rr.G_aiL())                     # interstitial atoms produced
        # Vacancy atoms produced = free (G_v) + cascade loop-borne (G_vL+G_avL),
        # the eps_vL channel now seeded as loop content. Sums to the full dpa
        # rate G, restoring the i/v production balance (prod_v was 0.9*G before).
        prod_v = rr.G_v() + rr.G_vL() + rr.G_avL()              # vacancies produced
        recomb = rr.R_i_v(y) + rr.R_2i_v(y) + rr.R_3i_v(y) + loop_recomb  # 1 i and 1 v per event
        sink_i = rr.R_i_s(y) + 2.0 * rr.R_2i_s(y) + 3.0 * rr.R_3i_s(y)  # i absorption at network
        sink_v = rr.R_v_s(y)                                     # v absorption at network

        dydt[12] = prod_i    # cum_prod_i
        dydt[13] = recomb    # cum_recomb_i
        dydt[14] = sink_i    # cum_sink_i
        dydt[15] = prod_v    # cum_prod_v
        dydt[16] = recomb    # cum_recomb_v  (same Frenkel events)
        dydt[17] = sink_v    # cum_sink_v

        # ── Geometric loop coalescence (growth-flux driven) ──────────────────
        # (1) like-loop coarsening: number density drops, content conserved
        #     (saturates the loop densities; only sink for i-loops).
        # (2) loop-network coalescence: number AND content removed; the removed
        #     content is booked as network sink absorption so the point-defect
        #     balance stays closed.  Mirrors rate_equations.cpp.
        coal = rr.coalescence_rates(y)
        dydt[4]  -= coal['iL']['num']      # dCiL/dt
        dydt[5]  -= coal['aiL']['num']     # dCaiL/dt
        dydt[6]  -= coal['vL']['num']      # dCvL/dt
        dydt[7]  -= coal['avL']['num']     # dCavL/dt
        dydt[8]  -= coal['iL']['content']  # dCiL_i/dt   (loop-network channel)
        dydt[9]  -= coal['aiL']['content'] # dCaiL_i/dt
        dydt[10] -= coal['vL']['content']  # dCvL_v/dt
        dydt[11] -= coal['avL']['content'] # dCavL_v/dt
        # Content absorbed by the network is counted as sink absorption.
        dydt[14] += coal['iL']['content'] + coal['aiL']['content']   # cum_sink_i
        dydt[17] += coal['vL']['content'] + coal['avL']['content']   # cum_sink_v

        # ── Evolving network dislocation density rho_N (index idx_rhoN) ───────
        # Grows by the line length of loops absorbed into the network
        # (loop-network coalescence) and recovers (first-order) toward the
        # grown-in seed, so it saturates.  Mirrors rate_equations.cpp.
        dydt[self.idx_rhoN] = rr.rho_N_source(y, coal=coal)

        # Enforce non-negativity on the physical species only: zero out any
        # derivative that would drive a floored concentration further negative
        # (handles rapid annealing). Accumulators are monotonic and untouched.
        n = self.n_physical
        dydt[:n] = np.where((y[:n] <= self.C_floor) & (dydt[:n] < 0), 0.0, dydt[:n])

        # Operator-split QSSA: hold the mobile species (0..3) frozen at their
        # input values by zeroing their derivatives LAST, after every reaction
        # term has already used them. Mirrors the C++ freeze_mobile branch.
        if getattr(self, 'freeze_mobile', False):
            dydt[0:4] = 0.0

        return dydt
    
    def dCv_dt(self, y):
        """
        Vacancy concentration rate equation (Equations 34-35)
        """
        y = np.maximum(y, self.C_floor)
        
        # Generation term
        generation = self.reaction_rates.G_v()

        # Loop absorption debits the free vacancy pool (mass-conserving coupling)
        _, loop_abs_v, _ = self.reaction_rates.loop_absorption(y)

        # Vacancies released back to the free pool by annealing vacancy loops
        # (internal transfer loop content -> Cv, conserves total vacancy atoms).
        loop_anneal_release = (self.reaction_rates.annealing_content_vL(y)
                               + self.reaction_rates.annealing_content_avL(y))

        # Consumption and reaction terms (R_v_s is network-only)
        reactions = self.reaction_rates.R_i_v(y) + self.reaction_rates.R_2i_v(y) \
            + self.reaction_rates.R_3i_v(y) + self.reaction_rates.R_v_s(y) \
            + loop_abs_v

        return generation + loop_anneal_release - reactions
    
    def dCi_dt(self, y):
        """
        Interstitial concentration rate equation (Equations 36-37)
        """
        y = np.maximum(y, self.C_floor)
        # Cv, Ci, C2i, C3i = y[0], y[1], y[2], y[3]
        
        # Generation term
        generation = self.reaction_rates.G_i()+self.reaction_rates.R_2i_v(y)

        # Emission terms
        emission = 2.*self.reaction_rates.emission_2i(y)+3.*self.reaction_rates.emission_3i(y)

        # Loop absorption debits the free interstitial pool (mass-conserving coupling)
        loop_abs_i, _, _ = self.reaction_rates.loop_absorption(y)

        # Consumption terms (R_i_s is network-only; loop capture via loop_abs_i)
        consumption = self.reaction_rates.R_i_v(y) +self.reaction_rates.R_i_i(y) +self.reaction_rates.R_i_2i(y) \
                      + self.reaction_rates.R_i_3i(y) + self.reaction_rates.R_i_s(y) \
                      + loop_abs_i
        return generation + emission - consumption
    
    def dC2i_dt(self, y):
        """
        Di-interstitial concentration rate equation (Equations 38-39)
        """
        y = np.maximum(y, self.C_floor)
        
        # Generation from cascades
        generation = self.reaction_rates.G_2i()

        # Formation from i+i (0.5*R_i_i: one di-interstitial per two monomers
        # consumed, since R_i_i = 2*omega_i*Ci^2 is the monomer-loss rate)
        formation = 0.5 * self.reaction_rates.R_i_i(y) + self.reaction_rates.R_3i_v(y)

        # Thermal emission (full dissolution 2i -> 2i monomers; the released
        # interstitials are added in dCi_dt as 2*emission_2i). Mirrors rate_equations.cpp.
        emission = self.reaction_rates.emission_2i(y)

        # Consumption terms
        consumption = self.reaction_rates.R_2i_v(y) + self.reaction_rates.R_i_2i(y) \
                      + self.reaction_rates.R_2i_2i(y) + self.reaction_rates.R_2i_s(y)

        return generation + formation - consumption - emission
    
    def dC3i_dt(self, y):
        """
        Tri-interstitial concentration rate equation (Equations 40-41)
        """
        y = np.maximum(y, self.C_floor)
        
        # Generation from cascades
        generation = self.reaction_rates.G_3i()
        
        # Formation from i+2i
        formation = self.reaction_rates.R_i_2i(y)

        # Thermal emission (full dissolution 3i -> 3i monomers; the released
        # interstitials are added in dCi_dt as 3*emission_3i). Mirrors rate_equations.cpp.
        emission = self.reaction_rates.emission_3i(y)

        # Consumption terms
        consumption = self.reaction_rates.R_3i_v(y) + self.reaction_rates.R_i_3i(y) + \
            self.reaction_rates.R_3i_s(y) + self.reaction_rates.R_2i_3i(y)

        return generation + formation - consumption - emission
    
    def dCiL_dt(self, y):
        """
        Interstitial loop concentration rate equation (Equation 42)
        """
        y = np.maximum(y, self.C_floor)
        # Nucleation rate
        nucleation = self.reaction_rates.nucleation_rate_iL(y)
        
        return nucleation
    
    def dCaiL_dt(self, y):
        """
        Aligned interstitial loop concentration rate equation (Equation 43)
        """
        y = np.maximum(y, self.C_floor)
        # Nucleation rate
        nucleation = self.reaction_rates.nucleation_rate_aiL(y)
        
        return nucleation
    
    def dCvL_dt(self, y):
        """
        Vacancy loop concentration rate equation (Equation 44)
        """
        y = np.maximum(y, self.C_floor)
        
        # Nucleation rate
        nucleation = self.reaction_rates.nucleation_rate_vL(y)

        # Thermal annealing
        annealing = self.reaction_rates.annealing_rate_vL(y)

        return nucleation - annealing

    def dCavL_dt(self, y):
        """
        Aligned vacancy loop concentration rate equation (Equation 45)
        """
        y = np.maximum(y, self.C_floor)
        # Nucleation rate
        nucleation = self.reaction_rates.nucleation_rate_avL(y)

        # Thermal annealing
        annealing = self.reaction_rates.annealing_rate_avL(y)

        return nucleation - annealing
    
    def dCiL_i_dt(self, y):
        """
        Interstitials in interstitial loops rate equation (Equation 46).
        Growth from flux + interstitial content deposited at clustering
        nucleation (non-aligned fraction f_na) + cascade content seed (G_iL
        atoms, the eps_iL channel — counted in cum_prod_i), so loop nucleation
        conserves mass.
        """
        y = np.maximum(y, self.C_floor)
        f_na = self.input_data.derived['f_na']
        return (self.reaction_rates.loop_growth_rate_iL(y)
                + f_na * self.reaction_rates.nucleation_content_i(y)
                + self.reaction_rates.G_iL())   # cascade content seed

    def dCaiL_i_dt(self, y):
        """
        Interstitials in aligned interstitial loops rate equation (Equation 47).
        Growth from flux + interstitial content deposited at clustering
        nucleation (aligned fraction f_a) + cascade content seed (G_aiL atoms,
        the eps_iL channel — counted in cum_prod_i).
        """
        y = np.maximum(y, self.C_floor)
        f_a = self.input_data.derived['f_a']
        return (self.reaction_rates.loop_growth_rate_aiL(y)
                + f_a * self.reaction_rates.nucleation_content_i(y)
                + self.reaction_rates.G_aiL())   # cascade content seed
    
    def dCvL_v_dt(self, y):
        """
        Vacancies in vacancy loops rate equation (Equation 48).
        Flux-driven growth + cascade nucleation content seed (G_vL vacancy
        atoms, the eps_vL channel — counted in cum_prod_v) - content released
        by loop annealing back to the free pool. The seed is essential: without
        finite content the sqrt(CvL*CvL_v) growth prefactor pins CvL_v at the
        floor and vacancy loops can never grow.
        """
        y = np.maximum(y, self.C_floor)
        return (self.reaction_rates.loop_growth_rate_vL(y)
                + self.reaction_rates.G_vL()
                - self.reaction_rates.annealing_content_vL(y))

    def dCavL_v_dt(self, y):
        """
        Vacancies in aligned vacancy loops rate equation (Equation 49).
        Growth + cascade content seed (G_avL) - annealing content release.
        """
        y = np.maximum(y, self.C_floor)
        return (self.reaction_rates.loop_growth_rate_avL(y)
                + self.reaction_rates.G_avL()
                - self.reaction_rates.annealing_content_avL(y))
    
    def calculate_r_iL(self, y):
        """
        Calculate interstitial loop radius (Equation 54)
        """
        y = np.maximum(y, self.C_floor)
        CiL, CiL_i = y[4], y[8]
        l_a = self.input_data.derived['l_a']
        
        if CiL > 0:
            return l_a * np.sqrt(CiL_i / CiL)
        else:
            return 5e-9
    
    def calculate_r_aiL(self, y):
        """
        Calculate aligned interstitial loop radius (Equation 56)
        """
        y = np.maximum(y, self.C_floor)
        CaiL, CaiL_i = y[5], y[9]
        l_a = self.input_data.derived['l_a']
        
        if CaiL > 0:
            return l_a * np.sqrt(CaiL_i / CaiL)
        else:
            return 5e-9
    
    def calculate_r_vL(self, y):
        """
        Calculate vacancy loop radius (Equation 55)
        """
        y = np.maximum(y, self.C_floor)
        CvL, CvL_v = y[6], y[10]
        l_c = self.input_data.derived['l_c']
        
        if CvL > 0:
            return l_c * np.sqrt(CvL_v / CvL)
        else:
            return 5e-9
    
    def calculate_r_avL(self, y):
        """
        Calculate aligned vacancy loop radius (Equation 57)
        """
        y = np.maximum(y, self.C_floor)
        CavL, CavL_v = y[7], y[11]
        l_c = self.input_data.derived['l_c']
        
        if CavL > 0:
            return l_c * np.sqrt(CavL_v / CavL)
        else:
            return 5e-9
    
    def calculate_strain_rates(self, y):
        """
        Calculate mechanical strain rates (Section 4.1)
        """
        # y = np.maximum(y, 0e-20)  # Element-wise maximum
        # Extract loop concentrations
        CiL_i, CaiL_i, CvL_v, CavL_v = y[8], y[9], y[10], y[11]
        
        # Strain factors
        A_a = self.input_data.model_params['A_a']
        A_c = self.input_data.model_params['A_c']
        

        # Growth strains
        strain_iL = A_a * CiL_i
        strain_aiL = A_a * CaiL_i
        strain_vL = -A_c * CvL_v
        strain_avL = -A_c * CavL_v

        # Total strains
        epsilon_a = strain_iL + strain_aiL
        epsilon_c = strain_vL + strain_avL
        
        # Irradiation creep strains
        creep_iL = strain_aiL - 0.5 * strain_iL
        creep_vL = strain_avL - 0.5 * strain_vL
        
        # Network dislocation contribution (simplified)
        creep_N = 0.0  # Would need network dislocation evolution
        
        creep_rate = creep_iL + creep_vL + creep_N
        
        return {
            'epsilon_a': epsilon_a,
            'epsilon_c': epsilon_c,
            'creep_rate': creep_rate,
            'strain_iL': strain_iL,
            'strain_aiL': strain_aiL,
            'strain_vL': strain_vL,
            'strain_avL': strain_avL
        }
    
    def calculate_hardening(self, y):
        """
        Calculate radiation hardening (Section 4.3)
        """
        # Extract concentrations
        y = np.maximum(y, self.C_floor)

        Cv, Ci, C2i, C3i = y[0], y[1], y[2], y[3]
        CiL, CaiL, CvL, CavL = y[4], y[5], y[6], y[7]
        
        # Material properties
        mu = 30e9  # Shear modulus (Pa) - example value for Zr
        b = self.input_data.physical_props['b_a']  # Burgers vector magnitude
        M = self.input_data.physical_props['M']  # Schmidt factor
        
        # Strength coefficients
        alpha_v = self.input_data.model_params['alpha_v']
        alpha_i = self.input_data.model_params['alpha_i']
        alpha_iL = self.input_data.model_params['alpha_iL']
        alpha_aiL = self.input_data.model_params['alpha_aiL']
        alpha_vL = self.input_data.model_params['alpha_vL']
        alpha_avL = self.input_data.model_params['alpha_avL']
        alpha_N = self.input_data.model_params['alpha_N']
        
        # Convert concentrations to number densities
        Omega = self.input_data.physical_props['Omega']
        N_v = Cv / Omega
        N_i = Ci / Omega
        N_2i = C2i / Omega
        N_3i = C3i / Omega
        N_iL = CiL / Omega
        N_aiL = CaiL / Omega
        N_vL = CvL / Omega
        N_avL = CavL / Omega
        
        # Defect sizes (simplified)
        d_v = b  # Vacancy size
        d_i = b  # Interstitial size
        d_2i = 2*b  # Di-interstitial size
        d_3i = 3*b  # Tri-interstitial size
        d_iL = 2 * self.calculate_r_iL(y)  # Loop diameter
        d_aiL = 2 * self.calculate_r_aiL(y)  # Aligned loop diameter
        d_vL = 2 * self.calculate_r_vL(y)  # Vacancy loop diameter
        d_avL = 2 * self.calculate_r_avL(y)  # Aligned vacancy loop diameter
        
        # Short-range hardening contributions (Equation 74)
        Delta_tau_v = alpha_v * mu * b * np.sqrt(N_v * d_v) if N_v * d_v > 0 else 0
        Delta_tau_i = alpha_i * mu * b * np.sqrt(N_i * d_i) if N_i * d_i > 0 else 0
        Delta_tau_2i = alpha_i * mu * b * np.sqrt(N_2i * d_2i) if N_2i * d_2i > 0 else 0
        Delta_tau_3i = alpha_i * mu * b * np.sqrt(N_3i * d_3i) if N_3i * d_3i > 0 else 0
        Delta_tau_iL = alpha_iL * mu * b * np.sqrt(N_iL * d_iL) if N_iL * d_iL > 0 else 0
        Delta_tau_aiL = alpha_aiL * mu * b * np.sqrt(N_aiL * d_aiL) if N_aiL * d_aiL > 0 else 0
        Delta_tau_vL = alpha_vL * mu * b * np.sqrt(N_vL * d_vL) if N_vL * d_vL > 0 else 0
        Delta_tau_avL = alpha_avL * mu * b * np.sqrt(N_avL * d_avL) if N_avL * d_avL > 0 else 0
        
        # Total short-range hardening (Equation 75)
        Delta_tau_SR = np.sqrt(Delta_tau_v**2 + Delta_tau_i**2 + Delta_tau_2i**2 + Delta_tau_3i**2 +
                              Delta_tau_iL**2 + Delta_tau_aiL**2 + Delta_tau_vL**2 + Delta_tau_avL**2)
        
        # Long-range hardening from network dislocations (Equation 76)
        rho_N = self.input_data.material_params['rho']
        alpha_LR = 0.3  # Long-range interaction strength
        Delta_tau_LR = alpha_LR * mu * b * np.sqrt(rho_N)
        
        # Total shear stress increase
        Delta_tau = Delta_tau_SR + Delta_tau_LR
        
        # Convert to yield strength increase (Equation 77)
        Delta_sigma_y = Delta_tau / M
        
        return Delta_sigma_y
    
    def calculate_dislocation_densities(self, y):
        """
        Calculate total dislocation densities (Equation 26)
        """
        y = np.maximum(y, self.C_floor)

        CiL, CaiL, CvL, CavL = y[4], y[5], y[6], y[7]
        
        # Network dislocation density
        rho_N = self.input_data.material_params['rho']
        
        # Loop radii
        if self.rate_equations is not None:
            r_iL = self.rate_equations.calculate_r_iL(y)
            r_aiL = self.rate_equations.calculate_r_aiL(y)
            r_vL = self.rate_equations.calculate_r_vL(y)
            r_avL = self.rate_equations.calculate_r_avL(y)
        else:
            r_iL = 5e-9   # fallback values
            r_aiL = 5e-9
            r_vL = 5e-9
            r_avL = 5e-9

        # Loop dislocation densities
        Omega = self.input_data.physical_props['Omega']
        rho_iL = 2 * np.pi * r_iL * CiL / Omega if CiL > 0 else 0
        rho_aiL = 2 * np.pi * r_aiL * CaiL / Omega if CaiL > 0 else 0
        rho_vL = 2 * np.pi * r_vL * CvL / Omega if CvL > 0 else 0
        rho_avL = 2 * np.pi * r_avL * CavL / Omega if CavL > 0 else 0
        
        # Total dislocation density
        rho_tot = rho_N + rho_iL + rho_aiL + rho_vL + rho_avL
        
        return {
            'rho_total': rho_tot,
            'rho_network': rho_N,
            'rho_iL': rho_iL,
            'rho_aiL': rho_aiL,
            'rho_vL': rho_vL,
            'rho_avL': rho_avL
        }
    
    def get_concentration_by_name(self, y, name):
        """
        Get concentration by name from the y array
        """
        if name in self.concentration_names:
            index = self.concentration_names.index(name)
            return y[index]
        else:
            raise ValueError(f"Unknown concentration name: {name}")
    
    def set_concentration_by_name(self, y, name, value):
        """
        Set concentration by name in the y array
        """
        if name in self.concentration_names:
            index = self.concentration_names.index(name)
            y[index] = value
        else:
            raise ValueError(f"Unknown concentration name: {name}")
    
    def print_current_state(self, t, y):
        """
        Print current state for debugging
        """
        print(f"\n=== Time: {t:.2e} s ===")
        for i, name in enumerate(self.concentration_names):
            print(f"{name}: {y[i]:.2e}")
        
        # Print derived quantities
        print(f"r_iL: {self.calculate_r_iL(y)*1e9:.2f} nm")
        print(f"r_vL: {self.calculate_r_vL(y)*1e9:.2f} nm")
        print(f"r_aiL: {self.calculate_r_aiL(y)*1e9:.2f} nm")
        print(f"r_avL: {self.calculate_r_avL(y)*1e9:.2f} nm")
        
        hardening = self.calculate_hardening(y)
        print(f"Hardening: {hardening/1e6:.2f} MPa")
    
    def validate_concentrations(self, y):
        """
        Validate that concentrations are physical
        """
        # Check for negative concentrations
        negative_indices = np.where(y < 0)[0]
        if len(negative_indices) > 0:
            print(f"Warning: Negative concentrations found at indices: {negative_indices}")
            for idx in negative_indices:
                print(f"  {self.concentration_names[idx]}: {y[idx]:.2e}")
            return False
        
        # Check for extremely large concentrations (might indicate numerical issues)
        large_indices = np.where(y > 1.0)[0]  # Concentrations > 1 (100%) are unphysical
        if len(large_indices) > 0:
            print(f"Warning: Very large concentrations found at indices: {large_indices}")
            for idx in large_indices:
                print(f"  {self.concentration_names[idx]}: {y[idx]:.2e}")
        
        return True
    
    def get_initial_conditions(self, custom_values=None):
        """
        Generate physically reasonable initial conditions
        """
        y0 = np.zeros(self.n_equations)
        
        # Default initial conditions
        C_v_eq = self.input_data.derived['C_v_eq']
        scale1 = 1e-6
        scale2 = 1e-10
        defaults = {
            'Cv': C_v_eq,  # Thermal equilibrium
            'Ci': scale1*C_v_eq,  # Small initial interstitial concentration
            'C2i': scale2*C_v_eq,  # Very small cluster concentrations
            'C3i': scale1*C_v_eq,
            'CiL': scale2*C_v_eq,  # Small initial loop concentrations
            'CaiL': scale2*C_v_eq,
            'CvL': scale1*C_v_eq,
            'CavL': scale1*C_v_eq,
            'CiL_i': scale1*C_v_eq,  # Very small initial defect content in loops
            'CaiL_i': scale1*C_v_eq,
            'CvL_v': scale1*C_v_eq,
            'CavL_v': scale2*C_v_eq
        }
        
        # Apply custom values if provided
        if custom_values:
            defaults.update(custom_values)
        
        # Set initial conditions
        for name, value in defaults.items():
            self.set_concentration_by_name(y0, name, value)

        # Evolving network dislocation density starts at the grown-in,
        # temperature-dependent material value (accumulators 12..17 start at 0).
        y0[self.idx_rhoN] = self.input_data.material_params['rho']

        return y0