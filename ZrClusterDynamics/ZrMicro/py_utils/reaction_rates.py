# reaction_rates.py
import numpy as np
class ReactionRates:
    """
    Class to calculate all reaction rates for the rate equations system
    Contains all the frequency calculations and reaction rate member functions
    """
    
    def __init__(self, input_data, rate_equations=None):
        """
        Initialize with input data and optional rate_equations reference
        """
        self.input_data = input_data
        self.rate_equations = rate_equations  # Store reference
        self.k_B = 8.617e-5  # Boltzmann constant [eV/K]
        self.T = input_data.material_params['T']
        
        # Calculate basic frequencies
        self.calculate_basic_frequencies()
        
        # Calculate thermal emission probabilities
        self.calculate_thermal_emissions()
        
        # Initialize current state
        self.current_concentrations = None
        self.current_time = 0.0

        # Initialize cached quantities (updated in update_state)
        self.C_v_s = 0.0
        self.C_i_s = 0.0
        self._flux_v = 0.0
        self._flux_i = 0.0
    
    def calculate_basic_frequencies(self):
        """
        Calculate basic frequencies (Section 2.1)
        """
        z_c = self.input_data.physical_props['z_c']
        nu_i = self.input_data.physical_props['nu_i']
        nu_v = self.input_data.physical_props['nu_v']
        E_m_i = self.input_data.physical_props['E_m_i']
        E_m_v = self.input_data.physical_props['E_m_v']
        E_m_2i = self.input_data.physical_props['E_m_2i']
        
        # Basic frequencies (Equations 1-4)
        self.omega_i = z_c * nu_i * np.exp(-E_m_i / (self.k_B * self.T))
        self.omega_2i = z_c * nu_i * np.exp(-E_m_2i / (self.k_B * self.T))
        self.omega_3i = self.omega_2i  # Assuming same as di-interstitials
        self.omega_v = z_c * nu_v * np.exp(-E_m_v / (self.k_B * self.T))
        
        # Irradiation frequency
        G = self.input_data.material_params['G']
        self.omega_irr = G  # Simplified - could include b_r factor
        
    def calculate_thermal_emissions(self):
        """
        Calculate thermal emission probabilities (Section 2.2)
        """
        E_F_v = self.input_data.physical_props['E_F_v']
        
        # Basic thermal emission (Equation 12)
        self.e_v = np.exp(-E_F_v / (self.k_B * self.T))
        E_B_2i = self.input_data.physical_props['E_b_2i']
        E_B_3i = self.input_data.physical_props['E_b_3i']
        
        self.e_2i = np.exp(-E_B_2i / (self.k_B * self.T))
        self.e_3i = np.exp(-E_B_3i / (self.k_B * self.T))  
     
        # Loop emission probabilities (will be updated based on current sizes)
        # self.e_v_iL = np.exp(-E_B_v_iL / (self.k_B * self.T))
        # self.e_v_vL = np.exp(-E_B_v_vL / (self.k_B * self.T))
        # self.e_v_avL = self.e_v_vL  # Will be updated with stress effects
        
        self.e_v_iL = np.exp(-5 / (self.k_B * self.T)) #approx for now
        self.e_v_vL = np.exp(-5 / (self.k_B * self.T))
        self.e_v_avL = self.e_v_vL  # Will be updated with stress effects

        
    def update_state(self, concentrations, time):
        """
        Update current state for calculations
        """
        self.current_concentrations = concentrations
        self.current_time = time

        # Update stress-dependent emissions if needed
        self.update_stress_effects()

        # Cache sink concentrations and fluxes to avoid redundant recomputation
        # within a single ODE evaluation
        self.C_v_s, self.C_i_s = self.calculate_sink_concentrations(concentrations)
        conc = np.maximum(concentrations, 0)
        self._flux_v = self.omega_v * conc[0]
        self._flux_i = self.omega_i * (conc[1] + 2*conc[2] + 3*conc[3])
        
    def update_stress_effects(self):
        """
        Update stress-dependent thermal emissions
        """
        
        sigma_h = self.input_data.material_params.get('sigma_h', 0.0)  # Pa
        Omega = self.input_data.physical_props['Omega']

        stress_factor = np.exp(sigma_h * Omega / (self.k_B * self.T * 1.602e-19))  # Convert eV to J

        self.e_sigma_v_iL = stress_factor * self.e_v_iL
        self.e_sigma_v_vL = stress_factor * self.e_v_vL
        self.e_sigma_v_avL = stress_factor * self.e_v_avL
    
    def calculate_diffusion_coefficients(self):
        """
        Calculate diffusion coefficients (Equations 23-24)
        """
        a = self.input_data.physical_props['a']
        z_c = self.input_data.physical_props['z_c']
        
        self.D_v = (a**2 / z_c) * self.omega_v
        self.D_i = (a**2 / z_c) * self.omega_i
        
        return self.D_v, self.D_i
    
    def calculate_sink_concentrations(self, concentrations):
        """
        Calculate equivalent sink concentrations (Equations 27-31).

        NETWORK-ONLY: loop absorption is handled explicitly by the
        mass-conserving loop-growth coupling (see loop_absorption() and
        RateEquations.dCi_dt/dCv_dt), so loops must NOT also appear in the
        point-defect sink strength or their absorption would be double counted.
        """
        if concentrations is None:
            concentrations = self.current_concentrations

        # Evolving network dislocation density (grows as loops are absorbed into
        # the network); falls back to the grown-in seed for legacy state vectors.
        rho_N = self.current_rho_N(concentrations)
        Z_i   = self.input_data.model_params.get('Z_N', 1.05)
        a     = self.input_data.physical_props['a']
        z_c   = self.input_data.physical_props['z_c']

        # Vacancy sink concentration (Equation 28) — network dislocations only
        self.C_v_s = (a**2 / z_c) * rho_N

        # Interstitial sink concentration (Equation 30) — network dislocations only
        self.C_i_s = (a**2 / z_c) * Z_i * rho_N

        return self.C_v_s, self.C_i_s

    def a_shrink_gate(self, C_num, C_cont):
        """
        Smooth minimum-stable-size gate for a-loops (iL, aiL).

        a-loops shrink by absorbing free vacancies (the vacancy annihilates a
        stored interstitial). At high T / high dose the vacancy flux can exceed
        the interstitial flux, so the net climb flips negative and the stored
        content drains to the floor — the reconstructed mean diameter
        r = l_a*sqrt(C_cont/C_num) then collapses to zero (unphysical: the loop
        count stays high while each loop shrinks to nothing).

        This gate g(r) in [0, 1] suppresses the vacancy-absorption channel as the
        radius approaches the minimum stable a-loop radius r_min_a, so loops stop
        shrinking there:  g -> 0 as r -> r_min_a,  g -> 1 for r >> r_min_a.
        Gating the WHOLE vacancy-absorption term (not just the content term) keeps
        the point-defect balance closed — the un-absorbed vacancies remain in the
        free pool instead of silently annihilating stored interstitials. A
        C1-continuous smoothstep avoids adding solver stiffness.
        """
        r_min = self.input_data.model_params.get('r_min_a', 1.0e-9)
        if r_min <= 0.0:
            return 1.0                      # gate disabled
        w     = self.input_data.model_params.get('w_rmin', 0.3)
        l_a   = self.input_data.derived['l_a']
        C_floor = self.input_data.model_params.get('C_floor', 1e-20)
        r = l_a * np.sqrt(max(C_cont, 0.0) / max(C_num, C_floor))
        x = (r / r_min - 1.0) / w
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        return x * x * (3.0 - 2.0 * x)

    def loop_absorption(self, concentrations=None):
        """
        Decomposed loop absorption fluxes for the mass-conserving coupling.

        Each loop grows by absorbing its own defect type and shrinks by absorbing
        the opposite type (which annihilates a stored defect — a defect-loop
        recombination event). Returns (loop_abs_i, loop_abs_v, loop_recomb), all
        >= 0, where:
            loop_abs_i  – interstitials removed from the free pool by all loops
            loop_abs_v  – vacancies removed from the free pool by all loops
            loop_recomb – defect-loop recombination events (1 i and 1 v each)

        The net loop-content growth equals
            growth_iL = iL_i - iL_v,  growth_vL = vL_v - vL_i,  etc.
        i.e. exactly the loop_growth_rate_* methods, so loop content is unchanged.
        """
        if concentrations is None:
            concentrations = self.current_concentrations
        c = np.maximum(concentrations, 0.0)

        CiL, CaiL, CvL, CavL = c[4], c[5], c[6], c[7]
        CiL_i, CaiL_i, CvL_v, CavL_v = c[8], c[9], c[10], c[11]

        l_c  = self.input_data.derived['l_c']
        l    = self.input_data.derived['l']
        Q    = self.input_data.model_params['Q']
        # Orientation- and species-resolved DAD capture efficiencies
        # (see InputData.calculate_derived_parameters).
        Z_i_a = self.input_data.derived['Z_i_a']   # a-loops capture interstitials
        Z_v_a = self.input_data.derived['Z_v_a']   # a-loops capture vacancies
        Z_i_c = self.input_data.derived['Z_i_c']   # c-loops capture interstitials
        Z_v_c = self.input_data.derived['Z_v_c']   # c-loops capture vacancies

        fi = self.flux_i()
        fv = self.flux_v()
        lc_l = l_c / l

        pref_iL  = lc_l       * np.sqrt(CiL_i  * CiL)
        pref_aiL = lc_l       * np.sqrt(CaiL_i * CaiL)
        pref_vL  = lc_l * Q   * np.sqrt(CvL    * CvL_v)
        pref_avL = lc_l * Q   * np.sqrt(CavL   * CavL_v)

        iL_i,  iL_v  = pref_iL  * Z_i_a * fi, pref_iL  * Z_v_a * fv
        aiL_i, aiL_v = pref_aiL * Z_i_a * fi, pref_aiL * Z_v_a * fv
        vL_v,  vL_i  = pref_vL  * Z_v_c * fv, pref_vL  * Z_i_c * fi
        avL_v, avL_i = pref_avL * Z_v_c * fv, pref_avL * Z_i_c * fi

        # Minimum-stable-size gate on the a-loop vacancy-absorption channel: as a
        # loop approaches r_min_a it stops absorbing vacancies, so it can no
        # longer shrink to zero. Applied identically here and in
        # loop_growth_rate_iL/aiL so growth and absorption stay consistent.
        iL_v  *= self.a_shrink_gate(CiL,  CiL_i)
        aiL_v *= self.a_shrink_gate(CaiL, CaiL_i)

        loop_abs_i  = iL_i + aiL_i + vL_i + avL_i
        loop_abs_v  = vL_v + avL_v + iL_v + aiL_v
        loop_recomb = iL_v + aiL_v + vL_i + avL_i
        return loop_abs_i, loop_abs_v, loop_recomb
    
    # Reaction rate member functions
    def R_i_v(self, concentrations=None):
        """Reaction rate between interstitials and vacancies"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations=np.maximum(concentrations, 0e-20)
        recom = self.input_data.physical_props['recom']
        # print(f"recom: {recom}")
        Ci, Cv = concentrations[1], concentrations[0]

        omega_iv= recom *(self.omega_i + self.omega_v)
        return omega_iv * Ci * Cv

    def R_2i_v(self, concentrations=None):
        """Reaction rate between di-interstitials and vacancies"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations=np.maximum(concentrations, 0e-20)
        
        C2i, Cv = concentrations[2], concentrations[0]
        omega_2iv= self.omega_2i + self.omega_v

        return omega_2iv * C2i * Cv

    def R_3i_v(self, concentrations=None):
        """Reaction rate between tri-interstitials and vacancies"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations=np.maximum(concentrations, 0e-20)
        
        C3i, Cv = concentrations[3], concentrations[0]
        omega_3iv= self.omega_3i + self.omega_v

        return omega_3iv * C3i * Cv

    def R_i_i(self, concentrations=None):
        """Reaction rate between interstitials"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations=np.maximum(concentrations, 0e-20)
        
        Ci = concentrations[1]
        
        return 2*self.omega_i * Ci**2
    
    def R_i_2i(self, concentrations=None):
        """Reaction rate between interstitials and di-interstitials"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations=np.maximum(concentrations, 0e-20)
        
        Ci, C2i = concentrations[1], concentrations[2]
        omega_i2i= self.omega_i + self.omega_2i
        
        return  omega_i2i * Ci * C2i

    def R_i_3i(self, concentrations=None):
        """Reaction rate between interstitials and tri-interstitials"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations=np.maximum(concentrations, 0e-20)
        
        Ci, C3i = concentrations[1], concentrations[3]
        omega_i3i= self.omega_i + self.omega_3i
        
        return omega_i3i * Ci * C3i

    def R_2i_2i(self, concentrations=None):
        """Reaction rate between di-interstitials"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations=np.maximum(concentrations, 0e-20)
        
        C2i = concentrations[2]
        omega_2i2i= self.omega_2i + self.omega_2i
        
        return 2*omega_2i2i * C2i**2

    def R_2i_3i(self, concentrations=None):
        """Reaction rate between di-interstitials"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations=np.maximum(concentrations, 0e-20)
        
        C2i = concentrations[2]
        C3i = concentrations[3]
        omega_2i3i= self.omega_2i + self.omega_3i
        
        return  omega_2i3i * C2i*C3i


    def R_v_s(self, concentrations=None):
        """Reaction rate between vacancies and sinks"""
        if concentrations is None:
            concentrations = self.current_concentrations
        Cv = max(concentrations[0], 0)
        return self.omega_v * Cv * self.C_v_s

    def R_i_s(self, concentrations=None):
        """Reaction rate between interstitials and sinks"""
        if concentrations is None:
            concentrations = self.current_concentrations
        Ci = max(concentrations[1], 0)
        return self.omega_i * Ci * self.C_i_s

    def R_2i_s(self, concentrations=None):
        """Reaction rate between di-interstitials and sinks"""
        if concentrations is None:
            concentrations = self.current_concentrations
        C2i = max(concentrations[2], 0)
        return self.omega_i * C2i * self.C_i_s

    def R_3i_s(self, concentrations=None):
        """Reaction rate between tri-interstitials and sinks"""
        if concentrations is None:
            concentrations = self.current_concentrations
        C3i = max(concentrations[3], 0)
        return self.omega_i * C3i * self.C_i_s
    
    # Generation rates (simplified notation from Section 3.1)
    def G_v(self):
        """Vacancy generation rate"""
        return self.input_data.derived['G_v']
    
    def G_i(self):
        """Interstitial generation rate"""
        return self.input_data.derived['G_i']
    
    def G_2i(self):
        """Di-interstitial generation rate"""
        return self.input_data.derived['G_2i']/2  # Divided by 2 for pairs
    
    def G_3i(self):
        """Tri-interstitial generation rate"""
        return self.input_data.derived['G_3i']/3  # Divided by 3 for pairs

    def G_iL(self):
        """Non-aligned cascade interstitial (a-)loop ATOM generation rate"""
        return self.input_data.derived['G_iL']

    def G_aiL(self):
        """Aligned cascade interstitial (a-)loop ATOM generation rate"""
        return self.input_data.derived['G_aiL']

    def G_vL(self):
        """Non-aligned vacancy loop generation rate"""
        return self.input_data.derived['G_vL']

    def G_avL(self):
        """Aligned vacancy loop generation rate"""
        return self.input_data.derived['G_avL']
    
    # Thermal emission fluxes
    def phi_v_emission_2i(self):
        """Vacancy emission flux from di-interstitials"""
        return self.omega_v * (self.e_2i - self.e_v)
    
    def phi_v_emission_3i(self):
        """Vacancy emission flux from tri-interstitials"""
        return self.omega_v * (self.e_3i - self.e_v)
    
    def phi_v_emission_iL(self, concentrations=None):
        """Vacancy emission flux from interstitial loops"""
        return self.omega_v * (self.e_v_iL - self.e_v)
    
    def phi_v_emission_vL(self, concentrations=None):
        """Vacancy emission flux from vacancy loops"""
        return self.omega_v * (self.e_v_vL - self.e_v)
    
    def phi_v_emission_avL(self, concentrations=None):
        """Vacancy emission flux from aligned vacancy loops"""
        return self.omega_v * (self.e_sigma_v_avL - self.e_v)
    
    def emission_2i(self, concentrations=None):
        """Thermal emission rate from di-interstitials"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations=np.maximum(concentrations, 0e-20)
        
        C2i = concentrations[2]
        return self.omega_2i * self.e_2i * C2i
    
    def emission_3i(self, concentrations=None):
        """Thermal emission rate from tri-interstitials"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations=np.maximum(concentrations, 0e-20)
        
        C3i = concentrations[3]
        return self.omega_3i * self.e_3i * C3i  

    # Vacancy and interstitial fluxes
    def flux_v(self, concentrations=None):
        """Vacancy flux to sinks"""
        if concentrations is None:
            return self._flux_v
        concentrations = np.maximum(concentrations, 0)
        return self.omega_v * concentrations[0]

    def flux_i(self, concentrations=None):
        """Interstitial flux to sinks"""
        if concentrations is None:
            return self._flux_i
        concentrations = np.maximum(concentrations, 0)
        Ci, C2i, C3i = concentrations[1], concentrations[2], concentrations[3]
        return self.omega_i * (Ci + 2*C2i + 3*C3i)


    # Loop growth rates with bias factors
    def loop_growth_rate_iL(self, concentrations=None):
        """Growth rate for interstitial loops"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations=np.maximum(concentrations, 0e-20)

        CiL = concentrations[4]
        CiL_i = concentrations[8]
        l_c = self.input_data.derived['l_c']  # Characteristic length scale
        l = self.input_data.derived['l']      # Current length scale

        # DAD bias factors (a-type interstitial loop)
        Z_i_a = self.input_data.derived['Z_i_a']
        Z_v_a = self.input_data.derived['Z_v_a']

        g = self.a_shrink_gate(CiL, CiL_i)   # halt shrinkage at r_min_a
        rate = (l_c/l)*np.sqrt(CiL_i*CiL)*(Z_i_a * self.flux_i() - g * Z_v_a * self.flux_v())
        return rate

    def loop_growth_rate_aiL(self, concentrations=None):
        """Growth rate for aligned interstitial loops"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations=np.maximum(concentrations, 0e-20)
       
        CaiL = concentrations[5]
        CaiL_i = concentrations[9]
        l_c = self.input_data.derived['l_c']  # Characteristic length scale
        l = self.input_data.derived['l']      # Current length scale

        # DAD bias factors (a-type interstitial loop)
        Z_i_a = self.input_data.derived['Z_i_a']
        Z_v_a = self.input_data.derived['Z_v_a']

        g = self.a_shrink_gate(CaiL, CaiL_i)   # halt shrinkage at r_min_a
        rate = (l_c/l)*np.sqrt(CaiL_i*CaiL)*(Z_i_a * self.flux_i() - g * Z_v_a * self.flux_v())
        return rate
    
    def loop_growth_rate_vL(self, concentrations=None):
        """Growth rate for vacancy loops"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations=np.maximum(concentrations, 1e-20)
    
        CvL = concentrations[6]
        CvL_v = concentrations[10]

        # DAD bias factors (c-type vacancy loop) and absorption efficiency
        Z_i_c = self.input_data.derived['Z_i_c']
        Z_v_c = self.input_data.derived['Z_v_c']

        Q = self.input_data.model_params['Q']
        l_c = self.input_data.derived['l_c']  # Characteristic length scale
        l = self.input_data.derived['l']      # Current length scale

        rate = (l_c/l)*Q*np.sqrt(CvL*CvL_v)*(Z_v_c * self.flux_v() - Z_i_c * self.flux_i())
        return rate

    def loop_growth_rate_avL(self, concentrations=None):
        """Growth rate for aligned vacancy loops"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations = np.maximum(concentrations, 1e-20)

        CavL = concentrations[7]
        CavL_v = concentrations[11]

        l_c = self.input_data.derived['l_c']  # Characteristic length scale
        l = self.input_data.derived['l']      # Current length scale

        # DAD bias factors (c-type vacancy loop) and absorption efficiency
        Z_i_c = self.input_data.derived['Z_i_c']
        Z_v_c = self.input_data.derived['Z_v_c']
        Q = self.input_data.model_params['Q']

        rate = (l_c/l)*Q*np.sqrt(CavL*CavL_v)*(Z_v_c * self.flux_v() - Z_i_c * self.flux_i())
        return rate
    
    # Loop nucleation rates
    def nucleation_rate_iL(self, concentrations=None):
        """Nucleation rate for interstitial loops"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations = np.maximum(concentrations, 0)
        # Non-aligned fraction f_na = 1 - f_a (was erroneously 1 - derived['f_na']
        # = f_a, which made both loop fractions f_a instead of summing to 1).
        f_na = self.input_data.derived['f_na']

        # Two a-loop nucleation sources:
        #  (1) homogeneous clustering from i+3i and 2i+2i reactions, and
        #  (2) cascade source G_iL/n_iL_nuc (each cascade loop carries n_iL_nuc
        #      interstitials, seeded as content in dCiL_i_dt). G_iL already
        #      carries the non-aligned (1-f_a) split, so it is NOT re-scaled by f_na.
        n_iL_nuc = self.input_data.model_params.get('n_iL_nuc', 20.0)
        rate = (f_na * (self.R_i_3i(concentrations) + self.R_2i_2i(concentrations))
                + self.G_iL() / n_iL_nuc)

        return rate

    def nucleation_rate_aiL(self, concentrations=None):
        """Nucleation rate for aligned interstitial loops"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations = np.maximum(concentrations, 0)

        f_a = self.input_data.derived['f_a']

        # Clustering source (f_a fraction) + cascade source G_aiL/n_iL_nuc
        # (G_aiL already carries the aligned f_a split).
        n_iL_nuc = self.input_data.model_params.get('n_iL_nuc', 20.0)
        rate = (f_a * (self.R_i_3i(concentrations) + self.R_2i_2i(concentrations))
                + self.G_aiL() / n_iL_nuc)
        return rate
    
    def nucleation_rate_vL(self, concentrations=None):
        """Nucleation rate for vacancy loops"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations=np.maximum(concentrations, 0e-20)

        # Constant cascade source; vacancy-loop density is instead saturated by
        # the T-dependent annealing term CvL/tau_vL in the rate equations
        # (mirrors rate_equations.cpp). The capture-volume factor (1 - n_cap*CvL)
        # is intentionally removed.
        # G_vL is the cascade vacancy ATOM rate; the loop NUMBER rate is
        # G_vL/n_vL_nuc (each nucleated loop carries n_vL_nuc vacancies, seeded
        # as loop content in dCvL_v_dt). See InputData 'n_vL_nuc'.
        n_vL_nuc = self.input_data.model_params.get('n_vL_nuc', 50.0)
        rate = self.input_data.derived['G_vL'] / n_vL_nuc

        return rate
    
    def nucleation_rate_avL(self, concentrations=None):
        """Nucleation rate for aligned vacancy loops"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations=np.maximum(concentrations, 0e-20)

        # Constant cascade source; aligned vacancy-loop density is instead
        # saturated by the T-dependent annealing term CavL/tau_avL in the rate
        # equations (mirrors rate_equations.cpp). The capture-volume factor
        # (1 - n_cap*CavL) is intentionally removed.
        # G_avL is the cascade vacancy ATOM rate; loop NUMBER rate = G_avL/n_vL_nuc.
        n_vL_nuc = self.input_data.model_params.get('n_vL_nuc', 50.0)
        rate = self.input_data.derived['G_avL'] / n_vL_nuc

        return rate
    
    def nucleation_content_i(self, concentrations=None):
        """
        Interstitial atoms deposited into loop content by the nucleation/loop
        reactions, equal to exactly the atoms removed from Ci/C2i/C3i by the
        R_i_3i (1+3), R_2i_2i (2) and R_2i_3i (3) reactions. Splitting this by the
        loop fractions (f_na, f_a) and adding it to dCiL_i/dCaiL_i makes loop
        nucleation mass-conserving (mirrors rate_equations.cpp).
        """
        if concentrations is None:
            concentrations = self.current_concentrations
        return (4.0 * self.R_i_3i(concentrations)
                + 2.0 * self.R_2i_2i(concentrations)
                + 3.0 * self.R_2i_3i(concentrations))

    # Thermal annealing rates
    def annealing_rate_vL(self, concentrations=None):
        """Thermal annealing rate for vacancy loops"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations=np.maximum(concentrations, 0e-20)

        CvL = concentrations[6]
        tau_vL = self.input_data.derived['tau_vL']  # = tau_vL0 * exp(E_a_vL / kT)

        return CvL / tau_vL
    
    def annealing_rate_avL(self, concentrations=None):
        """Thermal annealing rate for aligned vacancy loops"""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations=np.maximum(concentrations, 0e-20)

        CavL = concentrations[7]
        # Aligned vacancy loops share the non-aligned T-dependent lifetime law
        # (derived['tau_avL'] = tau_avL0*exp(E_a_avL/kT), defaulting to tau_vL).
        # The legacy constant model_params['tau_avL'] override is gone.
        tau_avL = self.input_data.derived['tau_avL']

        return CavL / tau_avL

    def annealing_content_vL(self, concentrations=None):
        """Vacancy content released to the free pool by loop annealing.

        The lifetime tau_vL represents dissolution of newly-nucleated, still-
        small EMBRYO loops; loops that survive and grow are stable. So each
        annealed loop releases its BIRTH content (~n_vL_nuc vacancies), not the
        mean: content release = n_vL_nuc * (number annealing rate CvL/tau_vL).
        At number saturation (G_vL/n_vL_nuc = CvL/tau_vL) this exactly balances
        the content seed G_vL, leaving net content growth = flux only, so the
        surviving loops accumulate vacancies by flux and the MEAN SIZE grows
        with dose (instead of being capped by a mean-content anneal term).
        Internal transfer (loop content -> free Cv): conserves vacancy atoms."""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations = np.maximum(concentrations, 0e-20)
        CvL = concentrations[6]
        tau_vL = self.input_data.derived['tau_vL']
        n_vL_nuc = self.input_data.model_params.get('n_vL_nuc', 50.0)
        return n_vL_nuc * CvL / tau_vL

    def annealing_content_avL(self, concentrations=None):
        """Vacancy content released from annealing aligned vacancy loops
        (birth content of dissolving embryos; see annealing_content_vL)."""
        if concentrations is None:
            concentrations = self.current_concentrations
        concentrations = np.maximum(concentrations, 0e-20)
        CavL = concentrations[7]
        tau_avL = self.input_data.derived['tau_avL']
        n_vL_nuc = self.input_data.model_params.get('n_vL_nuc', 50.0)
        return n_vL_nuc * CavL / tau_avL

    # ── Evolving network dislocation density (state index idx_rhoN) ──────────
    def current_rho_N(self, concentrations=None):
        """Current network dislocation density [m^-2].

        rho_N is now a state variable (see RateEquations.idx_rhoN): it starts at
        the grown-in, temperature-dependent material value and increases as loops
        are absorbed into the network (loop-network coalescence). This reads the
        evolving value out of the state vector, falling back to the constant
        grown-in seed when the state does not carry the rho_N row (e.g. a legacy
        12/18-component vector or a 2-D post-processing array).
        """
        seed = self.input_data.material_params['rho']
        if concentrations is None:
            concentrations = self.current_concentrations
        if concentrations is None or self.rate_equations is None:
            return seed
        idx = getattr(self.rate_equations, 'idx_rhoN', None)
        arr = np.asarray(concentrations)
        if idx is None or arr.ndim != 1 or arr.shape[0] <= idx:
            return seed
        val = arr[idx]
        return val if val > 0.0 else seed

    # ── Geometric loop coalescence (absorbed-flux–climb driven) ──────────────
    def coalescence_rates(self, concentrations=None):
        """
        Geometric loop coalescence, with the rate scale tied to the loop climb
        velocity driven by the ABSORBED (gain-side) point-defect flux — a-loops
        climb by absorbing interstitials, c-loops by absorbing vacancies. Unlike
        the previous formulation (tied to the NET growth velocity dr/dt, which
        collapses to ~0 once loop growth saturates and froze both N and d), the
        absorbed flux is set by the displacement rate and stays positive at
        steady state, so coalescence persists: the number density keeps falling
        gradually and — because the like-loop channel conserves content — the
        mean loop size keeps rising gradually.

        Two channels per loop population (iL, aiL, vL, avL):
          (1) like-loop coarsening — two loops merge into one: the NUMBER density
              drops while the stored CONTENT is conserved (point-defect neutral;
              the surviving loop simply becomes larger -> mean size grows).
          (2) loop-network coalescence — the loop is absorbed by the network:
              number AND content are removed at the same fractional rate; the
              removed content is booked by the caller as network sink absorption
              (cum_sink_i / cum_sink_v) so the point-defect balance stays closed,
              and the absorbed loop line length feeds the rho_N evolution
              (see RateEquations.drho_N_dt / rho_N_source below).

        Smooth Poisson/Avrami overlap probabilities gate the onset using the loop
        size r, the loop-loop distance N^(-1/3) and the network distance
        rho_N^(-1/2):
          phi_LL = 1 - exp(-kappa_LL * (4/3)*pi*r^3 * N)
          phi_LN = 1 - exp(-kappa_LN *       pi*r^2 * rho_N)
        with rate scales set by the absorbed-flux climb speed v_abs (>= 0):
          v_abs  = l_scale * (l_c/l) * [Q] * Z_gain * flux_gain / 2   [m/s]
          nu_LL  = c_LL * v_abs * N^(1/3)        [s^-1]
          nu_LN  = c_LN * v_abs * sqrt(rho_N)    [s^-1]

        Returns a dict keyed by population ('iL','aiL','vL','avL'); each value is
        {'num', 'content', 'num_LN', 'r'}:
          num     – total number-density loss rate (channels 1+2), >= 0
          content – content loss rate (channel 2 only; channel 1 conserves it)
          num_LN  – number-density loss to the network (channel 2 only); used to
                    grow rho_N (each absorbed loop adds 2*pi*r of line length)
          r       – mean loop radius [m]
        """
        if concentrations is None:
            concentrations = self.current_concentrations
        C_floor = self.input_data.model_params.get('C_floor', 1e-20)
        c = np.maximum(concentrations, C_floor)

        Omega = self.input_data.physical_props['Omega']
        rho_N = self.current_rho_N(concentrations)   # evolving network density
        l_a   = self.input_data.derived['l_a']
        l_c   = self.input_data.derived['l_c']
        l     = self.input_data.derived['l']
        Q     = self.input_data.model_params['Q']
        Z_i_a = self.input_data.derived['Z_i_a']   # a-loops capture interstitials
        Z_v_c = self.input_data.derived['Z_v_c']   # c-loops capture vacancies

        kappa_LL = self.input_data.model_params.get('kappa_LL', 1.0)
        kappa_LN = self.input_data.model_params.get('kappa_LN', 1.0)
        # Orientation-resolved coalescence rate-scale coefficients. SEPARATE
        # a-loop (iL,aiL) and c-loop (vL,avL) coefficients let the c-component
        # vacancy loops coarsen at a different pace than the a-component
        # interstitial loops while keeping each family independently calibrated.
        c_LL     = self.input_data.model_params.get('c_LL', 1.0)
        c_LN     = self.input_data.model_params.get('c_LN', 1.0)
        c_LL_a   = self.input_data.model_params.get('c_LL_a', c_LL)
        c_LN_a   = self.input_data.model_params.get('c_LN_a', c_LN)
        c_LL_c   = self.input_data.model_params.get('c_LL_c', c_LL)
        c_LN_c   = self.input_data.model_params.get('c_LN_c', c_LN)

        # Absorbed-flux climb speeds [m/s] (persistent; do NOT vanish when net
        # loop growth saturates).  Derived exactly as the old dr/dt prefactor but
        # using the gain-side flux (Z_i_a*flux_i for a-loops, Z_v_c*flux_v for
        # c-loops) instead of the net (gain - loss) bias flux.
        fi, fv = self.flux_i(concentrations), self.flux_v(concentrations)
        lc_l = l_c / l
        v_abs_a = max(l_a * lc_l       * Z_i_a * fi / 2.0, 0.0)
        v_abs_c = max(l_c * lc_l * Q   * Z_v_c * fv / 2.0, 0.0)

        FOUR_THIRDS_PI = (4.0 / 3.0) * np.pi
        sqrt_rhoN = np.sqrt(rho_N)
        d_N = 1.0 / sqrt_rhoN   # loop-network spacing [m]; caps the overlap radius

        def _one(C_num, C_cont, l_scale, v_abs, c_LL, c_LN):
            # Mean radius and number density [m^-3]
            r = l_scale * np.sqrt(C_cont / C_num)
            N = C_num / Omega
            # ── Bounded overlap gates (cap the radius at the relevant spacing) ──
            # Cap r at the loop-loop spacing d_LL = N^(-1/3) and the network
            # spacing d_N = rho_N^(-1/2) so the Avrami arguments SATURATE instead
            # of diverging as the number density collapses (an un-capped gate gave
            # a near-discontinuous number collapse that broke point-defect
            # conservation).  Onset stays smooth: phi grows continuously with r
            # until the loops touch their neighbours / the network.
            d_LL = (Omega / C_num) ** (1.0 / 3.0)   # loop-loop spacing  [m]
            r_LL = r if r < d_LL else d_LL
            r_LN = r if r < d_N  else d_N
            phi_LL = 1.0 - np.exp(-kappa_LL * FOUR_THIRDS_PI * r_LL**3 * N)
            phi_LN = 1.0 - np.exp(-kappa_LN * np.pi * r_LN**2 * rho_N)
            nu_LL = c_LL * v_abs * N**(1.0 / 3.0)
            nu_LN = c_LN * v_abs * sqrt_rhoN
            num_LN    = nu_LN * phi_LN * C_num
            num_loss  = nu_LL * phi_LL * C_num + num_LN
            cont_loss = nu_LN * phi_LN * C_cont   # channel 1 conserves content
            return {'num': num_loss, 'content': cont_loss, 'num_LN': num_LN, 'r': r}

        return {
            'iL':  _one(c[4], c[8],  l_a, v_abs_a, c_LL_a, c_LN_a),
            'aiL': _one(c[5], c[9],  l_a, v_abs_a, c_LL_a, c_LN_a),
            'vL':  _one(c[6], c[10], l_c, v_abs_c, c_LL_c, c_LN_c),
            'avL': _one(c[7], c[11], l_c, v_abs_c, c_LL_c, c_LN_c),
        }

    def rho_N_source(self, concentrations=None, coal=None):
        """Network-density growth/recovery rate d(rho_N)/dt [m^-2 s^-1].

        Growth: each loop absorbed into the network (loop-network coalescence,
        channel 2) contributes 2*pi*r of dislocation line length per unit volume.
        Summing the four families,
            source = c_rhoN * sum_k 2*pi*r_k * num_LN_k / Omega
        Recovery: first-order relaxation of the irradiation-grown excess back
        toward the grown-in seed (network climb annihilation), so rho_N saturates
        rather than growing without bound:
            recovery = k_rhoN_rec * (rho_N - rho_N_seed)
        """
        if coal is None:
            coal = self.coalescence_rates(concentrations)
        Omega   = self.input_data.physical_props['Omega']
        c_rhoN  = self.input_data.model_params.get('c_rhoN', 0.1)
        k_rec   = self.input_data.model_params.get('k_rhoN_rec', 1e-7)
        seed    = self.input_data.material_params['rho']
        rho_N   = self.current_rho_N(concentrations)

        two_pi = 2.0 * np.pi
        source = c_rhoN * two_pi / Omega * sum(
            coal[k]['r'] * coal[k]['num_LN'] for k in ('iL', 'aiL', 'vL', 'avL'))
        recovery = k_rec * (rho_N - seed)
        return source - recovery