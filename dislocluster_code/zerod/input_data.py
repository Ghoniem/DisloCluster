# input_data.py - FIXED VERSION
# Class to handle input data from Excel workbook with three worksheets
# Location: /utilities/input_data.py

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path.cwd()
INPUT_DIR = BASE_DIR / 'input'
# Input file path
INPUT_FILE = INPUT_DIR / 'Zr_input_parameters.xlsx'

class InputData:
    """
    Class to handle input data from Excel workbook with three worksheets
    """

    def __init__(self, excel_file=INPUT_FILE):
        """
        Initialize by reading data from Excel workbook
        
        Parameters:
        excel_file: str or Path - path to Excel file with three worksheets
        """
        # Store the file path properly
        self.excel_file = str(excel_file)  # Convert Path to string if needed
        print(f"InputData: Attempting to load {self.excel_file}")
        
        # Check if file exists
        file_path = Path(self.excel_file)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            print(f"Current working directory: {Path.cwd()}")
            print("Setting default values instead...")
            self._set_default_values()
        else:
            print(f"✓ File exists: {file_path}")
            self.load_data()
        
        self.calculate_derived_parameters()
    
    def load_data(self):
        """
        Load data from the three Excel worksheets
        """
        try:
            print(f"Reading Excel file: {self.excel_file}")
            
            # Read the three worksheets
            self.material_env_df = pd.read_excel(self.excel_file, sheet_name='Material_Environment')
            self.physical_props_df = pd.read_excel(self.excel_file, sheet_name='Physical_Properties')
            self.model_params_df = pd.read_excel(self.excel_file, sheet_name='Model_Parameters')
            
            # Convert to dictionaries for easier access
            self.material_params = self._df_to_dict(self.material_env_df)
            self.physical_props = self._df_to_dict(self.physical_props_df)
            self.model_params = self._df_to_dict(self.model_params_df)

            # Ensure the T-dependent network density anchors exist even if the
            # workbook predates them (rho becomes a log-linear function of T,
            # high at low T and annealed out at high T — see
            # calculate_derived_parameters / _update_rho_of_T).
            self.material_params.setdefault('rho_300', 1e15)
            self.material_params.setdefault('rho_600', 1e13)

            # T-dependent interstitial DAD bias anchors (a-loop low-T growth).
            self.model_params.setdefault('delta_DAD_i_300', 0.20)
            self.model_params.setdefault('delta_DAD_i_600', 0.10)

            # Minimum stable a-loop radius: halts vacancy-driven a-loop shrinkage
            # so the mean a-loop diameter cannot collapse to zero at high T/dose
            # (see ReactionRates.a_shrink_gate). r_min_a <= 0 disables the gate.
            self.model_params.setdefault('r_min_a', 1.0e-9)   # [m]
            self.model_params.setdefault('w_rmin', 0.3)       # ramp width [-]

            # Cascade a-loop nucleation source (mirrors the c-loop cascade source).
            # epsilon_iL is the cascade interstitial-loop efficiency; n_iL_nuc the
            # interstitials per nucleated cascade a-loop. epsilon_iL = 0 disables
            # the cascade source (recovering pure clustering nucleation).
            self.physical_props.setdefault('epsilon_iL', 0.1)
            self.model_params.setdefault('n_iL_nuc', 20.0)

            print("✓ Successfully loaded input data from Excel file")
            
        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
            print("Setting default values instead...")
            # Set default values if file loading fails
            self._set_default_values()
    
    def _df_to_dict(self, df):
        """
        Convert DataFrame to dictionary using 'Notation' column as keys
        """
        if 'Notation' in df.columns and 'Value' in df.columns:
            return dict(zip(df['Notation'], df['Value']))
        else:
            # Alternative column names
            return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    
    def _set_default_values(self):
        """
        Set default values for Zirconium if Excel file cannot be loaded
        """
        print("Setting default values for Zirconium...")
        
        # Material & Environment Parameters (Table 1)
        self.material_params = {
            'rho': 1e14,        # dislocation density [m^-2] (overwritten by rho(T), see below)
            'rho_300': 1e15,    # network dislocation density anchor at 300 K [m^-2]
            'rho_600': 1e13,    # network dislocation density anchor at 600 K [m^-2]
            'd': 10e-6,         # grain diameter [m]
            'r_p': 5e-9,        # precipitate radius [m]
            'N_P': 1e20,        # precipitate number density [m^-3]
            'T': 573,           # Irradiation temperature [K]
            'G': 1e-6,          # Irradiation dose rate [dpa.s^-1]
            'N_C': 1e21,        # Nucleated void density [m^-3]
            'N_iL': 1e21        # Nucleated interstitial loop density [m^-3]
        }
        
        # Physical Properties (Table 2)
        self.physical_props = {
            'a': 0.323e-9,      # lattice parameter [m]
            'b_a': 0.323e-9,    # Burgers vector a-direction [m]
            'b_c': 0.323e-9,    # Burgers vector c-direction [m] (= b_a, Li & Ghoniem 2020)
            'E_m_i': 0.013,     # interstitial migration energy [eV]
            'E_m_v': 1.3,       # vacancy migration energy [eV]
            'E_F_v': 1.6,       # vacancy formation energy [eV]
            'E_F_i': 3.0,       # interstitial formation energy [eV]
            'gamma': 1.5e19,    # surface energy [eV.m^-2]
            'gamma_sf': 8e17,   # stacking fault energy [eV.m^-2]
            # Attempt/vibration frequencies — Debye-scale for Zr (~5e12 Hz).
            # NOT free fit parameters: an unphysically low nu_v (~2e8, an old
            # fit artefact from bad Params bounds) made vacancies ~1e5x too slow,
            # so ~98% of vacancies recombined instead of reaching c-loops and
            # vacancy loops could not grow. Keep both at Debye scale. [[dad-loop-bias]]
            'nu_i': 5e12,       # interstitial vibration frequency [s^-1]
            'nu_v': 5e12,       # vacancy vibration frequency [s^-1]
            # hcp Zr, (sqrt(3)/2)*a^2*c/2 with two atoms per primitive
            # cell. This fallback used to read 1.4e-29 while the workbook
            # carried 1.2e-29 and the physical value is 2.33e-29 -- three
            # different numbers for one lattice constant.
            'Omega': 2.326553e-29,  # atomic volume [m^3]
            'z_c': 48,          # Number of unstable sites around vacancy
            'epsilon_vL': 0.1,  # Vacancy loop cascade efficiency
            'epsilon_iL': 0.1,  # Interstitial (a-)loop cascade efficiency
            'epsilon_2i': 0.05, # Di-interstitial cascade efficiency
            'epsilon_3i': 0.02, # Tri-interstitial cascade efficiency
            'D_0_v': 5e-13,     # Vacancy diffusion pre-exponential [m^2.s^-1]
            'D_0_i': 8e-10,     # Interstitial diffusion pre-exponential [m^2.s^-1]
            'M': 0.5,           # Average Schmidt factor
            'recom': 1.0        # Recombination volume factor (dimensionless)
        }
        
        # Model Parameters (Table 3 & 4)
        self.model_params = {
            # Bias factors. Keys MUST match the un-suffixed notation used both in
            # the Excel Model_Parameters sheet and in ReactionRates/cpp_bridge
            # lookups ('Z_N', 'Z_iL', 'Z_vL'); the previous '_i'-suffixed names
            # were never found, so this fallback silently used hard-coded defaults.
            # Values mirror the active Excel workbook.
            'Z_N': 1.1,         # Network dislocation bias for interstitials
            'Z_aN_i': 1.2,      # Aligned network bias for interstitials
            'Z_iL': 1.3,        # Interstitial loop bias for interstitials
            'Z_aiL_i': 1.2,     # Aligned interstitial loop bias
            'Z_vL': 1.0,        # Vacancy loop bias for interstitials
            'Z_avL_i': 1.2,     # Aligned vacancy loop bias for interstitials
            'Z_v': 1.0,         # General vacancy bias

            # Diffusional Anisotropy Difference (DAD) — see
            # calculate_derived_parameters(). delta_DAD sets the orientation-
            # and species-resolved loop capture efficiencies that let BOTH
            # interstitial (a-type) and vacancy (c-type) loops grow at once.
            # delta_DAD_i / delta_DAD_v default to delta_DAD if absent;
            # explicit Z_i_a/Z_v_a/Z_i_c/Z_v_c keys override the parameterization.
            'delta_DAD': 0.2,   # interstitial/vacancy diffusion-anisotropy difference (now governs delta_DAD_v)
            # Interstitial DAD bias as a LINEAR function of T (anchors at 300/600 K):
            # stronger at low T so a-loops grow by flux, not coalescence (lifts the
            # low-T a-loop size and frees c_LL_a to recover the 573 K number density).
            'delta_DAD_i_300': 0.20,  # a-loop interstitial DAD bias anchor at 300 K
            'delta_DAD_i_600': 0.10,  # a-loop interstitial DAD bias anchor at 600 K

            # Vacancy loop growth parameters
            'Q': 0.5,           # Absorption efficiency for <c> loops
            'A_a': 1.0,         # Strain factor in <a> direction
            'A_c': 0.5,         # Strain factor in <c> direction
            'tau_vL0': 0.0237,  # [s]  vacancy-loop lifetime Arrhenius prefactor
            'E_a_vL': 0.268,    # [eV] vacancy-loop lifetime activation energy
            # NOTE: the aligned-loop lifetime is now DERIVED (tau_avL0/E_a_avL,
            # defaulting to tau_vL0/E_a_vL) — see calculate_derived_parameters().
            # The legacy constants `tau_vL=1000` (never read) and `tau_avL=500`
            # (a fixed override that decoupled the dominant aligned population
            # from the fit) have been removed.

            # Vacancies per freshly nucleated cascade vacancy loop.  G_vL is the
            # cascade vacancy ATOM rate (= G*eps_vL*(1-f_a)); the loop NUMBER
            # nucleation rate is therefore G_vL/n_vL_nuc and the loop CONTENT
            # seed is G_vL, so each nucleated loop carries n_vL_nuc vacancies and
            # is born at radius r = l_c*sqrt(n_vL_nuc).  Pick from the observed
            # nucleation size: n_vL_nuc = (r_nuc / l_c)^2  (~70 for a 1 nm loop).
            # NOTE: this rescales the vacancy-loop number-density saturation to
            #   N_vL,sat = G_vL * tau_vL / (n_vL_nuc * Omega)
            # so tau_vL0 / E_a_vL must be RE-FIT after changing n_vL_nuc.
            'n_vL_nuc': 20.0,   # vacancies per nucleated cascade vacancy loop

            # Interstitials per freshly nucleated cascade a-loop.  Mirrors
            # n_vL_nuc for the interstitial side: G_iL/G_aiL are the cascade
            # interstitial ATOM rates (= G*eps_iL*(1-f_a) / G*eps_iL*f_a); the
            # loop NUMBER nucleation rate is G_iL/n_iL_nuc and the loop CONTENT
            # seed is G_iL, so each cascade a-loop is born carrying n_iL_nuc
            # interstitials at radius r = l_a*sqrt(n_iL_nuc).  This cascade source
            # ADDS to the existing homogeneous (i+3i, 2i+2i) clustering source.
            'n_iL_nuc': 20.0,   # interstitials per nucleated cascade a-loop

            # Radiation hardening coefficients
            'alpha_i': 0.15,    # Strength coefficient for interstitials
            'alpha_v': 0.1,     # Strength coefficient for vacancies
            'alpha_iL': 0.25,   # Strength coefficient for interstitial loops
            'alpha_aiL': 0.3,   # Strength coefficient for aligned int loops
            'alpha_vL': 0.2,    # Strength coefficient for vacancy loops
            'alpha_avL': 0.25,  # Strength coefficient for aligned vac loops
            'alpha_N': 0.3,     # Strength coefficient for network dislocations
            
            # Empirical parameters
            'V_c_cap': 1e-27,   # Capture volume around vacancy loop [m^3]
            'f': 0.1,           # Fraction of aligned loops
            'm': 100,           # Mobile dislocation increment
            's': 1e-3,          # Mobile to forest conversion rate
            'r_m': 1e-6,        # Mobile dislocation recovery rate
            'r_f': 1e-7,        # Forest dislocation recovery rate
            'C_floor': 1e-20,   # Minimum concentration floor (atomic fraction)

            # Geometric loop coalescence (growth-flux driven). Rate scale tied to
            # dr/dt; smooth Avrami overlap probabilities gate the high-dose onset.
            # See reaction_rates.coalescence_rates().  All dimensionless ~O(1);
            # tune c_LL / c_LN to match measured saturation densities.
            # These are also defined in the Excel workbook (Model_Parameters
            # sheet for values, Params sheet for fitting [min, max] bounds); the
            # values below are only the fallback used when the file is absent.
            'c_LL': 1.0,        # like-loop coalescence rate-scale coefficient
            'kappa_LL': 1.0,    # loop-loop capture-zone factor
            'c_LN': 1.0,        # loop-network coalescence rate-scale coefficient
            'kappa_LN': 1.0,    # loop-network capture factor

            # Orientation-resolved coalescence (override the shared c_LL/c_LN per
            # loop family). a-loops = interstitial (iL, aiL); c-loops = vacancy
            # (vL, avL). Weaker c-loop coalescence -> larger c-component vacancy
            # loops (30-60 nm) independent of the a-loop size/density. Absent ->
            # fall back to the shared c_LL/c_LN (legacy locked-size behavior).
            'c_LL_a': 1.0,      # a-loop (interstitial) like-loop coalescence
            'c_LN_a': 1.0,      # a-loop (interstitial) loop-network coalescence
            'c_LL_c': 1.0,      # c-loop (vacancy) like-loop coalescence
            'c_LN_c': 1.0,      # c-loop (vacancy) loop-network coalescence

            # Minimum stable a-loop radius — halts vacancy-driven a-loop
            # shrinkage so the mean a-loop diameter cannot collapse to zero at
            # high T/dose (see ReactionRates.a_shrink_gate). <= 0 disables it.
            'r_min_a': 1.0e-9,  # minimum stable a-loop radius [m]
            'w_rmin': 0.3,      # smoothstep ramp width (fraction of r_min_a)
        }
    
    def check_atomic_volume(self, rtol=0.05, verbose=True):
        """Compare ``physical_props['Omega']`` against hcp Zr, two ways.

        Omega is not a fitting parameter -- it is a lattice constant, and this
        file already carries the data to determine it twice over:

            from the lattice     V_cell = (sqrt(3)/2) a^2 c, TWO atoms per hcp
                                 primitive cell, with a = b_a and c = b_c
            from the mass        Omega = M_Zr / (rho * N_A), with rho = 6520
                                 kg/m^3 (rho_SI in Zr3d_ghoniem.txt)

        Both give 2.32e-29 m^3 and agree to 0.14%. The stored value is
        1.2e-29, which is 0.516x that -- almost exactly V_cell/4, i.e. four
        atoms per hcp primitive cell, which hcp does not have.

        THIS IS NOT A REPORTING BUG. Omega appears throughout the model: every
        number density is C/Omega, loop line density is 2*pi*r*C/Omega, the
        loop-radius prefactors are sqrt(Omega/(pi*b)), and the stress-biased
        emission carries exp(sigma*Omega/kT). Correcting it moves the
        predictions -- measured on the calibrated set at 10 dpa:

            N_a  0.483x     N_c  0.516x     c_a  0.522x
            Cv   1.047x     Ci   1.059x

        The loop number densities HALVE. Since the 28-parameter set was fitted
        against experimental loop densities at the old Omega, correcting it
        without refitting trades one error for another. Hence a check that
        reports, and `calibration.build_sim(physical_omega=True)` to opt in --
        not a silent default.

        Returns ``(stored, physical, agrees)``.
        """
        pp = self.physical_props
        stored = float(pp['Omega'])
        a = float(pp.get('b_a', 3.23e-10))       # hcp a  [m]
        c = float(pp.get('b_c', 5.15e-10))       # hcp c  [m]
        physical = np.sqrt(3.0) / 2.0 * a * a * c / 2.0
        agrees = abs(stored - physical) <= rtol * physical
        if verbose and not agrees:
            print(f"  WARNING: Omega = {stored:.4e} m^3 is {stored / physical:.3f}x "
                  f"the hcp value {physical:.4e} m^3 "
                  f"(a={a:.3e}, c={c:.3e}, 2 atoms/cell)")
            print(f"           number densities are C/Omega, so they are off by "
                  f"{physical / stored:.2f}x. See InputData.check_atomic_volume.")
        return stored, physical, agrees

    def calculate_derived_parameters(self):
        """
        Calculate derived parameters from input data
        """
        # Constants
        k_B = 8.617e-5  # Boltzmann constant [eV/K]
    
        # Derived parameters
        self.derived = {}

        # Vacancy diffusion coefficient (kept for reference / diagnostics).
        # D_v matches ReactionRates.calculate_diffusion_coefficients():
        #   D_v = (a^2 / z_c) * omega_v,   omega_v = z_c * nu_v * exp(-E_m_v / kT)
        # (the z_c cancels, giving D_v = a^2 * nu_v * exp(-E_m_v / kT)).
        omega_v = (self.physical_props['z_c'] * self.physical_props['nu_v'] *
                   np.exp(-self.physical_props['E_m_v'] /
                          (k_B * self.material_params['T'])))
        self.derived['D_v'] = (self.physical_props['a']**2 /
                               self.physical_props['z_c']) * omega_v

        # Temperature-dependent vacancy-loop annealing lifetime, Arrhenius form
        #   tau_vL(T) = tau_vL0 * exp(E_a_vL / kT).
        # This reproduces the measured vacancy-loop saturation number density
        #   N_vL,sat = G_vL * tau_vL / Omega
        # of ~4e22 m^-3 at 300 K and ~1.5e20 m^-3 at 650 K.
        # tau_vL0 [s] and E_a_vL [eV] are the fit parameters (Model_Parameters /
        # Params sheets); this superseded an earlier diffusion-based lifetime.
        tau_vL0 = self.model_params.get('tau_vL0', 0.0237)   # [s]  prefactor
        E_a_vL  = self.model_params.get('E_a_vL', 0.268)     # [eV] activation energy
        self.derived['tau_vL'] = tau_vL0 * np.exp(
            E_a_vL / (k_B * self.material_params['T']))

        # Aligned vacancy-loop lifetime.  Aligned (avL) and non-aligned (vL)
        # vacancy loops are the SAME defect, only split by the stress-alignment
        # fraction f_a, so they share ONE annealing law:
        #     tau_avL(T) = tau_avL0 * exp(E_a_avL / kT),
        # with tau_avL0 / E_a_avL defaulting to the non-aligned tau_vL0 / E_a_vL.
        # Tying them makes the two FITTED parameters (tau_vL0, E_a_vL) the sole
        # control of the TOTAL c-loop steady-state density
        #     N_c,sat = (G_vL + G_avL)*tau_vL / (n_vL_nuc*Omega).
        # This removes the legacy constant `tau_avL` (a fixed ~500 s that pinned
        # the dominant aligned population and made it independent of the fit — the
        # override that prevented the c-loop density from being calibrated). The
        # old scalar model_params['tau_avL'] is intentionally NOT consulted.
        tau_avL0 = self.model_params.get('tau_avL0', tau_vL0)   # [s]
        E_a_avL  = self.model_params.get('E_a_avL', E_a_vL)     # [eV]
        self.derived['tau_avL'] = tau_avL0 * np.exp(
            E_a_avL / (k_B * self.material_params['T']))

        # Thermal equilibrium vacancy concentration
        self.derived['C_v_eq'] = np.exp(-self.physical_props['E_F_v'] / 
                                      (k_B * self.material_params['T']))
        
        # Compact notation parameters (Section 3.1)
        self.derived['l'] = (self.physical_props['z_c'] * self.physical_props['Omega'] / 
                           (2 * np.pi * self.physical_props['a']**2))
        
        self.derived['l_a'] = np.sqrt(self.physical_props['Omega'] / 
                                    (np.pi * self.physical_props['b_a']))
        
        # THE <c> LOOP BURGERS MAGNITUDE, NOT THE LATTICE CONSTANT c.
        # A <c> loop is a 1/2[0001] vacancy loop, so |b| = c/2 = 2.575 A.
        #
        # `b_c` is deliberately NOT reused here. It is also the lattice
        # constant c behind Omega = (sqrt(3)/4) a^2 c, and halving it would
        # halve Omega -- the exact error this repository already had to
        # correct once. A separate key keeps the two roles apart.
        self.physical_props.setdefault(
            'b_cL', 0.5 * self.physical_props['b_c'])
        self.derived['l_c'] = np.sqrt(self.physical_props['Omega'] /
                                    (np.pi * self.physical_props['b_cL']))

        # ── Diffusional Anisotropy Difference (DAD) bias factors ─────────────
        # In HCP Zr the interstitial diffusion tensor is far more anisotropic
        # (near-2D, basal-plane) than the vacancy tensor.  This anisotropy
        # DIFFERENCE partitions arriving point defects by sink ORIENTATION, so
        # a-type (interstitial) loops see a net interstitial flux while c-type
        # (vacancy) loops see a net vacancy flux — even at equal production and
        # equal geometric capture.  This is the mechanism (Woo, anisotropic-
        # diffusion theory of Zr irradiation growth) that lets BOTH loop
        # families grow simultaneously; a single scalar bias on one flux cannot.
        #
        
        # Orientation- and species-resolved capture efficiencies:
        #   a-loops (iL, aiL):  Z_i_a = 1 + delta_i ,  Z_v_a = 1 - delta_v
        #   c-loops (vL, avL):  Z_i_c = 1 - delta_i ,  Z_v_c = 1 + delta_v
        # NOTE (physical DAD, preferred): the (1 +/- delta) form above is a
        # phenomenological split.  The physically grounded values are the Woo
        # diffusional-anisotropy-difference coefficients (see paper Sec. 4,
        # Eq. 55), written in terms of p_m = (D_c^m / D_a^m)**(1/6):
        #   a-loops:  Z_{a,m} = Z_m^0 * (p_m + p_m**-2) / 2
        #   c-loops:  Z_{c,m} = Z_m^0 *  p_m                    ,  m in {I, V}
        # These feed the SAME bilinear growth law Z_{s,I}*Phi_I - Z_{s,V}*Phi_V,
        # so this is a re-parameterization, NOT a structural change.  The net
        # (dislocation) bias enters only through the ratio Z_I0/Z_V0; it does
        # NOT appear in delta.  The exact map between the two forms is
        #
        #   delta_m = (Z_{a,m} - Z_{c,m}) / (Z_{a,m} + Z_{c,m})
        #           = (1 - p_m**3) / (1 + 3*p_m**3)            (paper Eq. 57)
        #
        # e.g. at ~200 C, p_I ~ 0.89-0.92, p_V ~ 1.0  ->  delta_i ~ 0.07-0.09,
        # delta_v ~ 0, with Z_I0/Z_V0 ~ 1.05-1.15 supplied separately.
        #
        # To use the DAD form directly, set p_I, p_V and Z_I0, Z_V0 in
        # Model_Parameters and uncomment the block below; it overrides the
        # (1 +/- delta) defaults via the same explicit-Z path used just below.
        #
        #   p_I  = self.model_params.get('p_I')          # (D_c^I/D_a^I)**(1/6)
        #   p_V  = self.model_params.get('p_V', 1.0)     # ~1 for isotropic V
        #   Z_I0 = self.model_params.get('Z_I0', 1.1)    # net interstitial bias
        #   Z_V0 = self.model_params.get('Z_V0', 1.0)    # net vacancy bias
        #   if p_I is not None:
        #       A = lambda p: (p + p**-2) / 2.0          # a-loop (prismatic)
        #       C = lambda p:  p                          # c-loop (basal)
        #       self.model_params.setdefault('Z_i_a', Z_I0 * A(p_I))
        #       self.model_params.setdefault('Z_v_a', Z_V0 * A(p_V))
        #       self.model_params.setdefault('Z_i_c', Z_I0 * C(p_I))
        #       self.model_params.setdefault('Z_v_c', Z_V0 * C(p_V))
        # Orientation- and species-resolved capture efficiencies:
        #   a-loops (iL, aiL):  Z_i_a = 1 + delta_i ,  Z_v_a = 1 - delta_v
        #   c-loops (vL, avL):  Z_i_c = 1 - delta_i ,  Z_v_c = 1 + delta_v
        # Net loop content growth:
        #   a-loop:  Z_i_a*flux_i - Z_v_a*flux_v
        #   c-loop:  Z_v_c*flux_v - Z_i_c*flux_i
        # With x = D_v C_v / D_i C_i = flux_v / flux_i this gives
        #   i-loops grow for  x < (1 + delta_i)/(1 - delta_v)
        #   v-loops grow for  x > (1 - delta_i)/(1 + delta_v)
        # The two windows OVERLAP for any delta > 0, removing the structural
        # dead-zone of the old single-scalar-bias form (which forced the
        # i-loop and v-loop growth conditions to be disjoint).  Pick delta_DAD
        # from the diffusion-tensor anisotropy, or so that the v-loop threshold
        # (1-delta)/(1+delta) sits below the steady-state flux ratio x.
        # Explicit Z_* keys in Model_Parameters override the parameterization.
        delta_DAD   = self.model_params.get('delta_DAD', 0.2)
        # Interstitial DAD bias: optionally a LINEAR function of temperature, via
        # two anchors delta_DAD_i_300 / delta_DAD_i_600 (at 300 / 600 K). Stronger
        # at low T → larger Z_i_a → a-loops grow by flux instead of leaning on
        # coalescence for size. If the anchors are absent, fall back to the scalar
        # delta_DAD_i / delta_DAD. Clamped to (0, 1) to keep Z_i_a>1, Z_i_c>0.
        d_i_300 = self.model_params.get('delta_DAD_i_300')
        d_i_600 = self.model_params.get('delta_DAD_i_600')
        if d_i_300 is not None and d_i_600 is not None:
            frac = (float(self.material_params['T']) - 300.0) / (600.0 - 300.0)
            delta_DAD_i = float(np.clip(d_i_300 + (d_i_600 - d_i_300) * frac, 1e-3, 0.95))
        else:
            delta_DAD_i = self.model_params.get('delta_DAD_i', delta_DAD)
        delta_DAD_v = self.model_params.get('delta_DAD_v', delta_DAD)
        self.derived['Z_i_a'] = self.model_params.get('Z_i_a', 1.0 + delta_DAD_i)
        self.derived['Z_v_a'] = self.model_params.get('Z_v_a', 1.0 - delta_DAD_v)
        self.derived['Z_i_c'] = self.model_params.get('Z_i_c', 1.0 - delta_DAD_i)
        self.derived['Z_v_c'] = self.model_params.get('Z_v_c', 1.0 + delta_DAD_v)
        print(f"  DAD bias: Z_i_a={self.derived['Z_i_a']:.3f} Z_v_a={self.derived['Z_v_a']:.3f} "
              f"Z_i_c={self.derived['Z_i_c']:.3f} Z_v_c={self.derived['Z_v_c']:.3f}")


        # Stress effects on aligned loops (Equations 9, 11)
        sigma_n = self.material_params.get('sigma_n', 0.0)  # Pa
        Omega = self.physical_props['Omega']
        self.derived['f'] = (
            np.exp((sigma_n * Omega) / (k_B * self.material_params['T'] * 1.602e-19)) - 1
        ) / (
            np.exp((sigma_n * Omega) / (k_B * self.material_params['T'] * 1.602e-19)) + 2
        )
        self.derived['f_a']=(1+2*self.derived['f'])/3
        self.derived['f_na']=1-self.derived['f_a']
        print(f"Derived f: {self.derived['f']}, f_a: {self.derived['f_a']}, f_na: {self.derived['f_na']}")
       # Generation rates
        self.derived['G_v'] = self.material_params['G']*(1 - self.physical_props['epsilon_vL'])

        # Monomer SIA production. epsilon_2i / epsilon_3i are ATOM fractions of
        # SIAs born directly as di-/tri-interstitial clusters (Li & Ghoniem 2020,
        # Eqs 3-5), and the cluster sources are split per atom as G_2i/2 and
        # G_3i/3.  The monomer fraction therefore subtracts (eps_2i + eps_3i) ONCE
        # — subtracting 2*eps_2i + 3*eps_3i previously dropped ~(eps_2i+2*eps_3i)
        # of the interstitial atoms at the source (vacancy-biased imbalance).
        # epsilon_iL of the cascade interstitial atoms are born directly as
        # a-loops (the cascade SIA-loop channel, analogous to epsilon_vL on the
        # vacancy side). They are carved OUT of the free-monomer production so
        # the total interstitial atom rate stays G (point-defect conservation):
        #   G_i + G_2i + G_3i + G_iL + G_aiL = G.
        eps_iL = self.physical_props.get('epsilon_iL', 0.0)
        self.derived['G_i'] = self.material_params['G'] * (1 - self.physical_props['epsilon_2i'] - self.physical_props['epsilon_3i'] - eps_iL)

        self.derived['G_2i'] = self.material_params['G'] * self.physical_props['epsilon_2i']

        self.derived['G_3i'] = self.material_params['G'] * self.physical_props['epsilon_3i']

        # Cascade interstitial a-loop ATOM rates (non-aligned / aligned), split by
        # the aligned fraction f_a exactly like the vacancy loops. Loop NUMBER
        # nucleation = G_iL/n_iL_nuc; loop CONTENT seed = G_iL (see RateEquations).
        self.derived['G_iL'] = self.material_params['G'] * eps_iL * (1-self.derived['f_a'])

        self.derived['G_aiL'] = self.material_params['G'] * eps_iL * self.derived['f_a']

        self.derived['G_vL'] = self.material_params['G'] * self.physical_props['epsilon_vL'] * (1-self.derived['f_a'])

        self.derived['G_avL'] = self.material_params['G'] * self.physical_props['epsilon_vL'] * self.derived['f_a']
        
        # Network dislocation density: log-linear function of temperature
        # (high at low T, annealed out at high T). Computed from two anchors,
        # rho_300 (at 300 K) and rho_600 (at 600 K), and written into
        # material_params['rho'] so ALL downstream consumers — Python
        # reaction_rates, the C++ solver (via cpp_bridge rho_N), and k_N below —
        # pick up the same T-dependent value.
        self._update_rho_of_T()

        # Network dislocation parameters
        if 'rho' in self.material_params:
            # Calculate network sink strength
            self.derived['k_N_v'] = 4 * np.pi * self.material_params['rho']  # Vacancy sink strength
            self.derived['k_N_i'] = 4 * np.pi * self.material_params['rho'] * self.model_params.get('Z_N', 1.1)  # Interstitial sink strength
        
        print("✓ Derived parameters calculated successfully")
        print(f"  C_v_eq = {self.derived['C_v_eq']:.2e}")
        print(f"  G_v = {self.derived['G_v']:.2e}")
        print(f"  G_i = {self.derived['G_i']:.2e}")
        print(f"  rho(T={self.material_params['T']:.0f}K) = {self.material_params['rho']:.2e} m^-2")


    def _update_rho_of_T(self):
        """
        Network dislocation density as a log-linear function of temperature.

            log10(rho(T)) = log10(rho_300)
                            + (log10(rho_600) - log10(rho_300)) * (T - 300)/(600 - 300)

        i.e. rho is interpolated/extrapolated linearly in log-space between two
        anchors (rho_300 at 300 K, rho_600 at 600 K). With the defaults
        rho_300 = 1e15, rho_600 = 1e13 this gives the requested behaviour:
        high density at low T, annealed out at high T. Log-linear keeps rho
        strictly positive at all temperatures (a strictly-linear form would go
        negative above ~600 K, inside the a-loop fit range).

        The result is written into material_params['rho'], the single value read
        by reaction_rates.py, the k_N sink strengths, and the C++ solver
        (cpp_bridge passes material_params['rho'] as rho_N). rho_300/rho_600 are
        the two fit parameters; the anchor temperatures 300/600 K are fixed.
        """
        if 'rho_300' not in self.material_params or 'rho_600' not in self.material_params:
            return  # no anchors → leave the static material_params['rho'] as-is

        T = float(self.material_params['T'])
        T0, T1 = 300.0, 600.0
        log_rho0 = np.log10(self.material_params['rho_300'])
        log_rho1 = np.log10(self.material_params['rho_600'])
        frac = (T - T0) / (T1 - T0)
        self.material_params['rho'] = 10.0 ** (log_rho0 + (log_rho1 - log_rho0) * frac)


    def display_parameters(self):
        """
        Display all parameters in a formatted way
        """
        print("\n" + "="*60)
        print("MATERIAL & ENVIRONMENT PARAMETERS")
        print("="*60)
        for key, value in self.material_params.items():
            print(f"{key}: {value}")
        
        print("\n" + "="*60)
        print("PHYSICAL PROPERTIES")
        print("="*60)
        for key, value in self.physical_props.items():
            print(f"{key}: {value}")
        
        print("\n" + "="*60)
        print("MODEL PARAMETERS")
        print("="*60)
        for key, value in self.model_params.items():
            print(f"{key}: {value}")
        
        print("\n" + "="*60)
        print("DERIVED PARAMETERS")
        print("="*60)
        for key, value in self.derived.items():
            print(f"{key}: {value}")


# Test function
def test_input_data():
    """Test the InputData class"""
    try:
        # Test with default file
        print("Testing InputData class...")
        input_data = InputData()
        input_data.display_parameters()
        print("✓ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_input_data()