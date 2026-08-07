"""
simulation.py – ZrMicro simulation orchestrator.

Responsibilities:
  1. Load inputs and initialise physics modules (input_data, reaction_rates,
     rate_equations).
  2. Run the ODE integration via scipy (run_simulation / _ode_wrapper).
  3. Delegate post-processing to post_process and plotting to visualization.

The ODE interface (_ode_wrapper, run_simulation) is kept as a thin, isolated
layer to facilitate future migration of the heavy-lifting to a C++ back-end.
"""

import signal
import traceback
import time as _time

import numpy as np
from scipy.integrate import solve_ivp


class _InterruptGuard:
    """Defer SIGINT for the duration of a scipy/f2py (LSODA/Radau/BDF) integration.

    Raising ``KeyboardInterrupt`` *inside* an f2py callback (the Fortran→Python
    ODE RHS) corrupts f2py's thread-local callback pointer and aborts the
    interpreter with::

        Fatal Python error: F2PySwapThreadLocalCallbackPtr: PyLong_AsVoidPtr failed

    which surfaces as a hard "Kernel crashed" in Jupyter — the surrounding
    ``try/except KeyboardInterrupt`` never runs because the process is already
    dead. To avoid that, we install a handler that merely *records* the
    interrupt. The ODE RHS polls ``.interrupted`` and, once set, returns NaNs so
    the integrator gives up via a normal convergence failure (no exception
    crosses the f2py boundary). The caller then re-raises a clean
    ``KeyboardInterrupt`` from pure-Python context.
    """

    def __init__(self):
        self.interrupted = False
        self._old_handler = None

    def __enter__(self):
        try:
            self._old_handler = signal.signal(signal.SIGINT, self._handle)
        except ValueError:
            # signal.signal only works in the main thread; if we're not there
            # (e.g. background worker) leave the default handler in place.
            self._old_handler = None
        return self

    def _handle(self, signum, frame):
        self.interrupted = True
        print("\n⚠ Interrupt received — stopping integration at the next RHS call...")

    def __exit__(self, *exc_info):
        if self._old_handler is not None:
            signal.signal(signal.SIGINT, self._old_handler)
        return False

from py_utils.input_data import InputData
from py_utils.reaction_rates import ReactionRates
from py_utils.rate_equations import RateEquations
from py_utils import pre_process, post_process


class ZrMicroSimulation:
    """
    Orchestrates the ZrMicro simulation workflow:
    load inputs → initialise physics → run ODE → post-process → plot.
    """

    def __init__(self, input_file=None):
        print("Initializing ZrMicro simulation...")
        try:
            self.input_file = pre_process.find_input_file(input_file)
            print(f"Loading input data from {self.input_file}...")
            self.input_data = InputData(str(self.input_file))

            print("Initializing reaction rates...")
            self.reaction_rates = ReactionRates(self.input_data)

            print("Setting up rate equations...")
            self.rate_equations = RateEquations(self.input_data, self.reaction_rates)

            pre_process.validate_setup(self.input_data)
            print("✓ Simulation initialized successfully!")
        except Exception as e:
            print(f"❌ Error during initialization: {e}")
            raise

    # ── ODE integration (C++ migration target) ────────────────────────────────

    def run_simulation(self, t_begin=1e-6, t_end=1e8, n_points=1000,
                       method='LSODA', rtol=1e-6, atol=1e-20,
                       log_time=True, max_step=None):
        """
        Run the microstructure evolution ODE and return processed results.

        Parameters
        ----------
        t_begin   : float  – start time [s]
        t_end     : float  – end time [s]
        n_points  : int    – number of output time points
        method    : str    – ODE solver ('LSODA', 'Radau', 'BDF')
        rtol, atol: float  – solver tolerances
        log_time  : bool   – logarithmic time spacing (True) or linear (False)
        max_step  : float  – maximum solver step; defaults to t_end/100

        Returns
        -------
        results : dict or None
        """
        print(f"Running simulation from 0 to {t_end:.1e} seconds...")
        print(f"Using {method} solver with rtol={rtol:.1e}, atol={atol:.1e}")

        t_eval = (np.logspace(np.log10(t_begin), np.log10(t_end), n_points)
                  if log_time else np.linspace(0, t_end, n_points))

        y0 = pre_process.get_initial_conditions(self.rate_equations)
        print("\nInitial conditions:")
        for i, name in enumerate(self.rate_equations.concentration_names):
            print(f"  {name}: {y0[i]:.2e}")

        # Progress monitoring state
        self._progress_wall_time = 0.0
        self._progress_reported  = 0
        self._t_end              = t_end

        if max_step is None:
            max_step = t_end / 100

        # SIGINT must NOT raise inside the f2py ODE callback (it crashes the
        # interpreter); the guard turns it into a cooperative flag instead.
        self._interrupt_guard = _InterruptGuard()
        try:
            print(f"\nIntegrating with {method} solver...")
            with self._interrupt_guard:
                sol = solve_ivp(
                    self._ode_wrapper,
                    [0, t_end], y0,
                    method=method, t_eval=t_eval,
                    rtol=rtol, atol=atol,
                    max_step=max_step,
                )
            if self._interrupt_guard.interrupted:
                # Re-raise from pure-Python context, now that the f2py callback
                # has unwound safely. Lets callers (e.g. the fitting loop) catch it.
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            print("\n⚠ Interrupted — stopping integration cleanly (no results).")
            raise
        except Exception as e:
            print(f"❌ Error during integration: {e}")
            traceback.print_exc()
            return None
        finally:
            self._interrupt_guard = None

        if not sol.success:
            print(f"❌ Integration failed: {sol.message}")
            return None

        print("✓ Integration successful!")
        post_process.check_solution_quality(sol, self.rate_equations.concentration_names)
        return post_process.process_solution(
            sol, self.input_data, self.rate_equations, self.input_file
        )

    def _ode_wrapper(self, t, y):
        """
        Thin adapter between solve_ivp and rate_equations.ode_system.

        Adds progress reporting and guards against unphysical states.
        Replace this method (and the solve_ivp call above) when migrating
        to a C++ solver.
        """
        # If the user interrupted, ask the integrator to abort by returning
        # non-finite derivatives. LSODA/BDF treat this as a corrector failure
        # and stop without unwinding an exception through the f2py boundary
        # (which would crash the interpreter). The flag is re-raised cleanly in
        # run_simulation once solve_ivp has returned.
        guard = getattr(self, '_interrupt_guard', None)
        if guard is not None and guard.interrupted:
            return np.full_like(y, np.nan)

        if np.any(y < -1e-15):
            neg = np.where(y < -1e-15)[0]
            if len(neg) < 5:
                names = [self.rate_equations.concentration_names[i] for i in neg[:3]]
                print(f"Warning: negative concentrations at t={t:.2e}: {names}")

        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError(f"Invalid concentrations at t={t:.2e}")

        self._report_progress(t)

        dydt = self.rate_equations.ode_system(t, y)

        if np.any(np.isnan(dydt)) or np.any(np.isinf(dydt)):
            raise ValueError(f"Invalid derivatives at t={t:.2e}")

        return dydt

    def _report_progress(self, t):
        """Print a brief progress line every ~10 wall-clock seconds."""
        now = _time.time()
        if now - self._progress_wall_time > 10:
            self._progress_wall_time = now
            pct = int(t / self._t_end * 100) if self._t_end > 0 else 0
            if pct > self._progress_reported + 10:
                print(f"  Integration progress: t = {t:.2e} s")
                self._progress_reported = pct

