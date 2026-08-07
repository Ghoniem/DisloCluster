/**
 * rate_equations.h – ODE right-hand side for the ZrMicro 12-equation system.
 *
 * Mirrors: py_utils/rate_equations.py  +  py_utils/reaction_rates.py
 *
 * State vector y[12]:
 *   [0]  Cv     – vacancy concentration
 *   [1]  Ci     – interstitial concentration
 *   [2]  C2i    – di-interstitial concentration
 *   [3]  C3i    – tri-interstitial concentration
 *   [4]  CiL    – interstitial loop density
 *   [5]  CaiL   – aligned interstitial loop density
 *   [6]  CvL    – vacancy loop density
 *   [7]  CavL   – aligned vacancy loop density
 *   [8]  CiL_i  – interstitials in interstitial loops
 *   [9]  CaiL_i – interstitials in aligned interstitial loops
 *   [10] CvL_v  – vacancies in vacancy loops
 *   [11] CavL_v – vacancies in aligned vacancy loops
 */
#pragma once

#include "parameters.h"
#include <nvector/nvector_serial.h>
#include <sundials/sundials_types.h>

/**
 * CVODE-compatible RHS callback.
 * user_data must point to a Parameters struct.
 */
int rhs_zrmicro(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);
