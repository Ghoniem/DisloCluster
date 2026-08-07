import numpy as np
import sys, os
from matplotlib import pyplot as plt
print(os.environ['PATH'])
sys.path.append('lib')
from readF import *
from readFile import *
from readEVL import *
import pathlib


# ------------------------------------------------------------------------ #
#                          Main Inputs
# ------------------------------------------------------------------------ #
sim_root = str(pathlib.Path().resolve()) + '/sims' # Adjust this path as needed to directory with simulation folders
folders = ['simA', 'simB', 'simC']  # Replace with actual folder names in the sims directory
matFileName = ['Zr3.txt']  # Replace with actual mat file name in the inputFiles directory

textfont = 18
axisfont = 16


# ------------------------------------------------------------------------ #
#                          Main Processing Loop
# ------------------------------------------------------------------------ #
for folder_name in folders:
    folder = os.path.join(sim_root, folder_name)
    if not os.path.isdir(folder):
        print(f"Skipped {folder}")
        continue

    print(f"Processing {folder}")
    matfile = os.path.join(folder, 'inputFiles', 'Zr3.txt')
    F, Flabels = readFfile(folder)
    t = getFarray(F, Flabels, 'time [b/cs]')
    DoseRate = get_scalar(matfile, 'doseRate_dpaPerSec')
    mu0_SI = get_scalar(matfile, 'mu0_SI')
    rho_SI = get_scalar(matfile, 'rho_SI')
    b_SI = get_scalar(matfile, 'b_SI')
    v_dd2SI = np.sqrt(mu0_SI / rho_SI)
    t_dd2SI = b_SI / v_dd2SI
    dpa = t * t_dd2SI * DoseRate  # <-- missing before

    # Plotting strain components and swelling
    b11 = getFarray(F, Flabels, 'betaP_11')
    b22 = getFarray(F, Flabels, 'betaP_22')
    b33 = getFarray(F, Flabels, 'betaP_33')
    b12 = getFarray(F, Flabels, 'betaP_12')
    b13 = getFarray(F, Flabels, 'betaP_13')
    b23 = getFarray(F, Flabels, 'betaP_23')
    swelling_trace_over_3 = []

    for i in range(len(t)):
        betaP = np.array([
            [b11[i], b12[i], b13[i]],
            [b12[i], b22[i], b23[i]],
            [b13[i], b23[i], b33[i]],
        ])
        trace = np.trace(betaP)
        swelling_trace_over_3.append(trace / 3)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dpa, b11, label=r'$\beta_{11}$', color='red')
    ax.plot(dpa, b22, label=r'$\beta_{22}$', color='green')
    ax.plot(dpa, b33, label=r'$\beta_{33}$', color='blue')
    ax.plot(dpa, swelling_trace_over_3, label=r'Swelling ($\mathrm{tr}(\beta^P)/3$)', color='gray', linestyle='-')
    ax.set_xlabel('Irradiation Dose [dpa]', fontsize=textfont)
    ax.set_ylabel('Strain', fontsize=textfont)
    ax.legend(fontsize=axisfont)
    ax.grid(True, linestyle=':')
    ax.set_xlim(left=0)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.tick_params(axis='both', which='major', direction='in', length=6, labelsize=axisfont)
    fig.tight_layout()

    # Plot dislocation density
    glissile_density = getFarray(F, Flabels, 'glissile density [m^-2]')
    sessile_density  = getFarray(F, Flabels, 'sessile density [m^-2]')
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(dpa, glissile_density+sessile_density, label='Dislocation Density', color='k', linewidth=2)
    ax2.set_xlabel('Irradiation Dose [dpa]', fontsize=textfont)
    ax2.set_ylabel('Dislocation Density [m$^{-2}$]', fontsize=textfont)
    ax2.legend(fontsize=axisfont)
    ax2.grid(True, linestyle=':')
    ax2.set_xlim(left=0)
    ax2.tick_params(axis='both', which='major', direction='in', length=6, labelsize=axisfont)
    fig2.tight_layout()
 
    # Save figures
    save_path = os.path.join(folder, 'StrainComponentsWithSwelling.pdf')
    fig.savefig(save_path)
    save_path2 = os.path.join(folder, 'DislocationDensities_vs_DPA.pdf')
    fig2.savefig(save_path2)   
    plt.close(fig)
    print(f"Saved {save_path}")
    plt.close(fig2)
    print(f"Saved {save_path2}")

