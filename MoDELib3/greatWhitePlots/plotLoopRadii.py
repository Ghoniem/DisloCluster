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
folders = ['simA', 'simB']  # Replace with actual folder names in the sims directory
matFileName = ['Zr3.txt']  # Replace with actual mat file name in the inputFiles directory


# ------------------------------------------------------------------------ #
#                           Helper Functions   
# ------------------------------------------------------------------------ #
def getAverageLoopData(folder):
    F = np.loadtxt(os.path.join(folder, 'F/F_0.txt'))
    R, A = np.empty(F.shape[0]), np.empty(F.shape[0])
    for i, runID in enumerate(F[:, 0]):
        evl = readEVLtxt(f"{folder}/evl/evl_{int(runID)}")
        c = np.mean(evl.nodesPos, axis=0)
        nodesR = np.linalg.norm(evl.nodesPos - c, axis=1)
        R[i] = np.mean(nodesR)
        A[i] = np.mean(evl.loopsArea)
    return F[:, 1], R, A

def getLoopData(folder):
    F = np.loadtxt(os.path.join(folder, 'F/F_0.txt'))
    A, b, n = [], [], []
    for runID in F[:, 0]:
        evl = readEVLtxt(f"{folder}/evl/evl_{int(runID)}")
        A.append(evl.loopsArea)
        b.append(evl.loopsBurger)
        n.append(evl.loopsNormal)
    return F[:, 1], A, b, n

def filteredAverageAreas(A, b, n):
    cVac, cInt, aVac, aInt = [], [], [], []
    for i in range(len(b)):
        if b[i][2] != 0:
            (cVac if np.dot(b[i], n[i]) > 0 else cInt).append(A[i])
        else:
            (aVac if np.dot(b[i], n[i]) > 0 else aInt).append(A[i])
    return map(np.mean, [cVac, cInt, aVac, aInt])


# ------------------------------------------------------------------------ #
#                          Main Processing Loop
# ------------------------------------------------------------------------ #
for folder_name in folders:
    folder = os.path.join(sim_root, folder_name)
    if not os.path.isdir(folder):
        print(f"Skipped {folder} (not a directory)")
        continue

    print(f"Processing {folder}")
    matfile = os.path.join(folder, 'inputFiles', 'Zr3.txt')
    DoseRate = get_scalar(matfile, 'doseRate_dpaPerSec')
    mu0_SI = get_scalar(matfile, 'mu0_SI')
    rho_SI = get_scalar(matfile, 'rho_SI')
    b_SI = get_scalar(matfile, 'b_SI')
    v_dd2SI = np.sqrt(mu0_SI / rho_SI)
    t_dd2SI = b_SI / v_dd2SI

    # Get loop data
    t, R, A = getAverageLoopData(folder)
    _, A2, b2, n2 = getLoopData(folder)

    # Filter and classify loop types
    cVac, cInt = [], []
    for i in range(len(A2)):
        cV, _, _, aI = filteredAverageAreas(A2[i], b2[i], n2[i])
        cVac.append(cV)
        cInt.append(aI)

    # Convert to radius (nm)
    def to_radius(area): return np.sqrt(np.array(area) * b_SI**2 * 1e18 / np.pi)
    dose = t * t_dd2SI * DoseRate

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dose, to_radius(cVac), label='c-Vacancy', color='blue')
    ax.plot(dose, to_radius(cInt), label='a-Interstitial', color='red')
    ax.plot(dose, np.sqrt(np.array(A) * b_SI**2 * 1e18 / np.pi), label='Total Avg Radius', color='black', linestyle='--')

    ax.set_xlabel('Irradiation Dose [dpa]', fontsize=14)
    ax.set_ylabel('Average Loop Radius [nm]', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle=':')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.savefig(os.path.join(folder, 'LoopRadius_vs_DPA.pdf'))
    print(f"Saved: {os.path.join(folder, 'LoopRadius_vs_DPA.pdf')}")
    plt.close(fig)