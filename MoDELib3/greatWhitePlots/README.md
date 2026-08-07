MoDELib2-NNL/
├── greatWhitePlots/                     # Working directory for analysis scripts and outputs
│   ├── sims/                            # Contains all simulation output folders
│   │   ├── simA/                        # Individual simulation case
│   │   │   ├── inputFiles/Zr3.txt       # Material parameters
│   │   │   ├── F/F_0.txt                # Simulation output (e.g., strain, densities)
│   │   │   └── ...                      # Additional output
│   │   ├── simB/
│   │   └── simC/
│   ├── plotGrowthSwellingDensity.py     # Script for plotting strain and swelling and density 
│   ├── plotLoopRadii.py                 # Script for plotting loop radii evolution of a and c
│   └── ...                              # Additional analysis scripts or outputs
├── src/                                 
└── tutorials/                           


plot_strains_with_swelling.py
    Purpose:
    Plots the evolution of the plastic strain tensor components and isotropic swelling strain over irradiation dose for each simulation.

    Details:
    Plots the principal strain components:
        Red: beta_11
        Green: beta_22
        Blue: beta_33
        Gray: Swelling Strain 

    Plots the total dislocation density as a function of irradiation dose.
    
    Output:
    A plot saved to each simulation folder as:
    StrainComponentsWithSwelling.pdf
    DislocationDensities_vs_DPA.pdf



plot_loop_radii.py
    Purpose:
    Plots the average dislocation loop radius over irradiation dose for each simulation, broken down by loop type.

    Details:
    Computes the radius from loop area assuming circular geometry:
        Blue: Vacancy-type loops
        Red: Interstitial-type loops
        Black dashed: Total average loop radius

    Output:
    A plot saved to each simulation folder as:
    LoopRadius_vs_DPA.pdf