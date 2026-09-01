Generate a publication-quality journal article (manuscript) with the title: "A Self-Consistent Spatially-Resolved Cluster Dynamics Model for Irradiated Zirconium"  inside the directory "Docs\Formulation\self-consistent\SC_manuscript". Keep the bibiliography file there. Also, have a subdirectory inside "Docs\Formulation\self-consistent\SC_manuscript" named \figures\ where you save all needed figures, gathered from the three sources below. The article is to be submitted to the Journal of Nuclear Materials. The language is American English.  The author list is taken from a"Docs\Formulation\self-consistent\expanded_self_consistent\self_consistent_SRCD.tex". 

The manuscript is to be composed of content from three main sources: 
(1) Docs\Formulation\self-consistent\expanded_self_consistent\self_consistent_SRCD.tex, 
(2) Docs\Formulation\key_references\Zirconium_Cluster_Dynamics.pdf (Docs\Formulation\key_references\Zr_CD_main.tex). 
(3) The DisloCluster Code. This is the source of truth.

The manuscript is to be in Latex (using Latex Workshop - inside vscode). The total length of the manuscript is around 35-45 pages, including references and appendices. The structure of the manuscript is as follows:

1. Abstract
2. Introduction:
    Draw from sources 2 (general) and 1 (more specific, emphasize the importance of spatially-resolved cluster dynamibs (SR-CD): breakdown of mean-field CD, effects of spatial gradients near gb or sinks, a bridge to discrete models of defects, etc.  Use references from a bib file: self-consistent.bib, where references are gathered from Docs\Formulation\self-consistent\expanded_self_consistent\self_consistent_SRCD.bbl and Docs\Formulation\key_references\ClusterDynamicsRefs.bib)
3. Defect Classes & Reactions
    Describe the defect classes in the general way as in source 2, adding any missing classes from 1. Summarize their reaction rates (absorption, emission/ dissociation, coalescence, etc.). From the more general, describe the reduced set of defects that is actually modeled in source 3.
4. Cluster Dynamics Formulation
    Start with the general master equation and RAG formulation in source 2. Reduce this formulation to the spatially-resolved reduced equations from sources 1 & 3. At this point, list all equations (mobile and immobile (the three moments from Fokker-Planck)) in the same notation, noting that the mobile set has spatial diffusion terms. If possible, produce a Reaction Admissibility Graph (RAG) similar to source 2. Make sure that all terms in the equations are defined when they first appear, with equation cross-references if needed. At the end of this section, make a summary of the equations in two groups: mobile, and immobile.
5. Numerical Solution with Operator Splitting (consult source 3 for any details)
    Discuss the motivation for the operatir splitting, the near impossibility to solve the entire coupled system, the stiff nature of ODEs, etc.  Then outline the way it is implemented in source 3: summarize the FEM/iteration for the fast solve, and the collocation idea to render immobile species decoupled and amenable to be "embarassingly parall". Discuss error control by substeps, etc. Refer to typical system size and wall clock from recent runs.
6. Continuum-to-Discrete Self-Consistency
    Discuss the mean-field assumption of any Cluster Dynamics formulation, at what point it is expected to breakdown and why (e.g. spatial correlations, cluster coalescence, local stress field effects, spatial gradients near sinks and gb, etc.). Make references to this dilute limit breakdown (if there is any discussion in the literature). Another benefit of continuum-to-discrete is a direct connection to micromechanics, like dislocation dynamics and microfracture models, etc. Discuss the challange in making the continuum representation equivalent to the discrte such that simulations can be based on either continuum, hybdrid, or discrete, depending on some critical conditions, like dose, or the need to afford computational costs. Then describe the mathematical models for the conversion of continuum to discrete from source 3. Discuss how grain boundaries are modeled as sinks to both point defects and to immobile species (diffent models)

7. Input Parameter Calibration with Experiments
    Make several tables for model controls, material parameters, etc (all input), defining each term and what physics it affects. Summarize the experimental data in tables and graphically and define calibration targets.  Discuss the Baysian method of generating a set of optimized parameters, then list the values of those. Show several figures where the 0D model is calibrated to experiments.

8. Results
    Show results of simulations for the case: "Simulations\output\20260828_190015_431eb18_500nmHex_Refit_GBabsorb". The results should have the following subsections:
    8.1. Single Crystal Geometry: show figures for the FEM mesh, the orientations, and any geometry feature needed. Discuss number of nodes, degrees of freedom, ODE system size, etc.
    8.2. Mobile Defect Fields: Make motanges of mobile defect fields at early time (10^-4 dpa) and later time (10 dpa). Discuss the diffusion anisotropy and how it affects each defect field. discuss the width of the boundary layer as a consequence of the diffusion coefficient and how it changes during the transient.
    8.3. Immobile Defect Fields: Make montages at the same doses as 8.2. for the three moments. Discuss each figure and the physical significance of the spatial distribution of each moment, etc.
    8.4. Global Conservation: discuss the various conservation channels, including grain boundaries, and accumulators. Show results and figures 
    8.5. Grain Boundary Effects: Show results of defect concentrations close to grain boundaries and elaborate why this is important to predict the denuded zones whach are experimentally observed (with references).
    8.6. Volume Averages: show several figures for the evolution of volume average defect concentrations. Show two figures for comparison with experiments (density and size). Comment on the descrepency and how it can be reconciled.
    8.7. Continuum-to-Discrete Conversion: Discuss the mathematical method for converting continuum fields to discrete loops, and the self-consistency aspect of this conversion via conservation. Show results and figures of the single crystal at 10^-4, 10^-2, 1, 10 dpa. Use figures like these: "Simulations\output\20260828_190015_431eb18_500nmHex_Refit_GBabsorb\3d\loops_a1_0p01dpa_discrete.png" 

9. Discussion and Conclusions

    Based on all the results in 8. summarize the conclusions from the study and suggest the path forward.
10. Appendices

    Compile any detailed derivations here, since appendices do not count towards the total page number of 45 pages.

