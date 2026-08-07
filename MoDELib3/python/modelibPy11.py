# /opt/local/bin/python3.12 modelibPy11.py
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../build/tools/pyMoDELib")
import pyMoDELib

simulationDir=os.path.abspath("../tutorials/crossSlip")
ddBase=pyMoDELib.DislocationDynamicsBase(simulationDir)

# Microstructure Generation
microstructureGenerator=pyMoDELib.MicrostructureGenerator(ddBase)

spec=pyMoDELib.ShearLoopDensitySpecification()
spec.targetDensity=5.0e12
spec.radiusDistributionMean=1.0e-07
spec.radiusDistributionStd=0.0e-8
spec.numberOfSides=20
spec.allowedGrainIDs=np.array([-1]); # set of grain IDs where loops are allowed. Use -1 for all grains
allowedSlipSystemIDs=np.array([-1]); # set of allowed slip system IDs. Use -1 for all slip systems
#microstructureGenerator.addShearLoopDensity(spec)

spec=pyMoDELib.ShearLoopIndividualSpecification()
spec.slipSystemIDs=[0,-1]
spec.loopRadii=[27.0e-8,27.0e-8]
spec.loopCenters=np.array([[200.0,0.0,0.0],[0.0,0.0,0.0]])
spec.loopSides=[10,10]
#microstructureGenerator.addShearLoopIndividual(spec)

spec=pyMoDELib.PeriodicDipoleDensitySpecification()
spec.targetDensity=5.0e13
#microstructureGenerator.addPeriodicDipoleDensity(spec)

spec=pyMoDELib.PeriodicDipoleIndividualSpecification()
spec.slipSystemIDs=[0,-1]
spec.exitFaceIDs=[1,0]
spec.dipoleCenters=np.array([[200.0,0.0,0.0],[0.0,0.0,0.0]])
spec.dipoleHeights=[100.,100.]
spec.nodesPerLine=[4,4]
spec.glideSteps=[10.,10.]
#microstructureGenerator.addPeriodicDipoleIndividual(spec)

spec=pyMoDELib.PrismaticLoopDensitySpecification()
spec.targetDensity=5.0e13
spec.radiusDistributionMean=3e-08
spec.radiusDistributionStd=0e-08
spec.allowedGrainIDs=np.array([-1]); # set of grain IDs where loops are allowed. Use -1 for all grains
allowedSlipSystemIDs=np.array([-1]); # set of slip system IDs whose Burgers vector are allowed to be the prism axis. Use -1 for all slip systems
microstructureGenerator.addPrismaticLoopDensity(spec)

spec=pyMoDELib.PrismaticLoopIndividualSpecification()
spec.slipSystemIDs=[0,7,13]
spec.loopRadii=[5e-8,2e-7,2e-8]
spec.loopCenters=np.array([[0,0,0],[500,600,500],[500,500,500]])
spec.glideSteps=[10.,10.,300.]
#microstructureGenerator.addPrismaticLoopIndividual(spec)

spec=pyMoDELib.FrankLoopsDensitySpecification()
spec.targetDensity=5.0e12
spec.radiusDistributionMean=1.0e-07
spec.radiusDistributionStd=0.0e-8
spec.numberOfSides=20
spec.areVacancyLoops=1
microstructureGenerator.addFrankLoopsDensity(spec)

spec=pyMoDELib.FrankLoopsIndividualSpecification()
spec.planeIDs=[0,-1]
spec.loopRadii=[27.0e-8,27.0e-8]
spec.loopCenters=np.array([[200.0,0.0,0.0],[0.0,0.0,0.0]])
spec.loopSides=[10,10]
spec.isVacancyLoop=[1,0]
#microstructureGenerator.addFrankLoopsIndividual(spec)

spec=pyMoDELib.StackingFaultTetrahedraDensitySpecification()
spec.targetDensity=1.0e21
spec.sizeDistributionMean=2.5e-08
spec.sizeDistributionStd=1.0e-9
#microstructureGenerator.addStackingFaultTetrahedraDensity(spec)

spec=pyMoDELib.StackingFaultTetrahedraIndividualSpecification()
spec.planeIDs=[0,1]
spec.areInverted=[0,0]
spec.sizes=[2.5e-08,2.5e-08]
spec.basePoints=np.array([[0.0,0.0,0.0],[100.0,0.0,0.0]])
#microstructureGenerator.addStackingFaultTetrahedraIndividual(spec)

spec=pyMoDELib.SphericalInclusionDensitySpecification()
spec.targetDensity=1e20;
spec.diameterLognormalDistribution_M=3.11e-8; #[m] of log-normal diameter M
spec.diameterLognormalDistribution_S=0.38; #[-] of log-normal distribution parameter S
spec.diameterLognormalDistribution_A=1.05e-8; #[m] log-normal distribution parameter A
spec.transformationEigenDistortion=[0.003,0.0,0.0,0.0,0.003,0.0,0.0,0.0,0.003] # [-] inclusion eigendistortion components 11,12,13,21,22,23,31,32,33
spec.patternVector_SI=[0.0,0.0,0.0e-7]; # [m] defines a planar pattern for the inclusions. Direction is plane normal, magnitude is plane spacing. Zero to zero for no pattern
spec.allowOverlap=0;
spec.allowOutside=0;
spec.velocityReductionFactor=1.0;
spec.phaseID=0;
#microstructureGenerator.addSphericalInclusionDensity(spec)

spec=pyMoDELib.SphericalInclusionIndividualSpecification()
spec.radii_SI=[27.0e-9,27.0e-9]
spec.centers=np.array([[200.0,0.0,0.0],[-200.0,0.0,0.0]])
spec.eigenDistortions=np.array([[1.0e-3,0.0,0.0,0.0,1.0e-3,0.0,0.0,0.0,1.0e-3],[2.0e-3,0.0,0.0,0.0,2.0e-3,0.0,0.0,0.0,2.0e-3]])
spec.velocityReductionFactors=[1.0,1.0]
spec.phaseIDs=[0,0]
spec.allowOverlap=0
spec.allowOutside=0
#microstructureGenerator.addSphericalInclusionIndividual(spec)

spec=pyMoDELib.PolyhedronInclusionIndividualSpecification()
spec.mshFile="../Library/Meshes/unitCube8_2D.msh"
spec.X0=np.array([[300.0,100.0,0.0],[-200.0,0.0,0.0]])
spec.F=np.array([[200.0,0.0,0.0],[0.0,200.0,0.0],[0.0,0.0,200.0]])
spec.eigenDistortion=np.array([[1.0e-3,0.0,0.0,0.0,1.0e-3,0.0,0.0,0.0,1.0e-3]])
spec.velocityReductionFactor=1.0
spec.phaseID=0
microstructureGenerator.addPolyhedronInclusionIndividual(spec)

microstructureGenerator.writeConfigFiles(1) # write evel_0.txt (optional)

# DefectiveCrystal
defectiveCrystal=pyMoDELib.DefectiveCrystal(ddBase)
defectiveCrystal.initializeConfiguration(microstructureGenerator.configIO)

points=np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0]])
#
disp=defectiveCrystal.displacement(points)
print(disp)

# DislocationNetwork
DN=defectiveCrystal.dislocationNetwork()
print(len(DN.loops()))
print(len(DN.loopNodes()))

meshSize=100
for loopID in DN.loops():
    loop=DN.loops().getRef(loopID)
#    meshedLoopVector=loop.meshed(meshSize)
#    for meshedLoop in meshedLoopVector:
#        disp=meshedLoop.plasticDisplacement(points)
#        print(disp)
#
#rp=ddBase.poly.grain(1)





#print(rp)

#mesh=pyMoDELib.SimplicialMesh()
#print(mesh.xMin())

#
## create the DefectsFieldsExtractor object
#dfe=DefectsFields.DefectsFieldsExtractor(simulationDir)
#
## Read a pre-existing configuration file (e.g. readConfiguration(X) reads file simulationDir/evl/evl_X.txt)
#dfe.readConfiguration(0)
#
## Alternatively, generate a new configuration using simulationDir/inputFiles/initialMicrostructure.txt
##dfe.readMicrostructure()
##dfe.writeConfiguration(0) # Optional. Write the generated configuration to file (writeConfiguration(X) writes file simulationDir/evl/evl_X.txt)
#
## grab the domain corners
#ldc=dfe.lowerDomainCorner()
#udc=dfe.upperDomainCorner()
#
## Extracting grids of values on a a y-z plane: e.g. solid angle and sigma_11
#n=200
#x=np.linspace(4*ldc[0], 4*udc[0], num=n) # grid x-range
#z=np.linspace(4*ldc[2], 4*udc[2], num=n) # grid z-range
#y=0.5*(ldc[1]+udc[1]) # grid position in y
#solidAngle=np.empty([n, n]) # grid of solid angle values
#s13=np.empty([n, n]) # grid of sigma_11 values
#for i in range(0,z.size):
#    for j in range(0,x.size):
#        solidAngle[j,i]=dfe.solidAngle(x[j],y,z[i])
#        stress=dfe.dislocationStress(x[j],y,z[i])
#        s13[i,j]=stress[0,2]
#
#fig=plt.figure()
#plt.imshow(solidAngle, origin='lower',cmap='jet')
#plt.colorbar()
#plt.show()
#
#fig=plt.figure()
#plt.imshow(s13, origin='lower',cmap='jet',vmin = -0.01,vmax = 0.01)
#plt.colorbar()
#plt.show()

