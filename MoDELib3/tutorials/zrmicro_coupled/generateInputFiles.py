import sys, os, shutil
sys.path.append("../../python/")
from modlibUtils import *
import numpy as np

for x in ['evl','F','inputFiles']:
    if not os.path.exists(x):
        os.makedirs(x)

# ---------------- DD.txt ----------------
DDfile='DD.txt'
shutil.copy2('../../Library/DislocationDynamics/'+DDfile,'inputFiles/'+DDfile)
setInputVariable('inputFiles/'+DDfile,'useFEM','1')
setInputVariable('inputFiles/'+DDfile,'useDislocations','1')
setInputVariable('inputFiles/'+DDfile,'useInclusions','0')
setInputVariable('inputFiles/'+DDfile,'useElasticDeformation','1')
setInputVariable('inputFiles/'+DDfile,'useClusterDynamics','1')
setInputVariable('inputFiles/'+DDfile,'Nsteps','30')
setInputVariable('inputFiles/'+DDfile,'timeSteppingMethod','fixed')
setInputVariable('inputFiles/'+DDfile,'dtMax','6.95868964e+19')
setInputVariable('inputFiles/'+DDfile,'dxMax','5')
setInputVariable('inputFiles/'+DDfile,'use_velocityFilter','0')
setInputVariable('inputFiles/'+DDfile,'use_stochasticForce','0')
setInputVariable('inputFiles/'+DDfile,'glideSolverType','none')
setInputVariable('inputFiles/'+DDfile,'climbSolverType','Galerkin')
setInputVariable('inputFiles/'+DDfile,'quadPerLength','0.0001')
setInputVariable('inputFiles/'+DDfile,'alphaLineTension','1.0')
setInputVariable('inputFiles/'+DDfile,'remeshFrequency','1')
setInputVariable('inputFiles/'+DDfile,'Lmin','450')
setInputVariable('inputFiles/'+DDfile,'Lmax','500')
setInputVariable('inputFiles/'+DDfile,'outputFrequency','1')
setInputVariable('inputFiles/'+DDfile,'outputQuadraturePoints','1')

# ---------------- material ----------------
materialFile='Zr4.txt'
shutil.copy2('../../Library/Materials/'+materialFile,'inputFiles/'+materialFile)
setInputVariable('inputFiles/'+materialFile,'enabledSlipSystems','fullBasal fullPrismatic')
setInputVariable('inputFiles/'+materialFile,'doseRate_dpaPerSec','1.00000000e-07')   # match ZrMicro

# ---------------- elastic deformation ----------------
edFile='ElasticDeformation.txt'
shutil.copy2('../../Library/ElasticDeformation/'+edFile,'inputFiles/'+edFile)
setInputVector('inputFiles/'+edFile,'ExternalStress0',np.array([0.,0.,0.,0.,0.,0.]),'applied stress')

# ---------------- polycrystal ----------------
meshFile='unitCube_15K.msh'
shutil.copy2('../../Library/Meshes/'+meshFile,'inputFiles/'+meshFile)
pf=PolyCrystalFile(materialFile)
pf.absoluteTemperature=573      # match ZrMicro
pf.meshFile=meshFile
pf.grain1globalX1=np.array([1,0,0])
pf.grain1globalX3=np.array([0,0,1])
pf.boxEdges=np.array([[1,0,0],[0,1,0],[0,0,1]])
pf.boxScaling=np.array([3093,3093,3093])
pf.X0=np.array([0,0,0])
pf.periodicFaceIDs=np.array([])
pf.write('inputFiles')

# ---------------- <c> (Frank/basal, vacancy) loops : from ZrMicro 0-D ------
mf1='frankLoopsDensity.txt'
shutil.copy2('../../Library/Microstructures/'+mf1,'inputFiles/'+mf1)
setInputVariable('inputFiles/'+mf1,'targetDensity','1.483037e+21')
setInputVector('inputFiles/'+mf1,'planeIDs',np.array([0]),'')
setInputVariable('inputFiles/'+mf1,'radiusDistributionMean','3.573767e-08')
setInputVariable('inputFiles/'+mf1,'radiusDistributionStd','0e-8')
setInputVariable('inputFiles/'+mf1,'numberOfSides','20')
setInputVariable('inputFiles/'+mf1,'burgersFactor','0.5')
setInputVariable('inputFiles/'+mf1,'areVacancyLoops','1')

# ---------------- <a> (prismatic, interstitial) loops : from ZrMicro 0-D ---
mf2='aLoopsDensity.txt'
shutil.copy2('../../Library/Microstructures/'+mf2,'inputFiles/'+mf2)
setInputVector('inputFiles/'+mf2,'slipSystemIDs',np.array([6,8,10]),'a1,a2,a3 prism variants')
setInputVector('inputFiles/'+mf2,'targetDensity',np.array([2.237225e+21]*3),'per-variant [m^-3]')
setInputVector('inputFiles/'+mf2,'loopRadiusMean',np.array([4.431778e-09]*3),'[m]')
setInputVector('inputFiles/'+mf2,'loopRadiusStd',np.array([0.,0.,0.]),'')
setInputVector('inputFiles/'+mf2,'numberOfSides',np.array([36,36,36]),'')
setInputVector('inputFiles/'+mf2,'areVacancyLoops',np.array([0,0,0]),'interstitial <a> loops')
setInputVector('inputFiles/'+mf2,'ellipticityFactor',np.array([1.5,1.5,1.5]),'')

with open('inputFiles/initialMicrostructure.txt','w') as f:
    f.write('microstructureFile='+mf1+';\n')
    f.write('microstructureFile='+mf2+';\n')
print("input files written")
