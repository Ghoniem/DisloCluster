# /opt/local/bin/python3.12 test.py
import sys, string, os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
sys.path.append("../../python")
from modlibUtils import *
sys.path.append("../../build/tools/pyMoDELib")
import pyMoDELib

compute=True
#periodicImageCases=np.array([[1,1,1,0],[1,1,1,1],[2,2,2,2],[4,4,4,2]])
periodicImageCases=np.array([[4,4,4,2]])

if compute:
    ###########################################################
    ## Preparing input files
    folders=['evl','F','inputFiles']
    for x in folders:
        if not os.path.exists(x):
            os.makedirs(x)

    # Make a local copy of DD parameters file and modify that copy if necessary
    DDfile='DD.txt'
    DDfileTemplate='../../Library/DislocationDynamics/'+DDfile
    print("\033[1;32mCreating  DDfile\033[0m")
    shutil.copy2(DDfileTemplate,'inputFiles/'+DDfile)
    setInputVariable('inputFiles/'+DDfile,'useFEM','0')
    setInputVariable('inputFiles/'+DDfile,'useDislocations','1')
    setInputVariable('inputFiles/'+DDfile,'useInclusions','0')
    setInputVariable('inputFiles/'+DDfile,'useElasticDeformation','0')
    setInputVariable('inputFiles/'+DDfile,'useClusterDynamics','0')
    setInputVariable('inputFiles/'+DDfile,'timeSteppingMethod','adaptive') # adaptive or fixed
    setInputVariable('inputFiles/'+DDfile,'dtMax','1e25')
    setInputVariable('inputFiles/'+DDfile,'dxMax','1') # max nodal displacement for when timeSteppingMethod=adaptive
    setInputVariable('inputFiles/'+DDfile,'alphaLineTension','0.1') # dimensionless scale factor in for line tension forces
    setInputVariable('inputFiles/'+DDfile,'Lmin','5')  # min segment length (in Burgers vector units)
    setInputVariable('inputFiles/'+DDfile,'Lmax','20')  # max segment length (in Burgers vector units)
    setInputVariable('inputFiles/'+DDfile,'maxJunctionIterations','1')  # max segment length (in Burgers vector units)
    setInputVariable('inputFiles/'+DDfile,'remeshFrequency','0')  # max segment length (in Burgers vector units)
    setInputVariable('inputFiles/'+DDfile,'outputFrequency','1')  # output frequency
    setInputVariable('inputFiles/'+DDfile,'quadPerLength','0.05')  # output quadrature data
    setInputVariable('inputFiles/'+DDfile,'outputQuadraturePoints','1')  # output quadrature data
    setInputVariable('inputFiles/'+DDfile,'computeElasticEnergyPerLength','1')  # output quadrature data
    setInputVariable('inputFiles/'+DDfile,'glideSolverType','Galerkin')  # type of glide solver, or none
    setInputVariable('inputFiles/'+DDfile,'climbSolverType','none')  # type of clim solver, or none
    setInputVariable('inputFiles/'+DDfile,'Nsteps','1')  # number of simulation steps

    # Make a local copy of material file, and modify that copy if necessary
    materialFile='Cu.txt';
    materialFileTemplate='../../Library/Materials/'+materialFile;
    print("\033[1;32mCreating  materialFile\033[0m")
    shutil.copy2(materialFileTemplate,'inputFiles/'+materialFile)
    setInputVariable('inputFiles/'+materialFile,'enabledSlipSystems','Shockley')
    setInputVariable('inputFiles/'+materialFile,'glidePlaneNoise','none')
    b_SI=getValueInFile('inputFiles/'+materialFile,'b_SI')
    #h_SI=b_SI/np.sqrt(2.0/3.0) # plane heigh in BCC

    # Create polycrystal.txt using local material file
    meshFile='unitCube24.msh';
    meshFileTemplate='../../Library/Meshes/'+meshFile;
    print("\033[1;32mCreating  polycrystalFile\033[0m")
    shutil.copy2(meshFileTemplate,'inputFiles/'+meshFile)
    pf=PolyCrystalFile(materialFile);
    pf.absoluteTemperature=300;
    pf.meshFile=meshFile
    pf.alignToSlipSystem0=1;
    pf.boxScaling=np.array([300e-10,180e-10,400e-10])/b_SI # length of box edges in Burgers vector units
    pf.periodicFaceIDs=np.array([-1])
    pf.write('inputFiles')

    ###########################################################
    ## Iterate over configurations and run one step
    simulationDir=os.path.abspath(".")

    for k in range(0,np.size(periodicImageCases,0)):
        fFile=file_path = "F/F_0.txt"
        if os.path.exists(fFile):
            os.remove(fFile)
        fLabes=file_path = "F/F_labels.txt"
        if os.path.exists(fLabes):
            os.remove(fLabes)
        periodicImageSize=np.array(periodicImageCases[k,:3]).astype(int)
        EwaldLengthFactor=periodicImageCases[k,3]
        setInputVariable('inputFiles/'+DDfile,'EwaldLengthFactor',str(EwaldLengthFactor))
        setInputVector('inputFiles/'+DDfile,'periodicImageSize',periodicImageSize,'number of periodic images along each period vector')
        print("BEFORE DDBASE")
        ddBase=pyMoDELib.DislocationDynamicsBase(simulationDir)
        mesh=ddBase.mesh
        xMax=mesh.xMax()
        xMin=mesh.xMin()
        xCenter=mesh.xCenter()
        print("BEFORE defectiveCrystal")
        defectiveCrystal=pyMoDELib.DefectiveCrystal(ddBase)
        print("AFTER defectiveCrystal")
        spec=pyMoDELib.PeriodicDipoleIndividualSpecification()
        spec.slipSystemIDs=[0,1]
        spec.exitFaceIDs=[1,1]
        spec.dipoleCenters=np.array([xCenter,xCenter])
        spec.dipoleHeights=[60,60]
        spec.nodesPerLine=[0,0]

        X=np.linspace(0, 150, num=100)
        S11=X*0.0;
        S12=X*0.0;
        S13=X*0.0;
        S22=X*0.0;
        S23=X*0.0;
        S33=X*0.0;
        j=0
        print("BEFORE FOR LOOP")
        for x in X:
            spec.glideSteps=[x,x+7]
            print("BEFORE MICROSTRUCTURE")
            microstructureGenerator=pyMoDELib.MicrostructureGenerator(ddBase)
            print("AFTER MICROSTRUCTURE")
            microstructureGenerator.addPeriodicDipoleIndividual(spec)
            print("DONE MICROSTRUCTURE")
#            if j==0:
#                microstructureGenerator.writeConfigFiles(0) # write evel_0.txt (optional)
            defectiveCrystal.initializeConfiguration(microstructureGenerator.configIO)
            defectiveCrystal.runSingleStep()
            aux =readAUXtxt('evl/ddAux_'+str(j))
            S11[j]=np.mean(aux.gaussPoints[:,10])
            S12[j]=np.mean(aux.gaussPoints[:,11])
            S13[j]=np.mean(aux.gaussPoints[:,12])
            S22[j]=np.mean(aux.gaussPoints[:,14])
            S23[j]=np.mean(aux.gaussPoints[:,15])
            S33[j]=np.mean(aux.gaussPoints[:,18])
            j=j+1
        print("AFTER FOR LOOP")
        # Store results
        F,Flabels=readFfile('./F')
        E=getFarray(F,Flabels,'dislocation elastic energy [mu b^3]')
        data=np.empty([np.size(X),8])
        data[:,0]=X
        data[:,1]=E
        data[:,2]=S11
        data[:,3]=S22
        data[:,4]=S33
        data[:,5]=S23
        data[:,6]=S13
        data[:,7]=S12
        outFileName='data'+'_'+str(periodicImageSize[0])+'_'+str(periodicImageSize[1])+'_'+str(periodicImageSize[2])+'_'+str(EwaldLengthFactor)+'.txt'
        np.savetxt(outFileName, data)
        
###########################################################
## Load data into dictionary for plotting
dataDict=dict()
for k in range(0,np.size(periodicImageCases,0)):
    periodicImageSize=np.array(periodicImageCases[k,:3]).astype(int)
    EwaldLengthFactor=periodicImageCases[k,3]
    key=str(periodicImageSize[0])+'_'+str(periodicImageSize[1])+'_'+str(periodicImageSize[2])+'_'+str(EwaldLengthFactor)
    FileName='data_'+key+'.txt'
    dataDict[key]=np.loadtxt(FileName);
    
fields=['energy','sigma11','sigma22','sigma33','sigma23','sigma13','sigma12']
fieldLabels=['elastic energy $[\mu b^3]$','$\sigma_{11}$ $[\mu]$','$\sigma_{22}$ $[\mu]$','$\sigma_{33}$ $[\mu]$','$\sigma_{23}$ $[\mu]$','$\sigma_{13}$ $[\mu]$','$\sigma_{12}$ $[\mu]$']
for k in range(0,len(fields)):
    # plots with centered images
    fig1 = plt.figure()
    ax1=plt.subplot(1,1,1)
    for key in dataDict:
        data=dataDict[key]
        ax1.plot(data[:,0], data[:,k+1],label=key)
    ax1.grid()
    ax1.legend()
    plt.xlabel('dipole distance $[b]$')
    plt.ylabel(fieldLabels[k])
    fig1.savefig(fields[k], bbox_inches='tight')
