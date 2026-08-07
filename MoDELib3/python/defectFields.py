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
spec=pyMoDELib.ShearLoopIndividualSpecification()
spec.slipSystemIDs=[0,-1]
spec.loopRadii=[27.0e-8,27.0e-8]
spec.loopCenters=np.array([[200.0,0.0,0.0],[0.0,0.0,0.0]])
spec.loopSides=[10,10]
microstructureGenerator.addShearLoopIndividual(spec)

#microstructureGenerator.writeConfigFiles(0) # write evel_0.txt (optional)

# DefectiveCrystal
defectiveCrystal=pyMoDELib.DefectiveCrystal(ddBase)
defectiveCrystal.initializeConfiguration(microstructureGenerator.configIO)

# Exctract displacement and stress fields on the xy-plane through the center of the mesh
mesh=ddBase.mesh
xMax=mesh.xMax(); # vector [max(x1) max(x2) max(x3)] in the mesh
xMin=mesh.xMin(); # vector [min(x1) min(x2) min(x3)] in the mesh

n=200
x=np.linspace(xMin[0], xMax[0], num=n) # grid x-range
y=np.linspace(xMin[1], xMax[1], num=n) # grid x-range
z=0.5*(xMin[2]+xMax[2])
points = np.zeros((0, 3))

for i in range(0,x.size):
    for j in range(0,y.size):
        pnt=np.array([x[i],y[j],z])
        points=np.vstack((points, pnt))

u=defectiveCrystal.displacement(points) # displacement field at points
s=defectiveCrystal.stress(points) # stress field at points

# Place u1 ans s11 on a grid
u1=np.empty([n, n])
s11=np.empty([n, n])
k=0
for i in range(0,x.size):
    for j in range(0,y.size):
        u1[i,j]=u[k,0]
        s11[i,j]=s[k][0,0]
        k=k+1

# Plot u1
fig=plt.figure()
plt.imshow(u1,origin='lower',cmap='jet')
plt.colorbar()
plt.show()

# Plot s11
fig=plt.figure()
plt.imshow(s11,origin='lower',cmap='jet')
plt.colorbar()
plt.show()
