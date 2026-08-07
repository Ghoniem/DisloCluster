import sys, string, os
import numpy as np

class EVL:
    # Node Data
    nodesPos=np.empty([0,0])
    nodesV=np.empty([0,0])
    nodesVOld=np.empty([0,0])
    nodesClimbVScalar=np.empty([0,0])
    
    # Loop Data
    loopsNumber=np.empty(0)
    loopsArea=np.empty(0)
    loopsBurger=np.empty([0,0])
    loopsNormal=np.empty([0,0])
    loopsPos=np.empty([0,0])

    # Loop Node Data
    loopNumber=np.empty(0)
    loopNodeNumber=np.empty(0)
    loopNodePos=np.empty([0,0])

def readEVLtxt(filename):
    try: 
        with open(filename+'.txt', "r") as evlFile:
            numNodes=int(evlFile.readline().rstrip())
            numLoops=int(evlFile.readline().rstrip())
            numLinks=int(evlFile.readline().rstrip())
            numLoopNodes=int(evlFile.readline().rstrip())
            numSpInc=int(evlFile.readline().rstrip())
            numPolyInc=int(evlFile.readline().rstrip())
            numPolyIncNodes=int(evlFile.readline().rstrip())
            numPolyIncEdges=int(evlFile.readline().rstrip())
            numEDrow=int(evlFile.readline().rstrip())
            numCDrow=int(evlFile.readline().rstrip())
        
            evl=EVL();
            
            evl.nodesPos=np.empty([numNodes, 3])
            evl.nodesV=np.empty([numNodes, 3])
            evl.nodesVOld=np.empty([numNodes, 3])
            evl.nodesClimbVScalar=np.empty([numNodes, 2])

            evl.loopsNumber=np.empty(numLoops)
            evl.loopsArea=np.empty(numLoops)
            evl.loopsBurger=np.empty([numLoops, 3])
            evl.loopsNormal=np.empty([numLoops, 3])
            evl.loopsPos=np.empty([numLoops, 3])

            evl.loopNumber=np.empty(numLoopNodes)
            evl.loopNodeNumber=np.empty(numLoopNodes)
            evl.loopNodePos=np.empty([numLoopNodes,3])

            # node data
            for k in range(numNodes):
                data = np.fromstring(evlFile.readline().strip(), sep=' ')
                evl.nodesPos[k, :] = data[1:4]
                evl.nodesV[k, :] = data[4:7]
                evl.nodesClimbVScalar[k, :] = data[7:9]
                
            # loop data
            for k in range(numLoops):
                data = np.fromstring(evlFile.readline().strip(), sep=' ')
                evl.loopsNumber[k] = data[0]
                evl.loopsArea[k] = data[-1]
                evl.loopsBurger[k, 0:3] = data[1:4]
                evl.loopsNormal[k, 0:3] = data[4:7]
                evl.loopsPos[k, 0:3] = data[7:10]

            # loop links
            for k in range(numLinks):
                data = np.fromstring(evlFile.readline().strip(), sep=' ')

            # loop nodes
            for k in range(numLoopNodes):
                data = np.fromstring(evlFile.readline().strip(), sep=' ')
                evl.loopNumber[k]=data[0]
                evl.loopNodeNumber[k]=data[1]
                evl.loopNodePos[k, 0:3]=data[2:5]

        return evl

    except Exception as e:
        print(f"Error reading file {filename}: {e}")
    return None
