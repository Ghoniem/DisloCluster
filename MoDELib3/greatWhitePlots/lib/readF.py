import subprocess
import os
import pathlib
import sys
import numpy as np
import string, os, math, sys

# Read F File
def readFfile(folder):
    F=np.loadtxt(folder +'/F/F_0.txt');
    with open(folder +'/F/F_labels.txt') as f:
        lines = f.readlines()
        for idx in range(len(lines)):
            lines[idx] = lines[idx].rstrip()
    return F,lines

# Get array from F file
def getFarray(F,Flabels,label):
    k=0;
    for line in Flabels:
        if line==label:
            return F[:,k]
        k=k+1
    return np.zeros(shape=(0,0))
