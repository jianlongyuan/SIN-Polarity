#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in 2025

@author: huilian Ma

    
"""


import pandas as pd
import fnmatch, os
import numpy as np
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import obspy
plt.rcParams["font.family"] = "Times New Roman"
    


#%%-- Read CSV file of first arrival time information  
filePath = './templates.csv' 
data = pd.read_csv(filePath)
net = data['network']
sta = data['station']
onsetP = data['onsetP']

#-- P-wave polarity
filePath_pol = './StationinformationPhasenet.csv' 
data_pol = pd.read_csv(filePath_pol)
p_pol = data_pol['p_POL(Z)']
sta_pol = data_pol['StName']

#-- Read the mseed data file containing three components
dataPath = './dataENZ/'
nFiles = fnmatch.filter( sorted(os.listdir(dataPath)), '*.mseed') 
nFiles = sorted(nFiles, key=lambda x: x.split('.')[1], reverse=True)

d = 0 #-- Parameter used to set the vertical spacing of the waveform

fig = plt.figure( constrained_layout=True, figsize=(7,22))
fig.subplots_adjust(hspace=0.1)
fig.subplots_adjust(wspace=0.0)
gs0 = fig.add_gridspec(1, 1)  
ax0 = fig.add_subplot(gs0[0, 0])
for idx,iFile in enumerate( nFiles ): 
  
  # Draw the waveform in parts
  if idx >= 19:
    
    ENZ = obspy.read(dataPath + iFile)  
    
    starttime = ENZ[0].stats.starttime
    delta = ENZ[0].stats.delta  
    network = ENZ[0].stats.network
    station = ENZ[0].stats.station
    channelZ = ENZ[2].stats.channel   
    S_R = ENZ[0].stats.sampling_rate
    
    # vertical component
    filenameZ = str(network)+'.'+str(station)+'.'+str(channelZ)+'.sac'
    stZ = obspy.read(dataPath+filenameZ)
    
    stZno = stZ.copy() # Copy data to extract noise segment and protect the original data
    stZsi = stZ.copy() # Copy data to extract signal segment and protect the original data
    
    stZ2 = stZ.copy() # Copy data for drawing, protect original data
    stZ22 = stZ.copy() # Copy data for drawing, protect original data
    
    # P-wave arrival 
    for i in range(len(data)):
        if net[i] == network and sta[i] == station:
            ENZonsetP = onsetP[i]
            
    # P-wave polarity  
    for j in range(len(data_pol)):
        if sta_pol[j] == station:
            P_pol = p_pol[j]
            if P_pol == -1:
                polZ = 'D'
            else:
                polZ = 'U'
    
    # noise segment and signal segment
    Znosiebeg  = ENZonsetP-1.0
    Znosieend  = ENZonsetP
    Zsignalbeg = ENZonsetP
    Zsignalend = ENZonsetP+1.0
    
    Znoisewave  = stZno.trim( starttime+Znosiebeg, starttime+Znosieend )
    Zsignalwave = stZsi.trim( starttime+Zsignalbeg, starttime+Zsignalend ) 
    yZ = Zsignalwave[0].data / max(np.fabs(stZ2[0].data)) 
    
    # The range of the drawn waveform
    stZ22 = stZ22.trim( starttime+Zsignalbeg-0.3, starttime+Zsignalbeg+0.5 ) 
    dataZ = stZ22[0].data / max(np.fabs(stZ22[0].data))  
    xZ = np.arange(0, len(dataZ), 1)
                                
    # Period of the generated sine wave template
    height_waveZ = Znoisewave[0].data / max(np.fabs(stZ2[0].data)) 
    heightZ = np.max( np.fabs( height_waveZ ) )
    
    posPeakIdxZ, _ = find_peaks( yZ, heightZ, distance=int(delta/delta) )
    negPeakIdxZ, _ = find_peaks( yZ*-1, heightZ, distance=int(delta/delta) )
    periodZ = 2.0*abs( (posPeakIdxZ[0]-negPeakIdxZ[0])*delta )
    
    # Sine Wave Template
    frequency = 1 / periodZ  
    t = np.linspace(0, periodZ, int(S_R * periodZ))  
    temWaveZ = 1 * np.sin(2 * np.pi * frequency * t) 
    total_nptsZ = len(temWaveZ)
    
    # Index position of the two extreme points
    if posPeakIdxZ[0] < negPeakIdxZ[0]:
        aZ = [posPeakIdxZ[0], negPeakIdxZ[0]]
    else:
        aZ = [negPeakIdxZ[0], posPeakIdxZ[0]]
    
    # The position where the template matches
    xSinZ = np.arange(0, len(temWaveZ), 1)+(0.3/delta+aZ[0]-(periodZ/4)/delta)
    
    ax0.plot(xZ*delta, dataZ+d, lw=1.5, c='black')     
    ax0.plot(xSinZ*delta, temWaveZ+d, lw=1.5, c='blue') 
    
    ax0.vlines( 0.3+aZ[0]*delta, ymin=-1+d, ymax=1+d, linestyle='--', colors='blue', lw=1.2) 
    ax0.vlines( 0.3+aZ[1]*delta, ymin=-1+d, ymax=1+d, linestyle='--', colors='blue', lw=1.2) 
    ax0.vlines( 0.3, ymin=-0.7+d, ymax=0.7+d, linestyle='--', colors='red', lw=1.2) #phasenet 
    ax0.text(0, d+0.3, station, va="center", fontsize=24)
    ax0.text(0.2, d+0.3, polZ, va="center", fontsize=24)
    
    d += 3 
        
plt.xlabel("Time(s)", fontsize=26)
plt.tick_params(axis='x', labelsize=26)  
plt.ylim(-1.5, 58.5)
ax0.set_yticks([])

ax0.spines['right'].set_color('none')
ax0.spines['left'].set_color('none')
ax0.spines['top'].set_color('none')

plt.tight_layout()

corrfigurepath = './figure_polZ/'
if not os.path.exists(corrfigurepath):
    os.makedirs(corrfigurepath )
plt.savefig(corrfigurepath+'AllStations1.png', dpi=300)
plt.show()
        
        
    
    
    
                        
                

                
                
                


    







        
    







