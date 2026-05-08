
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:53:55 2019

@author: jianlongyuan

This script is used for earthquake source mechanism inversion to obtain all solutions that meet the conditions.
Update: Combine the radiation pattern calculation part and the initial motion polarity matching part into 
        one subroutine (scanFM()) in order to greatly improve computational efficiency.
      
"""

from obspy.taup import TauPyModel
from obspy.taup import velocity_model
from obspy.taup import taup_create
import matplotlib.pyplot as plt
from obspy.geodetics.base import kilometer2degrees
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
import math
import cmath
import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis as kurt
import matplotlib.cm as cm
import subWavePattern_v1 as WP
import timeit
import os, fnmatch, sys
import pandas as pd
import pickle
import numba as nb
begin_timer = timeit.default_timer()
plt.rcParams["font.family"] = "Times New Roman"



#%%-- Used to set up the basic framework for drawing
def subPlotSettings( numSta, stAz, p_offAngle, ax1, ax2, ax3 ):
    azs = np.arange( 0, 361, 1 )/180*np.pi
    rs  = np.arange( 0, 91, 1 )
    r, theta = np.meshgrid( rs, azs ) 
    ax1.set_xticks( np.deg2rad( np.arange( 0, 360, 30 ) ) )
    ax1.set_axisbelow(True)
    ax1.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax1.set_theta_zero_location( 'N' )
    ax1.set_theta_direction( -1 )
    ax1.tick_params(axis='both', labelsize=8 )
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.grid( ls='--', lw=0 )
    ax1.set_ylim(0, 91)
    
    ax2.set_xticks( np.deg2rad( np.arange( 0, 360, 30 ) ) )
    ax2.set_axisbelow(True)
    ax2.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax2.set_theta_zero_location( 'N' )
    ax2.set_theta_direction( -1 )
    ax2.tick_params(axis='both', labelsize=8 )
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.grid( ls='--', lw=0 )
    ax2.set_ylim(0, 91)
    
    ax3.set_xticks( np.deg2rad( np.arange( 0, 360, 30 ) ) )
    ax3.set_axisbelow(True)
    ax3.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax3.set_theta_zero_location( 'N' )
    ax3.set_theta_direction( -1 )
    ax3.tick_params(axis='both', labelsize=8 )
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.grid( ls='--', lw=0 )
    ax3.set_ylim(0, 91)

    #-- Plot station, southern hemisphere projection
    for ist in range( numSta ):
        lowAz = stAz[ist] # azimuth in lower semiphere
        up2lowAz = lowAz+180 # translate azimuth from upper semiphere to lower semiphere
        if up2lowAz >= 360:  # translate azimuth from upper semiphere to lower semiphere
            up2lowAz = lowAz-180  # translate azimuth from upper semiphere to lower semiphere
        lowAz = lowAz/180*np.pi # azimuth in lower semiphere
        up2lowAz = up2lowAz/180*np.pi
        up2lowOffAngP   = 180-p_offAngle[ist]
        # up2lowOffAngsPg = 180-sPg_offAngle[ist]
                                   
        ax1.scatter( up2lowAz, up2lowOffAngP, s=30, marker='^', c='red', alpha=1, zorder=111 )
        ax2.scatter( up2lowAz, up2lowOffAngP, s=30, marker='^', c='red', alpha=1, zorder=111 )
        ax3.scatter( up2lowAz, up2lowOffAngP, s=30, marker='^', c='red', alpha=1, zorder=111 )
        # ax2.scatter( up2lowAz, up2lowOffAngsPg, s=30, marker='o', c='red', alpha=1, zorder=111 )
    return ax1, ax2, ax3, r, theta
  
      

#%%-- Used for drawing contour lines
def subPlotContours( ax1, ax2, ax3, theta, r, ampP, ampSV, ampSH, flag ):
    
    if flag == 'inv':
        ax1.contour( theta, r, ampP,  colors=['black', ], levels=[0,0.0001], linewidths=0.2 )
        ax2.contour( theta, r, ampSV, colors=['black', ], levels=[0,0.0001], linewidths=0.2 )
        ax3.contour( theta, r, ampSH, colors=['black', ], levels=[0,0.0001], linewidths=0.2 )      
        return ax1, ax2, ax3      
    else:
        ax1.contour( theta, r, ampP,  colors=['lime', ], levels=[0,0.0001], linestyles='--', linewidths=1 )
        ax2.contour( theta, r, ampSV, colors=['lime', ], levels=[0,0.0001], linestyles='--', linewidths=1 )
        ax3.contour( theta, r, ampSH, colors=['lime', ], levels=[0,0.0001], linestyles='--', linewidths=1 )      
        return ax1, ax2, ax3         



#%%   
@nb.jit(nopython=1) # Accelerate statement processing
def scanFM( strikeFrom, strikeTo, strikeInc,
            dipFrom, dipTo, dipInc,
            rakeFrom, rakeTo, rakeInc,
            numSta,
            azFinalIdxLst, offAngPIdxLst,
            p_POL_Z, pctDirect ):
    
    
    class1 = []   
    for istrike in np.arange( strikeFrom, strikeTo, strikeInc ):
        if istrike % 30 == 0: print( 'istrike =', istrike, '/', strikeTo )             
        for idip in np.arange( dipFrom, dipTo, dipInc ):
            for irake in np.arange( rakeFrom, rakeTo, rakeInc ):
                
                #------------------------------------
                #-- calculate P, SV, and SH patterns
                #------------------------------------
                strike = istrike
                dip    = idip
                rake   = irake
                maxAzimuth = 361
                maxOffAngle= 91
                m = np.zeros((3,3))
                DEG2RAD = np.pi/180.0
                strike *= DEG2RAD
                rake   *= DEG2RAD
                dip    *= DEG2RAD
                
                m[0][0] = -1 * math.sin(dip)*math.cos(rake)*math.sin(2*strike) -\
                            math.sin(2*dip)*math.sin(rake)*math.sin(strike)*math.sin(strike)
                m[0][1] = math.sin(dip)*math.cos(rake)*math.cos(2*strike) +\
                             0.5*math.sin(2*dip)*math.sin(rake)*math.sin(2*strike)
                m[0][2] = -1 * math.cos(dip)*math.cos(rake)*math.cos(strike) -\
                            math.cos(2*dip)*math.sin(rake)*math.sin(strike)
                m[1][1] = math.sin(dip)*math.cos(rake)*math.sin(2*strike) -\
                            math.sin(2*dip)*math.sin(rake)*math.cos(strike)*math.cos(strike)
                m[1][2] = -1 * math.cos(dip)*math.cos(rake)*math.sin(strike) +\
                            math.cos(2*dip)*math.sin(rake)*math.cos(strike)
                m[2][2] = math.sin(2*dip)*math.sin(rake)
                m[1][0] = m[0][1]
                m[2][0] = m[0][2]
                m[2][1] = m[1][2]
                 
                gamma = np.zeros((3))
                p     = np.zeros((3))
                phi   = np.zeros((3))   
                ampP  = np.zeros(( maxAzimuth, maxOffAngle ))
                ampSV = np.zeros(( maxAzimuth, maxOffAngle ))
                ampSH = np.zeros(( maxAzimuth, maxOffAngle ))
                
                for i in range( maxAzimuth ): # Azimuth
                    for j in range( maxOffAngle ): # take-off angle
                        az    = (float)(i) / 1.0 * DEG2RAD # Azimuth
                        theta = (float)(j) / 1.0 * DEG2RAD # take-off angle
                        
                        # P
                        gamma[0] = math.sin(theta)*math.cos(az)
                        gamma[1] = math.sin(theta)*math.sin(az)
                        gamma[2] = math.cos(theta)
                
                        # SV
                        p[0] = math.cos(theta)*math.cos(az)
                        p[1] = math.cos(theta)*math.sin(az)
                        p[2] = -1.0*math.sin(theta)
                        
                        # SH
                        phi[0] = -1.0*math.sin(az)
                        phi[1] = math.cos(az)
                        phi[2] = 0.0
                
                        amp1 = 0.0
                        amp2 = 0.0
                        amp3 = 0.0
                
                        for k in range(3):
                            for l in range(3):       
                                amp1 += gamma[k]*m[k][l]*gamma[l]
                                amp2 += p[k]*m[k][l]*gamma[l]
                                amp3 += phi[k]*m[k][l]*gamma[l]
                        
                        ampP[i][j]  = amp1
                        ampSV[i][j] = amp2
                        ampSH[i][j] = amp3
                
                
                #------------
                #-- matching
                #------------
                flag0 = 0
                
                #-- Calculate the azimuth and take-off angle indices, so that 
                #-- the corresponding amplitudes on the source beachball can be retrieved later.
                # Explanation: The following statement uses 180 minus the take-off angle because 
                # when calculating the focal mechanism solution, only the amplitude in the lower hemisphere
                # is calculated. When the take-off angle is greater than 90 degrees, the amplitude in the upper 
                #hemisphere needs to be replaced with the amplitude at the corresponding position in the lower hemisphere.         
                for ist in range( numSta ):
                    azFinalIdx = azFinalIdxLst[ist]
                    offAngPIdx = offAngPIdxLst[ist]
                    
                    #-- Inversion using only the polarity of the direct P-wave (Z component)
                    if ampP[azFinalIdx][offAngPIdx] * p_POL_Z[ist]   > 0:
                           flag0 += 1                         
                    else:  flag0 = 0

                # Judgment condition: whether the number of stations meeting the matching conditions equals the total number of 
                # stations multiplied by the receivable ratio. If equal, it indicates that the quality of the currently scanned 
                # source mechanism solution is good, and it is recorded (plotted); otherwise, the quality is poor and it is not recorded.     
                if flag0 >= numSta*pctDirect:
                    tmp = istrike, idip, irake
                    class1.append(tmp)
                
    return class1
                    


#%%
@nb.jit(nopython=1) # Accelerate statement processing
def subCalFMslns( nSlns, strikeInv, dipInv, rakeInv ):

    ampPall  = []
    ampSVall = []
    ampSHall = []
    
    for isln in range(nSlns):
        #------------------------------------
        #-- calculate P, SV, and SH patterns
        #------------------------------------
        strike = strikeInv[isln]
        dip    = dipInv[isln]
        rake   = rakeInv[isln]
        
        maxAzimuth = 361
        maxOffAngle= 91
        m = np.zeros((3,3))
        DEG2RAD = np.pi/180.0
        strike *= DEG2RAD
        rake   *= DEG2RAD
        dip    *= DEG2RAD
        
        m[0][0] = -1 * math.sin(dip)*math.cos(rake)*math.sin(2*strike) -\
                    math.sin(2*dip)*math.sin(rake)*math.sin(strike)*math.sin(strike)
        m[0][1] = math.sin(dip)*math.cos(rake)*math.cos(2*strike) +\
                     0.5*math.sin(2*dip)*math.sin(rake)*math.sin(2*strike)
        m[0][2] = -1 * math.cos(dip)*math.cos(rake)*math.cos(strike) -\
                    math.cos(2*dip)*math.sin(rake)*math.sin(strike)
        m[1][1] = math.sin(dip)*math.cos(rake)*math.sin(2*strike) -\
                    math.sin(2*dip)*math.sin(rake)*math.cos(strike)*math.cos(strike)
        m[1][2] = -1 * math.cos(dip)*math.cos(rake)*math.sin(strike) +\
                    math.cos(2*dip)*math.sin(rake)*math.cos(strike)
        m[2][2] = math.sin(2*dip)*math.sin(rake)
        m[1][0] = m[0][1]
        m[2][0] = m[0][2]
        m[2][1] = m[1][2]
         
        gamma = np.zeros((3))
        p     = np.zeros((3))
        phi   = np.zeros((3))   
        ampP  = np.zeros(( maxAzimuth, maxOffAngle ))
        ampSV = np.zeros(( maxAzimuth, maxOffAngle ))
        ampSH = np.zeros(( maxAzimuth, maxOffAngle ))
        
        for i in range( maxAzimuth ): # Azimuth
            for j in range( maxOffAngle ): # take-off angle
                az    = (float)(i) / 1.0 * DEG2RAD # Azimuth
                theta = (float)(j) / 1.0 * DEG2RAD # take-off angle
                
                # P
                gamma[0] = math.sin(theta)*math.cos(az)
                gamma[1] = math.sin(theta)*math.sin(az)
                gamma[2] = math.cos(theta)
        
                # SV
                p[0] = math.cos(theta)*math.cos(az)
                p[1] = math.cos(theta)*math.sin(az)
                p[2] = -1.0*math.sin(theta)
                
                # SH
                phi[0] = -1.0*math.sin(az)
                phi[1] = math.cos(az)
                phi[2] = 0.0
        
                amp1 = 0.0
                amp2 = 0.0
                amp3 = 0.0
        
                for k in range(3):
                    for l in range(3):       
                        amp1 += gamma[k]*m[k][l]*gamma[l]
                        amp2 += p[k]*m[k][l]*gamma[l]
                        amp3 += phi[k]*m[k][l]*gamma[l]
                
                ampP[i][j]  = amp1
                ampSV[i][j] = amp2
                ampSH[i][j] = amp3
        ampPall.append( ampP )
        ampSVall.append( ampSV )
        ampSHall.append( ampSH )
          
    return ampPall, ampSVall, ampSHall       

  

#%%#########################################
# Main
#%%############################################
outPath = './'

#-- Focal mechanism solution matching condition: whether the number of stations = 
#-- total number of stations * acceptable proportion
pctDirect  = 1
strikeFrom =0
strikeTo   =360
strikeInc  =5
dipFrom    =0
dipTo      =91
dipInc     =5
rakeFrom   =-180
rakeTo     =181
rakeInc    =5
maxAzimuth = 361
maxOffAngle= 91

nStrike  = len( np.arange( strikeFrom, strikeTo, strikeInc ) )
nDip     = len( np.arange( dipFrom, dipTo, dipInc ) )
nRake    = len( np.arange( rakeFrom, rakeTo, rakeInc ) )
nAzimuth = len( range( maxAzimuth ) )
nOffAngle= len( range( maxOffAngle ) )

        
#%%-- Statistics of Polarities of Each Seismic Phase
nFiles = fnmatch.filter( sorted(os.listdir(outPath)), 'StationinformationSTALTA.csv')

for idxFile, iFile in enumerate( nFiles ):
  if idxFile >= int( len(nFiles)*0/4) and \
     idxFile <  int( len(nFiles)*4/4):

    print( 'iFile =', iFile )
    
    inPath       = str(outPath)+str(iFile)
    data         = pd.read_csv( inPath )
    stAz         = data[ 'Az(deg)' ]
    dis          = data[ 'EpDis(km)' ]
    p_offAngle   = data[ 'p_offAng(deg)' ]
    p_POL_Z      = data[ 'p_POL(Z)' ]
    
    numSta = len( stAz )

    #-- added at 20201227
    #-- If the number of stations is less than two, 
    #-- the conditions for solving the earthquake source mechanism are not met.
    if numSta < 2:
        print( '\n\n\n Number of stations < 2! \n\n\n')
        continue
    minDis = min(dis)
    maxDis = max(dis)
    print( 'Number of stations =', numSta )
    print( 'minDis, maxDis =', minDis, maxDis )

    #-- 计算直达波和sPg震相的方位角和出射角（下半球投影方式）
    azFinalIdxLst   = []
    offAngPIdxLst   = []
    offAngsPgIdxLst = []
    for ist in range( numSta ):
        azIdx         = int( round(stAz[ist], 1) )
        offAngPIdx    = int( round(p_offAngle[ist], 1) )
        
        azFinalIdxLst.append(azIdx)
        offAngPIdxLst.append(offAngPIdx)

    #%%-- Focal Mechanism Solution
    nSlns1 = 0
    
    class1 = []
      
    #-- Match the source mechanism solutions that meet the conditions
    print( '\n========================================================')
    class1 = scanFM( strikeFrom, strikeTo, strikeInc,
            dipFrom, dipTo, dipInc,
            rakeFrom, rakeTo, rakeInc,
            numSta,
            np.array(azFinalIdxLst), np.array(offAngPIdxLst),
            np.array(p_POL_Z), pctDirect )
        
 
    print( 'Number of theorectical FM Slns =', nStrike*nDip*nRake )    
    print( 'nSlns1 =', len(class1) )
    


#%%-- output fm solutions to file
import csv 
outFile = str(outPath)+'output_FMsln.csv'
with open( '{0}'.format( outFile ), mode='w', newline=''  ) as resultsFile:
    writer = csv.writer( resultsFile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([ 'CLASS', 'STRIKE(DEG)', 'DIP(DEG)', 'RAKE(DEG)', 'pctDirect' ])
    for i in class1:
        writer.writerow([ '1', i[0], i[1], i[2], pctDirect ])



#%% calculate computing time
end_timer = timeit.default_timer()
elapsedTime = end_timer - begin_timer
print('Inversion elapsed time: ', format( elapsedTime, '.1f'),
  'sec = ', format( elapsedTime/60.0, '.1f'), 'min' )



#%%-- plot
#-- get station's info
fileName = 'StationinformationSTALTA.csv'
inPath = str(outPath)+str(fileName)
data   = pd.read_csv( inPath )
dis    = data[ 'EpDis(km)' ]
numSta = len( dis ) 
minDis = min(dis)
maxDis = max(dis)
print( 'Number of stations =', numSta )
print( 'minDis, maxDis =', minDis, maxDis )
    
#-- get inverted fm solutions
inFile = str(outPath)+'output_FMsln.csv'
fms = pd.read_csv(inFile, header=0 )

#-- Obtain the focal mechanism solutions under each constraint condition
c1 = fms.query('CLASS==1')
nSlnsC1 = len(c1)
print( 'nSlnsC1 =', nSlnsC1)

strikeC1 = np.array( c1['STRIKE(DEG)'] )
dipC1    = np.array( c1['DIP(DEG)'] )
rakeC1   = np.array( c1['RAKE(DEG)'] )
pct1C1   = np.array( c1['pctDirect'] )



#%%-- Plot all focal mechanism solutions (radiation patterns of P, SV, SH)
fig = plt.figure( constrained_layout=True, figsize=(6,3) )
fig.subplots_adjust( hspace=0.1 )
fig.subplots_adjust( wspace=0.2 )
gs0 = fig.add_gridspec( 1, 1 )
gs00 = gs0[0].subgridspec( 2, 4 )
ax0_0 = fig.add_subplot(gs00[0, 0], projection='polar' )
ax0_1 = fig.add_subplot(gs00[0, 1], projection='polar' )
ax0_2 = fig.add_subplot(gs00[0, 2], projection='polar' )
ax0_3 = fig.add_subplot(gs00[0, 3], projection='polar' )

title1 = 'P'

ax0_1, ax0_2, ax0_3, r, theta = subPlotSettings( numSta, stAz, p_offAngle,\
     ax0_1, ax0_2, ax0_3 )

ampP1, ampSV1, ampSH1 = subCalFMslns(nSlnsC1, strikeC1, dipC1, rakeC1)

for i in range(nSlnsC1):
    subPlotContours( ax0_1, ax0_2, ax0_3, theta, r, ampP1[i], ampSV1[i], ampSH1[i], 'inv' ) 

ax0_0.set_title( 'a) '+str(title1)+'\n'+'    Total: '+str(nSlnsC1), loc='left', pad=-50, fontsize=12 )

ax0_0.axis('off')
ax0_1.set_title( 'P',  pad=5, fontsize=12 )
ax0_2.set_title( 'SV', pad=5, fontsize=12 )
ax0_3.set_title( 'SH', pad=5, fontsize=12 )

#-- Output drawing
plt.tight_layout()
figName = 'dist'+str(minDis)+'-'+str(maxDis)+'_nst'+str(numSta)+\
          '_pctDirect'+str(pct1C1[0])+\
          '_Flow4_FM_Slns.png'
plt.savefig( str(outPath)+str(figName), dpi=300 )

plt.show()
  
        
#%% calculate computing time
end_timer = timeit.default_timer()
elapsedTime = end_timer - begin_timer
print('Plot elapsed time: ', format( elapsedTime, '.1f'),
  'sec = ', format( elapsedTime/60.0, '.1f'), 'min' )