#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 17:00:00 2020

@author: jianlongyuan

Modified from SeisMan at:
    blog.seisman.info/radiation-pattern-and-beach-ball/#p-

"""

import math
import numpy as np
import matplotlib.pyplot as plt
import numba as nb

#%%-- plot 
def subPlotWavePattern( maxAzimuth, maxOffAngle, ampP, ampSV, ampSH ): 
    #-- Figure: One large figure including three subfigures
    fig = plt.figure( constrained_layout=True, figsize=(8,4))
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)
    gs0 = fig.add_gridspec(1, 1 )
    gs00 = gs0[0].subgridspec(1,3)
    ax1 = fig.add_subplot(gs00[0, 0], projection='polar' )
    ax2 = fig.add_subplot(gs00[0, 1], projection='polar' )
    ax3 = fig.add_subplot(gs00[0, 2], projection='polar' )
    
    #-- P
    val = ampP
    azs = np.arange( 0, maxAzimuth, 1 )/180*np.pi
    rs  = np.arange( 0, 91, 1 )
    r, theta = np.meshgrid( rs, azs ) 
    ax1.contourf( theta, r, val, colors=['white', 'black'], levels=0 )
    ax1.contour( theta, r, val, colors=['black', ], levels=0 )
    ax1.set_xticks( np.deg2rad( np.arange( 0, 360, 30 ) ) )
    ax1.set_axisbelow(True)
    ax1.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax1.set_theta_zero_location( 'N' )
    ax1.set_theta_direction( -1 )
    ax1.tick_params(axis='both', labelsize=14 )
    ax1.set_yticks([]) 
    ax1.set_title( 'P', pad=20, fontsize=16 )
    ax1.grid( ls='--', lw=0 )
    
    #-- SV
    val = ampSV
    azs = np.arange( 0, maxAzimuth, 1 )/180*np.pi
    rs  = np.arange( 0, 91, 1 )
    r, theta = np.meshgrid( rs, azs ) 
    ax2.contourf( theta, r, val, colors=['white', 'black'], levels=0 )
    ax2.contour( theta, r, val, colors=['black', ], levels=0 )
    ax2.set_xticks( np.deg2rad( np.arange( 0, 360, 30 ) ) )
    ax2.set_axisbelow(True)
    ax2.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax2.set_theta_zero_location( 'N' )
    ax2.set_theta_direction( -1 )
    ax2.tick_params(axis='both', labelsize=14 )
    ax2.set_yticks([])
    ax2.set_title( 'SV', pad=20, fontsize=16 )
    ax2.grid( ls='--', lw=0 )
    
    #-- SH
    val = ampSH
    azs = np.arange( 0, maxAzimuth, 1 )/180*np.pi
    rs  = np.arange( 0, 91, 1 )
    r, theta = np.meshgrid( rs, azs ) 
    ax3.contourf( theta, r, val, colors=['white', 'black'], levels=0 )
    ax3.contour( theta, r, val, colors=['black', ], levels=0 )
    ax3.set_xticks( np.deg2rad( np.arange( 0, 360, 30 ) ) )
    ax3.set_axisbelow(True)
    ax3.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax3.set_theta_zero_location( 'N' )
    ax3.set_theta_direction( -1 )
    ax3.tick_params(axis='both', labelsize=14 )
    ax3.set_yticks([])
    ax3.set_title( 'SH', pad=20, fontsize=16 )
    ax3.grid( ls='--', lw=0 )
    plt.show()



#-- Calculate wave pattern
@nb.jit(nopython=False)
def subCalWavePattern( strike, dip, rake, verbose ):
    maxAzimuth = 361
    maxOffAngle= 91
    m = np.zeros((3,3))
    DEG2RAD = np.pi/180.0
    strike *= DEG2RAD
    dip    *= DEG2RAD
    rake   *= DEG2RAD    

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
            
#    #--
#    if verbose == 1:
#        print( '\n------------------------------------------------------------')
#        print( '    /', "{:+.2f}".format(m[0][0]), "{:+.2f}".format( m[0][1]), "{:+.2f}".format( m[0][2]), '\\' )
#        print( 'M = |', "{:+.2f}".format(m[1][0]), "{:+.2f}".format( m[1][1]), "{:+.2f}".format( m[1][2]),  '|'  )
#        print( '    \\', "{:+.2f}".format( m[2][0]), "{:+.2f}".format( m[2][1]), "{:+.2f}".format( m[2][2]), '/' )
#        #--
#        print( '           |ampP: ', "{:+.2f}".format(np.min(ampP)), "{:+.2f}".format(np.max(ampP)) )        
#        print( 'Min, max = |ampSV:', "{:+.2f}".format(np.min(ampSV)), "{:+.2f}".format(np.max(ampSV)) ) 
#        print( '           |ampSH:', "{:+.2f}".format(np.min(ampSH)), "{:+.2f}".format(np.max(ampSH)) )
#        print( '------------------------------------------------------------\n')
#
#    if verbose == 1:
#        subPlotWavePattern( maxAzimuth, maxOffAngle, ampP, ampSV, ampSH )
          
    return ampP, ampSV, ampSH