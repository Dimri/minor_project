import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from astropy.io import ascii
from astropy.table import Table
from astropy.io import fits
from numpy import arange

import angularDistance

def angle_to_grb(ra, dec, trigdat_file, verbose = False):
    #Co-ordinates of the object
    ra_obj =  ra #171.166#in degrees
    dec_obj =  dec #49.45 #in degrees

    #Get the spacecraft pointing 
    #Event file name of trigdat
    event_filename = trigdat_file
    #Open the fits file 
    pha_list = fits.open(event_filename, memmap=True)


    #Get info of the fits file
    ra_scx = pha_list[0].header['RA_SCX']
    dec_scx = pha_list[0].header['DEC_SCX']
    ra_scz = pha_list[0].header['RA_SCZ']
    dec_scz = pha_list[0].header['DEC_SCZ']

    #The detectors of Fermi satellite
    det = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1','LAT-LLE','LAT']

    angls={}
    val_ang=[]
    for d in det:
        angle = angularDistance.getDetectorAngle(ra_scx, dec_scx, ra_scz, dec_scz, ra_obj, dec_obj,
                                                              d)
        angls[round(angle)]=d
        val_ang.append(round(angle))
        if verbose:
            print(d,round(angle))
    brightest_det = angls[np.min(val_ang)] 

    #Sorting the dictionary to select only NaI and BGO detectors into separate dictionaries
    dict_nai={}
    dict_bgo={}
    for key, val in angls.items():
            if val in ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb']:dict_nai[key]=val
            else:
                if val in ['b0','b1']:dict_bgo[key]=val

    #Sort the NaI and BGO dictionary in ascending order
    l_nai = list(dict_nai.items())
    l_nai.sort() #sorting in ascending order
    angls_nai = dict(l_nai)
    l_bgo = list(dict_bgo.items())
    l_bgo.sort() #sorting in ascending order
    angls_bgo = dict(l_bgo)
    #Get all the keys of the sorted NaI and BGO dictionaries
    res_nai = list(angls_nai.keys())
    res_bgo = list(angls_bgo.keys())
    if verbose:
        print('The brightest NaI detector is',angls_nai[res_nai[0]],'- Source angle is:',res_nai[0],'deg')
        print('The brightest 3 NaI detectors are',angls_nai[res_nai[0]],'(',res_nai[0],'deg)',angls_nai[res_nai[1]],'(',res_nai[1],'deg)',angls_nai[res_nai[2]],'(',res_nai[2],'deg)')
        print('The brightest BGO detector is',angls_bgo[res_bgo[0]],'(',res_bgo[0],'deg )')
    brightest_nai = angls_nai[res_nai[0]]
    bright_nais = [angls_nai[res_nai[0]],angls_nai[res_nai[1]],angls_nai[res_nai[2]]]
    brightest_bgo = [angls_bgo[res_bgo[0]]]
    return(brightest_nai,bright_nais,brightest_bgo)