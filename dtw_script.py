import pandas as pd
import numpy as np
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from dtw import dtw
from scipy.stats import poisson, norm
import time
import pickle

import light_curve as lc
import estimate_source_angles_detectors as esad


def simulate(fitfile, grbname, num, size = 0):
    '''
    function to simulate light curves with poisson distribution
    fitfile = GRB to simulate
    size = binsize of GRB
    num = number of simulated curves
    returns simulated curves
    '''
    reference = lc.getLightCurve(fitfile, grbname, size)
    simcurves = []  # for simulated curves
    for i in range(num):
        curve = []
        for n in reference:
            curve.append(poisson.rvs(mu = n, size = 1)[0])
        simcurves.append(curve)
    return simcurves


def brightest_detectors(ra, dec, grbnumbers):
    '''
    function to find the name of the brightest NaI detector for a particular GRB
    ra = list of ra values 
    dec = list of dec values
    grbnumbers = list of numbers of grbs
    returns list of brightest detectors
    '''
    detectors = []  # brightest detector name
    for i, number in enumerate(grbnumbers):
        dig = 0  # suffix digit _v00 or _v01 or v_02 ...
        path = ''
        while not os.path.exists(path):
            path = 'C:/Users/hhsud/Downloads/GRBS/glg_trigdat_all_bn' + number + '_v0' + str(dig) + '.fit'
            dig += 1
            if dig > 100:
                raise FileNotFoundError
        b = esad.angle_to_grb(ra[i], dec[i], path)
        detectors.append(b[0])
    
    return detectors


def _round(x):
    '''
    function to round all elements of a list to 2 digits
    '''
    return [round(a,2) for a in x]


def find_mean_std(simulated_curves, referenceLC):
    '''
    function to find mean and sigma of DTW distribution given simulated curves
    and reference light curve.
    '''
    distances = []
    for curve in simulated_curves:
        distances.append(lc.getDTW(referenceLC, curve))

    #find distribution of dtw distances of simulated light curves
    # plt.hist(distances, density=True, color='g')
    # xmin, xmax = plt.xlim()
    # x = np.linspace(xmin, xmax, 100)
    # p = norm.pdf(x, mu, std)
    # plt.plot(x, p, 'k', linewidth=2)
    
    return norm.fit(distances)  # returns mean and stddev


def distance_data(nbd_info):
    '''
    function to take nbd array and append all related info to a dataframe
    nbd_info = neighborhood array with a center grb index and a list of nbd grbs
    '''
    start_time = time.perf_counter()
    #-----get the ra and dec values of grbs to compare-----
    lcdf = pd.read_csv('data/gbmdatacleaned.csv', index_col=0)  # light curve data frame

    grb_index = nbd_info[1].copy()  # copy neighborhood indices list
    grb_index.insert(0, nbd_info[0])  # insert the center index at 0th pos

    # ra-dec, name of selected grbs
    ra = [] 
    dec = [] 
    grbnames = [] 
    for i in grb_index:
        ra.append(lcdf.iloc[i].ra_val)
        dec.append(lcdf.iloc[i].dec_val)
        grbnames.append(lcdf.name.iloc[i])
    
    ra = _round(ra)
    dec = _round(dec)
    
    grbnumbers = [x[3:] for x in grbnames] # GRB number
    grbt90 = [lc.get_t90(name) for name in grbnames] # t90's
    grbbinsize = [lc.binsize(t90) for t90 in grbt90] # bin sizes

    detectors = brightest_detectors(ra, dec, grbnumbers)
    
    #-----open the fits file corresponding to the brightest NaI detectors-----
    fitfiles = []
    for detector, number in zip(detectors, grbnumbers):
        # try suffix _v01 or _v00
        dig = 0
        path = ''
        while not os.path.exists(path):
            path = 'C:/Users/hhsud/Downloads/GRBS/glg_tte_' + detector + '_bn' + number + '_v0' + str(dig) + '.fit'
            dig += 1
            if dig > 100:
                print(path)
                raise FileNotFoundError
        
        fitfile = fits.open(path)
        fitfiles.append(fitfile)

    #-----make DTW distane list-----
    ref_name = grbnames[0]
    referenceLC = lc.getLightCurve(fitfiles[0], ref_name) # reference fit file
    distance_list = [] # for DTW distances
    for fitfile, tar_name in zip(fitfiles, grbnames): 
        targetLC = lc.getLightCurve(fitfile, tar_name) # target fit file
        d = lc.getDTW(referenceLC, targetLC, ref_name, tar_name) # get DTW distance
        distance_list.append(round(d,3))


    #-----simulate the light curves-----
    simulated_curves_big = simulate(fitfiles[0], ref_name, 100)  # simulation with binsize 1
    mu_big, sigma_big = find_mean_std(simulated_curves_big, referenceLC)
    
    if 0.1 in grbbinsize:
        simulated_curves_small = simulate(fitfiles[0], ref_name, 100, 0.1)
        mu_small, sigma_small = find_mean_std(simulated_curves_small, referenceLC)
        # print(f'mu : {mu_small}, sigma : {sigma_small}')

    # number of stdev away from mean
    stddev = []
    for size, d in zip(grbbinsize, distance_list):
        # if binsize is 0.1
        if size == 0.1:
            delta = round((d - mu_small)/sigma_small,3)
        else:
            delta = round((d - mu_big)/sigma_big,3)
        stddev.append(abs(delta))
            
    stddev[0] = None # sigma value for ref grb compared to itself
    
    isref = [1] + [0] * (len(grbnames) - 1) # boolean array ; 1 if GRB is a reference GRB
    datadf1 = pd.read_csv('data/distance_dat_file.csv')
    datadf = pd.DataFrame({'Index':grb_index, 'Is Reference': isref, 'Name':grbnames,
                           'Brightest Detector':detectors, 'ra': ra, 'dec':dec,
                           't90':grbt90, 'Binsize':grbbinsize, 'Distance':distance_list,
                           'Sigma':stddev})

    # concat dataframes and reset index
    new_datadf = pd.concat([datadf1, datadf]).reset_index(drop=True)
    # save new dataframe
    new_datadf.to_csv('data/distance_dat_file.csv', index=False)
    end_time = time.perf_counter()
    delta_time = end_time - start_time
    print(f'Time = {delta_time}')
    return delta_time


