import pandas as pd
from light_curve import GRB, get_dtw_distance, dtw_distance_helper, GRBData
from scipy.stats import norm
from time import perf_counter



def timeit(fn):
    '''
        function decorator for execution time of 'fn'
    '''
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = fn(*args, **kwargs)
        end_time = perf_counter()
        print(f'Execution time : {end_time - start_time:.3f}')
        return result

    return wrapper

def _round(x):
    '''
    Function to round all elements of a list to 2 digits

        params : x = list of floats
        returns : list of floats
    '''
    return [round(a,2) for a in x]


def find_mean_std(simulated_curves, grb):
    '''
    Function to find mean and sigma of DTW distribution given simulated curves
    and reference light curve

        params : simulated_curves = list of simulated curves
        params : grb = reference GRB obj
        returns : mean and standard deviation by normally approx. distance distribution
    '''
    reference = grb.get_photon_counts()
    distances = [dtw_distance_helper(reference, target)[0] for target in simulated_curves]
    
    return norm.fit(distances)  # returns mean and stddev


def simulate(grb, grb_type):

    simulated_curves_long = grb.simulate(100)  # simulation with binsize 1
    mustd_long = find_mean_std(simulated_curves_long, grb)
    
    if grb_type == 'short':
        simulated_curves_short = grb.simulate(100, 0.1)
        mustd_short = find_mean_std(simulated_curves_short, grb)

        return mustd_long, mustd_short

    return mustd_long

def find_stddev_LS(distances, mustd_long, mustd_short, grbbinsize):
    stddev = []
    for binsize, d in zip(grbbinsize, distances):
        if binsize == 0.1:
            delta = (d - mustd_short[0])/mustd_short[1]
        else:
            delta = (d - mustd_long[0])/mustd_long[1]   
        stddev.append(round(abs(delta),3))

    return stddev

def find_stddev_L(distances, mustd_long):
    stddev = []
    for d in distances:
        delta = (d - mustd_long[0])/mustd_long[1]   
        stddev.append(round(abs(delta),3))

    return stddev


@timeit
def distance_data(ref_idx : int, tar_idx : list[int]):
    '''
    Function to take nbd array and append all related info to a dataframe

        params : ref_idx = index of reference grb in 
        params : tar_idx = list of indices of target grbs
        returns : time taken to compare grbs 
    '''

    lcdf = GRB.lcdf                     # light curve data frame
    grbidx = [ref_idx, *tar_idx]

    print(f'grbidx : {grbidx}')

    # name of selected grbs
    grbnames = [lcdf['name'].iloc[i] for i in grbidx] 
    grbs = [GRB(name) for name in grbnames]
    grbbinsize = [grb.binsize for grb in grbs]

    refGRB = grbs[0]
    distance_list = [get_dtw_distance(refGRB, tarGRB) for tarGRB in grbs] # for DTW distances

    if 0.1 in grbbinsize:
        mustd_long, mustd_short = simulate(refGRB, 'short')
        stddev = find_stddev_LS(distance_list, mustd_long, mustd_short, grbbinsize)
    else:
        mustd_long = simulate(refGRB, 'long')
        stddev = find_stddev_L(distance_list, mustd_long)


    stddev[0] = None # sigma value for ref grb compared to itself

    grb_data = []

    for i, grb in enumerate(grbs):
        if i == 0:
            continue

        grb_obj = GRBData(name=grb.name,
                        refGRBname=refGRB.name,
                        index=int(grb.get_indexin_df()),
                        brightest_detector=grb.get_brightest_detector(),
                        ra=grb.ra,
                        dec=grb.dec,
                        binsize=grb.binsize,
                        t90=grb.t90,
                        distance=distance_list[i],
                        sigma=stddev[i])


        grb_data.append(grb_obj)


    return grb_data