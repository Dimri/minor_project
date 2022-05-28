import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw

def dfFromFitFile(fitfile):
    '''
    function to get a DataFrame from a raw fitfile
    '''
    # convert the EVENTS header into a dataframe
    timedf = pd.DataFrame(fitfile[2].data)
    
    # change the dtype of both the columns
    timedf.PHA = timedf.PHA.astype('int32')
    timedf.TIME = timedf.TIME.astype('float')
    
    # get trigger time from the fits file
    trigtime = fitfile[0].header['TRIGTIME']
    
    # make a new translated time column
    timedf['TTIME'] = timedf.TIME - trigtime
    
    return timedf

def get_t90(grbname):
    '''
    function to get the t90 value of a given grb from the cleaned 
    gbm dataset
    grbname = name of the grb 
    '''
    lcdf = pd.read_csv('data/gbmdatacleaned.csv', index_col = 0)
    result = lcdf.loc[lcdf.name == grbname] # find the row with name = grbname
    t90 = result.t90 # this is an object
    return float(t90.iloc[0].strip()) # strip to remove the trailing whitespaces

# function to calculate binsize
binsize = lambda t90 : 1 if t90 >= 2 else 0.1

def getLightCurve(grb, size = 0):
    '''
    function to plot histogram and get the photon counts/ light curve
    from the raw fitfile
    grb = fitfile of the GRB
    size = binsize ; default = 0 -> calculate binsize according to t90
    '''
    # create the dataframe with time and ttime
    df = dfFromFitFile(grb)
    # get t90
    grbname = grb[0].header['OBJECT']
    t90 = get_t90(grbname)
    # start, end 
    start, end = -10, t90 + 10
    # binsize
    if size == 0:
        size = binsize(t90)
    # plot hist and get count value
    N = plt.hist(df.TTIME, bins=int((end - start)/size), range=(start,end),
                 alpha=0.5, label=grbname)
    plt.legend(loc='upper right')
    plt.close()
    return N[0]

def getDTW(reference, target, ref_name, tar_name, show_plot = False):
    '''
    function to calculate DTW distance between reference and target
    light curve
    reference = reference light curve
    target = target light curve
    ref_name = GRB name of referene LC
    tar_name = GRB name of target LC
    show_plot = shows DTW plot 
    '''
    # distance metric
    manhattan_distance = lambda x, y : np.abs(x-y)
    
    # calculate distance
    d, cost_matrix, acc_cost_matrix, path = dtw(reference, target, dist=manhattan_distance)

    # visualizing the plot
    if show_plot:
        plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
        plt.plot(path[0], path[1], 'w')
        # t90 values
        t90_1 = get_t90(ref_name)
        t90_2 = get_t90(tar_name)
        print(t90_1, t90_2)
        plt.axvline(t90_1, color='r')
        plt.axhline(t90_2, color='b')
        plt.show()
    
    return d