import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw

def df_from_fit_file(fitfile):
    '''
    function to get a DataFrame from a raw fitfile

        params : fitfile = raw .fit file
        returns : pandas Dataframe with cols = ['TIME', 'PHA', 'TTIME']
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
    Function to get the t90 value of a given grb from the cleaned gbm dataset
        
        params : grbname = name of the grb 
        returns : t90 of given grb
    '''
    lcdf = pd.read_csv('data/gbmdatacleaned.csv', index_col = 0)
    result = lcdf.loc[lcdf['name'] == grbname] # find the row with name = grbname
    t90 = result.t90 # this is an object
    return float(t90.iloc[0].strip()) # strip to remove the trailing whitespaces

def binsize(t90):
        '''
        Function to return binsize according to T90
            params : t90 of grb
            returns : 0.1 if t90 < 2 else 1
        '''
        if t90 >= 2:
            return 1
        else:
            return 0.1


def get_light_curve(grb, grbname, size = 0, show = False):
    '''
    function to plot histogram and get the photon counts/ light curve
    from the raw fitfile
    grb = fitfile of the GRB
    size = binsize ; default = 0 -> calculate binsize according to t90
    '''

    # create the dataframe with time and ttime
    df = df_from_fit_file(grb)

    # get t90
    t90 = get_t90(grbname)

    # start, end 
    start, end = -10, t90 + 10

    # binsize
    if size == 0:
        size = binsize(t90)
        
    # get count value in bins 
    N = np.histogram(df.TTIME, bins=int((end - start)/size), range=(start,end))

    return N[0]

def get_dtw_distance(reference, target, ref_name='', tar_name='', show_plot = False):
    '''
    function to calculate DTW distance between reference and target
    light curve
        params : reference = reference light curve, list[int]
        params : target = target light curve, list[int]
        params : ref_name = GRB name of referene LC, str
        params : tar_name = GRB name of target LC, str
        params : show_plot = shows DTW plot 
        returns : DTW distance between reference and target GRB
    '''
    # distance metric
    manhattan_distance = lambda x, y : np.abs(x-y)
    
    # calculate distance
    d, cost_matrix, acc_cost_matrix, path = dtw(reference, target, dist=manhattan_distance)

    # visualizing the plot
    if show_plot:
        plt.figure(figsize=(10,10))
        plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
        plt.colorbar(shrink=0.35)
        plt.plot(path[0], path[1], 'w')
        # t90 values
        t90_1 = get_t90(ref_name)
        t90_2 = get_t90(tar_name)
        print(t90_1, t90_2)
        plt.xlabel('Reference Light Curve', fontsize=14)
        plt.ylabel('Target Light Curve', fontsize=14)
        plt.axvline(t90_1, color='r')
        plt.axhline(t90_2, color='b')
        plt.show()
    
    return d