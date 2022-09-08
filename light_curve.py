import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from datetime import datetime

from astropy.io import fits
from dtw import dtw
from detector.estimate_source_angles_detectors import angle_to_grb
from scipy.stats import poisson 
from dataclasses import dataclass



def dtw_distance_helper(reference, target):
    # distance metric
    manhattan_distance = lambda x, y : np.abs(x-y)
    
    # calculate distance

    d, cost_matrix , acc_cost_matrix, path = dtw(reference, target,
                                                dist=manhattan_distance)

    return d, cost_matrix, acc_cost_matrix, path

def get_dtw_distance(refGRB, tarGRB, show_plot=False):
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
    
    reference = refGRB.get_photon_counts()
    target = tarGRB.get_photon_counts()

    d, _, acc_cost_matrix, path = dtw_distance_helper(reference, target)

    # visualizing the plot
    if show_plot:
        plot_dtw_distance(refGRB, tarGRB, acc_cost_matrix, path)
    
    return d


def plot_dtw_distance(refGRB, tarGRB, acc_cost_matrix, path):
    _, ax = plt.subplots()
    ax = sns.heatmap(acc_cost_matrix.T, cmap='gray', xticklabels=10, yticklabels=10)
    # invert yaxis to show origin at bottom left
    ax.invert_yaxis() 
    ax.plot(path[0], path[1], 'w')
    ax.set_title(f'{refGRB.name} vs {tarGRB.name}', {'fontsize':15})
    # axis labels
    ax.set_xlabel('Reference Light Curve', fontsize=14)
    ax.set_ylabel('Target Light Curve', fontsize=14)
    ax.axvline(refGRB.t90, color='r', label='Reference t90')
    ax.axhline(tarGRB.t90, color='b', label='Target t90')
    ax.legend()
    plt.show()



class FileNotFoundError(Exception):

    def __init__(self, filepath, message='file not found!!'):
        self.filepath = filepath
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.filepath} {self.message}'

class GRB:

    lcdf = pd.read_csv('data/gbmdatacleaned.csv', index_col=0)
    SEARCH_PATH = 'C:/Users/hhsud/Downloads/GRBS'

    def __init__(self, name):
        # print(name)
        self.name = name
        self.number = name[3:]
        self.ra = self.get_ra()
        self.dec = self.get_dec()
        self.fitfile = self.get_fitfile()
        self.t90 = self.get_t90()
        self.binsize = self.get_binsize()

    def find_file(self, pref):
        filepath = self.SEARCH_PATH + '/' + pref + self.number + '*.fit'
        found = glob.glob(filepath)
        if len(found):
            return found[0]
        else:
            raise FileNotFoundError(filepath)

    def get_ra(self):
        return self.lcdf[self.lcdf['name'] == self.name].ra_val.values[0]
    
    def get_dec(self):
        return self.lcdf[self.lcdf['name'] == self.name].dec_val.values[0]

    def get_brightest_detector(self):
        trigdat_path = self.find_file('glg_trigdat_all_bn')
        detector = angle_to_grb(self.ra, self.dec, trigdat_path)
        return detector[0]
    
    def get_fitfile(self):
        detector = self.get_brightest_detector()
        fitfile_path = self.find_file('glg_tte_' + detector + '_bn')
        return fits.open(fitfile_path)


    def timedf_from_fitfile(self) -> pd.DataFrame:
        '''
        function to get a DataFrame from a raw fitfile

            params : fitfile = raw .fit file
            returns : pandas Dataframe with cols = ['TIME', 'PHA', 'TTIME']
        '''
        # convert the EVENTS header into a dataframe
        timedf = pd.DataFrame(self.fitfile[2].data)
        
        # change the dtype of both the columns
        timedf.PHA = timedf.PHA.astype('int32')
        timedf.TIME = timedf.TIME.astype('float')
        
        # get trigger time from the fits file
        trigtime = self.fitfile[0].header['TRIGTIME']
        
        # make a new translated time column
        timedf['TTIME'] = timedf.TIME - trigtime
        
        return timedf        

    def get_t90(self) -> float:
        '''
        Function to get the t90 value of a given grb from the cleaned gbm dataset
            
            params : grbname = name of the grb 
            returns : t90 of given grb
        '''
        t90 = self.lcdf[self.lcdf['name'] == self.name].t90     # this is an object
        return float(t90.iloc[0].strip())                       # strip to remove the trailing whitespaces


    def get_binsize(self) -> float:
        '''
        Function to return binsize according to T90
            params : t90 of grb
            returns : 0.1 if t90 < 2 else 1
        '''
        return 1 if self.t90 >= 2 else 0.1

    def get_indexin_df(self) -> int:
        return self.lcdf[self.lcdf['name'] == self.name].index.values[0]

        
    def get_trigger_time(self):
        return datetime.strptime(self.lcdf[self.lcdf['name'] == self.name]['trigger_time'].values[0], '%Y-%m-%d %H:%M:%S.%f')


    def get_photon_counts(self, size=0) -> list :
        '''
        function get the photon counts/ light curve from the raw fitfile
        grb = fitfile of the GRB
        size = binsize ; default = 0 -> calculate binsize according to t90
        '''

        # create the dataframe with time and ttime
        df = self.timedf_from_fitfile()

        # get t90
        t90 = self.t90

        # start, end 
        start, end = -10, t90 + 10

        # binsize
        if size == 0:
            size = self.binsize
            
        # get count value in bins 
        N = np.histogram(df.TTIME, bins=int((end - start)/size), range=(start,end))

        return N[0]

    def simulate(self, nums, size=0) -> list:
        '''
        Function to simulate light curves with poisson distribution
            
            params : fitfile = GRB to simulate
            params : size = binsize of GRB
            params : num = number of simulated curves
            returns : simulated curves (list)
        '''
        lc = self.get_photon_counts(size=size)
        simcurves = []  # for simulated curves
        for _ in range(nums):
            simcurves.append([poisson.rvs(mu=n, size=1)[0] for n in lc])
        return simcurves



@dataclass(frozen=True)
class GRBData:
    name : str
    index : int
    refGRBname : str
    ra : float
    dec : float
    t90 : float
    binsize : float
    brightest_detector : str
    distance : float
    sigma : float