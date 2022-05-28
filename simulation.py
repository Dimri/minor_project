import pandas as pd
import numpy as np
import dtw
from scipy.stats import poisson, norm
import time
import seaborn as sns

def simulateCurves(lc, num):
    simulated_curves = []
    for i in range(num):
        simcurve = []
        for n in lc[0]:
            simcurve.append(poisson.rvs(mu = n, size = 1)[0])
        simulated_curves.append(simcurve)
    return simulated_curves

def 