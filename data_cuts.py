import pandas as pd
import json

# read dataframe
df = pd.read_csv('data/gbmdatacleaned.csv', index_col = 0)

# radius in degrees
RADIUS = 10


def nearbyGRB(n):
    '''
    find nearbyGRB in a 'RADIUS' degree radius
        params : n = index (center of circle)
        params : ra_val = ra values , dec_val = dec values
        returns : list of grbs inside 'R' degree radius
    '''
    nbd = []
    ra = df.ra_val.iloc[n]
    dec = df.dec_val.iloc[n]
    for i in range(n+1, df.ra_val.shape[0]):
        if((df.ra_val.iloc[i] - ra)**2 + (df.dec_val.iloc[i] - dec)**2 <= RADIUS**2):
            nbd.append(i)
    return nbd


# Dictionary for GRBs --> center_index : [neighbouring indices]

import time 

s = time.perf_counter()
nbd = dict()
for i in range(df.ra_val.shape[0]):
    nbr_idx = nearbyGRB(i)
    # if ls is not empty
    if nbr_idx:
        nbd[i] = nbr_idx
e = time.perf_counter()

print(f'{e-s}')

# pickling nbd array
with open('data/nbd.json', 'w') as f:
    json.dump(nbd, f)