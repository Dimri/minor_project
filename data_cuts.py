import pandas as pd
import pickle

# read dataframe
df = pd.read_csv('data/gbmdatacleaned.csv', index_col = 0)

# radius in degrees
RADIUS = 10


def nearbyGRB(n, ra_val, dec_val):
    '''
    find nearbyGRB in a 'RADIUS' degree radius
        params : n = index (center of circle)
        params : ra_val = ra values , dec_val = dec values
        returns : list of grbs inside 'R' degree radius
    '''
    nbd = []
    ra = ra_val[n]
    dec = dec_val[n]
    for i in range(n+1, len(ra_val)):
        if ra_val[i] == ra or dec_val[i] == dec:
            continue
        if((ra_val[i] - ra)**2 + (dec_val[i] - dec)**2 <= RADIUS**2):
            nbd.append(i)
    return nbd


# array to store (center_index, [neighbouring indices])
nbd = []
for i in range(len(df['ra_val'])):
    ls = nearbyGRB(i, df['ra_val'], df['dec_val'])
    # if ls is not empty
    if ls:
        nbd.append((i,ls))

# pickling nbd array
with open('data/nbd_array', 'wb') as f:
    pickle.dump(nbd, f)