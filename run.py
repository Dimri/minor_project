import json
import dataclasses
import pandas as pd

from dtw_script import distance_data


with open('data/nbd.json', 'r') as f:
    nbd = json.load(f)

# cnt = 0
for key, value in nbd.items():
    # cnt += 1

    # if cnt > 10:
    #     break

    result = distance_data(int(key), value)
    bigdct = dict()

    for i, r in enumerate(result):
        bigdct[i] = dataclasses.asdict(r)

    df = pd.DataFrame(bigdct)
    df = df.transpose()
    
    newdf = pd.read_csv('data/new_distance_file.csv')
    newdf = pd.concat([newdf, df])
    newdf.to_csv('data/new_distance_file.csv', index=False)
