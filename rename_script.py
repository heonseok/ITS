import numpy as np
import os, sys

# mypath = 'synthetic'
mypath = 'DKVMN/checkpoint/'
mypath = mypath + 'assist2009_updated'

from os import listdir
from os.path import isfile, join
print(listdir(mypath))

for name in listdir(mypath):
    print(name)
    parts = name.split('_')
    print(parts)

    if len(parts) > 10:
        new_name = 'knowledge_{}_niLoss_{}_rIdx_{}'.format(parts[1], parts[10], parts[14])
        print(new_name)
        os.rename(os.path.join(mypath,name), os.path.join(mypath,new_name))
