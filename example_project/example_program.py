import torch

#x = torch.randn(100)

#print('Mean of randn: %s' % (x.mean()))

print('Device count: %s' % torch.cuda.device_count())

print('Retrieving data')

import os
print(os.getcwd())

print(__file__)

open('data/stuff.txt')

print('Data retrieved: %s' % open('data/stuff.txt', 'r').read().strip())