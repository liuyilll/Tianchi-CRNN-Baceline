import numpy as np
import torch
h = 200
w = 100
trans_range = [w / 10, h / 10]
tr_x = trans_range[0]*np.random.uniform()-trans_range[0]/2
tr_y = trans_range[1]*np.random.uniform()-trans_range[1]/2
transform = np.float32([[1,0, tr_x], [0,1, tr_y]])
#print(transform)
x = np.ones([200, 100])
print(x.size)
