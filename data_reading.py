import pandas as pd

import numpy as np

# count num of time steps

 data_file = 'C:\PHY487\pbtio3x10.xyz';
 num_lines = sum(1 for line in open(data_file))
 num_at = open(data_file).readline()
 num_steps = num_lines / (num_at + 2)

 for i in range(num_steps):
    data_step = np.loadtxt('C:\PHY487\pbtio3x10.xyz', dtype = str, delimiter = ' ',skiprows=2 , usecols = (0,1,2,3))

 print('C:\PHY487\pbtio3x10.xyz')

 a = data_file[:,0]
 a = [map(str, data_file[:,0])]
 b = [map(float, data_file[:,1])]
 c = [map(float, data_file[:,2])]
 d = [map(float, data_file[:,3])]

 print(type(a), type(b), type(c), type(d))

# For particle a and particle b in list 0. Find the corresponding x,y,z vals
# in lists 1,2,3 and pass all those values to func. "distance"

# CHOOSE PARTICLE:
p1 = "Pb"; p2 = "Ti"
data_file = 'C:\PHY487\pbtio3x10.xyz';
num_lines = open(data_file).readline()
print(num_lines)
print(num_at)


def particles(p1, p2, a, b, c, d):
    p1index = float(a.index(p1));
    p2index = float(a.index(p2));
    for p1index in b:
        x1 = b[p1index];
    for p1index in c:
        y1 = c[p1index];
    for p1index in d:
        z1 = d[p1index];
    for p2index in b:
        x2 = b[p2index];
    for p2index in c:
        y2 = c[p2index];
    for p2index in d:
        z2 = d[p2index];
    return x1, y1, z1, x2, y2, z2


# DISTANCE FORMULA: |(1,2,3)-(4,5,6)|=sqrt((4-1)^2 + (5-2)^2 + (6-3)^2)

def distance(x1, y1, z1, x2, y2, z2):
    d = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2));
    return d;


print(d)