import inline as inline
import matplotlib
import numpy as np
import pandas as pd
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import glob
import time
import os
import sys

plt.rcParams["figure.figsize"] = (3, 3)

print(time.ctime())

# file1 = "/home/imaginglab/Data/TPX3/CHIP0/Test/8x8Grid_visEq_1s_W0028_F03-200228-095644-1.csv"
file1 = "/QuantumRouter/8x8-20200316T165613Z-001/8x8Grid_blnkext_viseq_10s_W0057_H07-200306-101040-1-003.csv"
print(file1, time.ctime())

# data1 = np.loadtxt(file1, dtype=float, delimiter=",", usecols=(0, 1, 2, 3, 4))
# data1 = pd.read_csv(file1)

# chunk_size = 5000
# batch_no =1
# for chunk in pd.read_csv(file1,chunksize=chunk_size):
#     chunk.to_csv('8x8Grid_blnkext_viseq_10s_W0057_H07-200306-101040-1-003' + str(batch_no) + '.csv')
#     batch_no +=1
GridX1min = 20
GridX1max = 65
GridY1min = 200
GridY1max = 245
chunk_size = 10000
for chunk in pd.read_csv(file1,chunksize=chunk_size):
    GridX1minvals = ((chunk.Col < GridX1min) & (chunk.Col > GridX1max))
    GridX1maxvals = (chunk.Col > GridX1max)
    GridY1minvals = (chunk.Row < GridY1min)
    GridY1maxvals = (chunk.Row > GridY1max)

# print data
# y1 = data1[:, 0]
# x1 = data1[:, 1]
# t1 = data1[:, 2]
# a1 = data1[:, 3]
# A1 = data1[:, 4]
y1 = data1.iloc[:, 0].values
x1 = data1.iloc[:, 1].values
t1 = data1.iloc[:, 2].values
a1 = data1.iloc[:, 3].values
A1 = data1.iloc[:, 4].values

# for i in range(0,20):
#     print (i, x1[i],y1[i],t1[i],a1[i],A1[i])

print(len(t1), time.ctime())

# file2 = "/home/imaginglab/Data/TPX3/CHIP0/Test/8x8Grid_SettingsEq_1s_W0028_F03-200228-100050-1.csv"
file2 = "/QuantumRouter/8x8-20200316T165613Z-001/8x8-20200316T165613Z-0018x8Grid_SettingsEq_10s_W0028_F03-200228-100025-1.csv"
print(file2, time.ctime())

data2 = np.loadtxt(file2, dtype=float, delimiter=",", usecols=(0, 1, 2, 3, 4))

# print data
y2 = data2[:, 0]
x2 = data2[:, 1]
t2 = data2[:, 2]
a2 = data2[:, 3]
A2 = data2[:, 4]

# for i in range(0,20):
#     print (i, x2[i],y2[i],t2[i],a2[i],A2[i])

print(len(t2), time.ctime())

fig, ax1 = plt.subplots(ncols=1, figsize=(20, 8))

nbins = 40

plt.hist(a1, bins=nbins, range=[0, 20000], lw=3, histtype='step', label='Visually Equalized', color='blue')
plt.hist(a2, bins=nbins, range=[0, 20000], lw=3, histtype='step', label='Settings Equalized', color='red')

plt.xlabel('ToT, ns')
plt.legend()
plt.show()

fig1, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9.5, 4))

h = ax0.hist2d(x1, y1, bins=256, range=[(0, 256), (0, 256)])
fig1.colorbar(h[3], ax=ax0)

h = ax1.hist2d(x1, y1, bins=256, range=[(0, 256), (0, 256)], norm=mpl.colors.LogNorm())
fig1.colorbar(h[3], ax=ax1)
fig1.tight_layout()
plt.show()

GridX1min = 60
GridX1max = 105
GridY1min = 155
GridY1max = 198

fig2, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9.5, 4))

h = ax0.hist2d(x1, y1, bins=GridX1max - GridX1min, range=[(GridX1min, GridX1max), (GridY1min, GridY1max)])
fig2.colorbar(h[3], ax=ax0)

h = ax1.hist2d(x1, y1, bins=GridX1max - GridX1min, range=[(GridX1min, GridX1max), (GridY1min, GridY1max)],
               norm=mpl.colors.LogNorm())
fig2.colorbar(h[3], ax=ax1)

fig2.tight_layout()
plt.show()

fig3, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9.5, 4))

h = ax0.hist2d(x2, y2, bins=256, range=[(0, 256), (0, 256)])
fig3.colorbar(h[3], ax=ax0)

h = ax1.hist2d(x2, y2, bins=256, range=[(0, 256), (0, 256)], norm=mpl.colors.LogNorm())
fig3.colorbar(h[3], ax=ax1)
fig3.tight_layout()
plt.show()

GridX2min = 60
GridX2max = 105
GridY2min = 155
GridY2max = 198

fig4, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9.5, 4))

h = ax0.hist2d(x2, y2, bins=GridX2max - GridX2min, range=[(GridX2min, GridX2max), (GridY2min, GridY2max)])
fig4.colorbar(h[3], ax=ax0)

h = ax1.hist2d(x2, y2, bins=GridX2max - GridX2min, range=[(GridX2min, GridX2max), (GridY2min, GridY2max)],
               norm=mpl.colors.LogNorm())
fig4.colorbar(h[3], ax=ax1)

fig4.tight_layout()
plt.show()

fig5, ax0 = plt.subplots(ncols=1, figsize=(10, 4))
plt.hist(t1 / 4096. * 25., bins=1000, color='r', ec='r', range=(.998e9, 1e9))
plt.title("TOA", fontsize=12)  # change the title
plt.xlabel('TOA, ns', fontsize=12)
plt.ylabel('counts', fontsize=12)
plt.show()

# fig, ax0 = plt.subplots(ncols=1, figsize=(10, 4))
# plt.hist(tt/4096.*25., bins = 100, color = 'r', ec = 'k')
# plt.title("TOA", fontsize = 12) # change the title
# plt.xlabel('TOA, ns',fontsize = 12)
# plt.ylabel('counts',fontsize = 12)
# plt.show()

fig6, ax0 = plt.subplots(ncols=1, figsize=(10, 4))
# plt.hist(t/4096.*25., bins = 10, range = (0.5E+9, 1.5E+9), color = 'r', ec = 'k')
plt.hist(t1 / 4096. * 25., bins=1000, range=(-1.E+9, 1.E+9), color='r', ec='r')
plt.title("TOA", fontsize=12)  # change the title
plt.xlabel('TOA, ns', fontsize=12)
plt.ylabel('counts', fontsize=12)
plt.show()

fig7, ax0 = plt.subplots(ncols=1, figsize=(10, 4))
plt.hist(t2 / 4096. * 25., bins=1000, color='r', ec='r', range=(.998e9, 1e9))
plt.title("TOA", fontsize=12)  # change the title
plt.xlabel('TOA, ns', fontsize=12)
plt.ylabel('counts', fontsize=12)
plt.show()

# fig, ax0 = plt.subplots(ncols=1, figsize=(10, 4))
# plt.hist(tt/4096.*25., bins = 100, color = 'r', ec = 'k')
# plt.title("TOA", fontsize = 12) # change the title
# plt.xlabel('TOA, ns',fontsize = 12)
# plt.ylabel('counts',fontsize = 12)
# plt.show()

fig8, ax0 = plt.subplots(ncols=1, figsize=(10, 4))
# plt.hist(t/4096.*25., bins = 10, range = (0.5E+9, 1.5E+9), color = 'r', ec = 'k')
plt.hist(t2 / 4096. * 25., bins=1000, range=(-1.E+9, 1.E+9), color='r', ec='r')
plt.title("TOA", fontsize=12)  # change the title
plt.xlabel('TOA, ns', fontsize=12)
plt.ylabel('counts', fontsize=12)
plt.show()

fig9, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 3))

ax0.hist(x1, bins=GridX1max - GridX1min, range=(GridX1min, GridX1max), color='g', alpha=0.5, histtype='stepfilled')
plt.title("x", fontsize=12)  # change the title
plt.xlabel('x, pixel', fontsize=12)
plt.ylabel('counts', fontsize=12)

ax1.hist(y1, bins=GridY1max - GridY1min, range=(155, 198), color='g', alpha=0.5, histtype='stepfilled')
plt.title("y", fontsize=12)  # change the title
plt.xlabel('y, pixel', fontsize=12)
plt.ylabel('counts', fontsize=12)

fig10, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 3))

ax0.hist(x2, bins=GridX2max - GridX2min, range=(GridX2min, GridX2max), color='g', alpha=0.5, histtype='stepfilled')
plt.title("x", fontsize=12)  # change the title
plt.xlabel('x, pixel', fontsize=12)
plt.ylabel('counts', fontsize=12)

ax1.hist(y2, bins=GridY2max - GridY2min, range=(155, 198), color='g', alpha=0.5, histtype='stepfilled')
plt.title("y", fontsize=12)  # change the title
plt.xlabel('y, pixel', fontsize=12)
plt.ylabel('counts', fontsize=12)

# Sort arrays to exclude points outside Grid


# Initialize trash arrays
Grid_x1trash = []
Grid_y1trash = []
Grid_t1trash = []
Grid_a1trash = []
Grid_A1trash = []

# Sort out points NOT in 8x8 Grid, add to trash
for i in range(len(x1) - 1):
    if GridY1min > y1[i]:
        Grid_x1trash.append(i)
        Grid_y1trash.append(i)
        Grid_t1trash.append(i)
        Grid_a1trash.append(i)
        Grid_A1trash.append(i)
    elif GridY1max < y1[i]:
        Grid_x1trash.append(i)
        Grid_y1trash.append(i)
        Grid_t1trash.append(i)
        Grid_a1trash.append(i)
        Grid_A1trash.append(i)
    elif GridX1min > x1[i]:
        Grid_x1trash.append(i)
        Grid_y1trash.append(i)
        Grid_t1trash.append(i)
        Grid_a1trash.append(i)
        Grid_A1trash.append(i)
    elif GridX1max < x1[i]:
        Grid_x1trash.append(i)
        Grid_y1trash.append(i)
        Grid_t1trash.append(i)
        Grid_a1trash.append(i)
        Grid_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
Grid_x1 = np.delete(x1, Grid_x1trash)
Grid_y1 = np.delete(y1, Grid_y1trash)
Grid_t1 = np.delete(t1, Grid_t1trash)
Grid_a1 = np.delete(a1, Grid_a1trash)
Grid_A1 = np.delete(A1, Grid_a1trash)

# Sanity Check
print(len(x1), len(Grid_x1trash), len(Grid_x1))

# Define column segregations

C1min = GridX1min
C1max = C1min + 5
C2min = C1max + 1
C2max = C2min + 4
C3min = C2max + 1
C3max = C3min + 4
C4min = C3max + 1
C4max = C4min + 5
C5min = C4max + 1
C5max = C5min + 5
C6min = C5max + 1
C6max = C6min + 5
C7min = C6max + 1
C7max = C7min + 5
C8min = C7max + 1
C8max = C8min + 5

# Defime Row segregations

R1min = GridY1min
R1max = R1min + 5
R2min = R1max + 1
R2max = R2min + 4
R3min = R2max + 1
R3max = R3min + 4
R4min = R3max + 1
R4max = R4min + 5
R5min = R4max + 1
R5max = R5min + 5
R6min = R5max + 1
R6max = R6min + 5
R7min = R6max + 1
R7max = R7min + 4
R8min = R7max + 1
R8max = R8min + 5

fig11, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 4))

h = ax1.hist2d(Grid_x1, Grid_y1, bins=C1max - C1min, range=[(C1min, C1max), (R1min, R1max)], norm=mpl.colors.LogNorm())
fig11.colorbar(h[3], ax=ax1)

h = ax2.hist2d(Grid_x1, Grid_y1, bins=C2max - C2min, range=[(C2min, C2max), (R1min, R1max)], norm=mpl.colors.LogNorm())
fig11.colorbar(h[3], ax=ax2)

h = ax3.hist2d(Grid_x1, Grid_y1, bins=C3max - C3min, range=[(C3min, C3max), (R1min, R1max)], norm=mpl.colors.LogNorm())
fig11.colorbar(h[3], ax=ax3)

h = ax4.hist2d(Grid_x1, Grid_y1, bins=C4max - C4min, range=[(C4min, C4max), (R1min, R1max)], norm=mpl.colors.LogNorm())
fig11.colorbar(h[3], ax=ax4)

fig11.tight_layout()
plt.show()

fig12, (ax5, ax6, ax7, ax8) = plt.subplots(ncols=4, figsize=(20, 4))

h = ax5.hist2d(Grid_x1, Grid_y1, bins=C5max - C5min, range=[(C5min, C5max), (R1min, R1max)], norm=mpl.colors.LogNorm())
fig12.colorbar(h[3], ax=ax5)

h = ax6.hist2d(Grid_x1, Grid_y1, bins=C6max - C6min, range=[(C6min, C6max), (R1min, R1max)], norm=mpl.colors.LogNorm())
fig12.colorbar(h[3], ax=ax6)

h = ax7.hist2d(Grid_x1, Grid_y1, bins=C7max - C7min, range=[(C7min, C7max), (R1min, R1max)], norm=mpl.colors.LogNorm())
fig12.colorbar(h[3], ax=ax7)

h = ax8.hist2d(Grid_x1, Grid_y1, bins=C8max - C8min, range=[(C8min, C8max), (R1min, R1max)], norm=mpl.colors.LogNorm())
fig12.colorbar(h[3], ax=ax8)

fig12.tight_layout()
plt.show()

# Sort arrays into row 1 array


# Initialize trash arrays
row1_x1trash = []
row1_y1trash = []
row1_t1trash = []
row1_a1trash = []
row1_A1trash = []

# Sort out points NOT in row, add to trash
for i in range(len(Grid_x1) - 1):
    if R1min > y1[i]:
        row1_x1trash.append(i)
        row1_y1trash.append(i)
        row1_t1trash.append(i)
        row1_a1trash.append(i)
        row1_A1trash.append(i)
    elif R1max < y1[i]:
        row1_x1trash.append(i)
        row1_y1trash.append(i)
        row1_t1trash.append(i)
        row1_a1trash.append(i)
        row1_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
row1_x1 = np.delete(Grid_x1, row1_x1trash)
row1_y1 = np.delete(Grid_y1, row1_y1trash)
row1_t1 = np.delete(Grid_t1, row1_t1trash)
row1_a1 = np.delete(Grid_a1, row1_a1trash)
row1_A1 = np.delete(Grid_A1, row1_A1trash)

# Sanity Check
print(len(Grid_x1), len(row1_x1trash), len(row1_x1))

# Sort Row 1 array into points


# Initialize trash arrays
C1R1_x1trash = []
C1R1_y1trash = []
C1R1_t1trash = []
C1R1_a1trash = []
C1R1_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row1_x1) - 1):
    if C1min > row1_x1[i]:
        C1R1_x1trash.append(i)
        C1R1_y1trash.append(i)
        C1R1_t1trash.append(i)
        C1R1_a1trash.append(i)
        C1R1_A1trash.append(i)
    elif C1max < row1_x1[i]:
        C1R1_x1trash.append(i)
        C1R1_y1trash.append(i)
        C1R1_t1trash.append(i)
        C1R1_a1trash.append(i)
        C1R1_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C1R1_x1 = np.delete(row1_x1, C1R1_x1trash)
C1R1_y1 = np.delete(row1_y1, C1R1_y1trash)
C1R1_t1 = np.delete(row1_t1, C1R1_t1trash)
C1R1_a1 = np.delete(row1_a1, C1R1_a1trash)
C1R1_A1 = np.delete(row1_A1, C1R1_A1trash)

# Sanity Check
print(len(row1_x1), len(C1R1_x1trash))
print('Point 1 Rate (Hz):', len(C1R1_x1))

# save rate in case it is needed later
rate_C1R1 = len(C1R1_x1)

# Initialize trash arrays
C2R1_x1trash = []
C2R1_y1trash = []
C2R1_t1trash = []
C2R1_a1trash = []
C2R1_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row1_x1) - 1):
    if C2min > row1_x1[i]:
        C2R1_x1trash.append(i)
        C2R1_y1trash.append(i)
        C2R1_t1trash.append(i)
        C2R1_a1trash.append(i)
        C2R1_A1trash.append(i)
    elif C2max < row1_x1[i]:
        C2R1_x1trash.append(i)
        C2R1_y1trash.append(i)
        C2R1_t1trash.append(i)
        C2R1_a1trash.append(i)
        C2R1_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C2R1_x1 = np.delete(row1_x1, C2R1_x1trash)
C2R1_y1 = np.delete(row1_y1, C2R1_y1trash)
C2R1_t1 = np.delete(row1_t1, C2R1_t1trash)
C2R1_a1 = np.delete(row1_a1, C2R1_a1trash)
C2R1_A1 = np.delete(row1_A1, C2R1_A1trash)

# Sanity Check
print(len(row1_x1), len(C2R1_x1trash))
print('Point 2 Rate (Hz):', len(C2R1_x1))

# save rate in case it is needed later
rate_C2R1 = len(C2R1_x1)

# Initialize trash arrays
C3R1_x1trash = []
C3R1_y1trash = []
C3R1_t1trash = []
C3R1_a1trash = []
C3R1_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row1_x1) - 1):
    if C3min > row1_x1[i]:
        C3R1_x1trash.append(i)
        C3R1_y1trash.append(i)
        C3R1_t1trash.append(i)
        C3R1_a1trash.append(i)
        C3R1_A1trash.append(i)
    elif C3max < row1_x1[i]:
        C3R1_x1trash.append(i)
        C3R1_y1trash.append(i)
        C3R1_t1trash.append(i)
        C3R1_a1trash.append(i)
        C3R1_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C3R1_x1 = np.delete(row1_x1, C3R1_x1trash)
C3R1_y1 = np.delete(row1_y1, C3R1_y1trash)
C3R1_t1 = np.delete(row1_t1, C3R1_t1trash)
C3R1_a1 = np.delete(row1_a1, C3R1_a1trash)
C3R1_A1 = np.delete(row1_A1, C3R1_A1trash)

# Sanity Check
print(len(row1_x1), len(C3R1_x1trash))
print('Point 3 Rate (Hz):', len(C3R1_x1))

# save rate in case it is needed later
rate_C3R1 = len(C3R1_x1)

# Initialize trash arrays
C4R1_x1trash = []
C4R1_y1trash = []
C4R1_t1trash = []
C4R1_a1trash = []
C4R1_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row1_x1) - 1):
    if C4min > row1_x1[i]:
        C4R1_x1trash.append(i)
        C4R1_y1trash.append(i)
        C4R1_t1trash.append(i)
        C4R1_a1trash.append(i)
        C4R1_A1trash.append(i)
    elif C4max < row1_x1[i]:
        C4R1_x1trash.append(i)
        C4R1_y1trash.append(i)
        C4R1_t1trash.append(i)
        C4R1_a1trash.append(i)
        C4R1_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C4R1_x1 = np.delete(row1_x1, C4R1_x1trash)
C4R1_y1 = np.delete(row1_y1, C4R1_y1trash)
C4R1_t1 = np.delete(row1_t1, C4R1_t1trash)
C4R1_a1 = np.delete(row1_a1, C4R1_a1trash)
C4R1_A1 = np.delete(row1_A1, C4R1_A1trash)

# Sanity Check
print(len(row1_x1), len(C4R1_x1trash))
print('Point 4 Rate (Hz):', len(C4R1_x1))

# save rate in case it is needed later
rate_C4R1 = len(C4R1_x1)

# Initialize trash arrays
C5R1_x1trash = []
C5R1_y1trash = []
C5R1_t1trash = []
C5R1_a1trash = []
C5R1_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row1_x1) - 1):
    if C5min > row1_x1[i]:
        C5R1_x1trash.append(i)
        C5R1_y1trash.append(i)
        C5R1_t1trash.append(i)
        C5R1_a1trash.append(i)
        C5R1_A1trash.append(i)
    elif C5max < row1_x1[i]:
        C5R1_x1trash.append(i)
        C5R1_y1trash.append(i)
        C5R1_t1trash.append(i)
        C5R1_a1trash.append(i)
        C5R1_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C5R1_x1 = np.delete(row1_x1, C5R1_x1trash)
C5R1_y1 = np.delete(row1_y1, C5R1_y1trash)
C5R1_t1 = np.delete(row1_t1, C5R1_t1trash)
C5R1_a1 = np.delete(row1_a1, C5R1_a1trash)
C5R1_A1 = np.delete(row1_A1, C5R1_A1trash)

# Sanity Check
print(len(row1_x1), len(C5R1_x1trash))
print('Point 5 Rate (Hz):', len(C5R1_x1))

# save rate in case it is needed later
rate_C5R1 = len(C5R1_x1)

# Initialize trash arrays
C6R1_x1trash = []
C6R1_y1trash = []
C6R1_t1trash = []
C6R1_a1trash = []
C6R1_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row1_x1) - 1):
    if C6min > row1_x1[i]:
        C6R1_x1trash.append(i)
        C6R1_y1trash.append(i)
        C6R1_t1trash.append(i)
        C6R1_a1trash.append(i)
        C6R1_A1trash.append(i)
    elif C6max < row1_x1[i]:
        C6R1_x1trash.append(i)
        C6R1_y1trash.append(i)
        C6R1_t1trash.append(i)
        C6R1_a1trash.append(i)
        C6R1_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C6R1_x1 = np.delete(row1_x1, C6R1_x1trash)
C6R1_y1 = np.delete(row1_y1, C6R1_y1trash)
C6R1_t1 = np.delete(row1_t1, C6R1_t1trash)
C6R1_a1 = np.delete(row1_a1, C6R1_a1trash)
C6R1_A1 = np.delete(row1_A1, C6R1_A1trash)

# Sanity Check
print(len(row1_x1), len(C6R1_x1trash))
print('Point 6 Rate (Hz):', len(C6R1_x1))

# save rate in case it is needed later
rate_C6R1 = len(C6R1_x1)

# Initialize trash arrays
C7R1_x1trash = []
C7R1_y1trash = []
C7R1_t1trash = []
C7R1_a1trash = []
C7R1_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row1_x1) - 1):
    if C7min > row1_x1[i]:
        C7R1_x1trash.append(i)
        C7R1_y1trash.append(i)
        C7R1_t1trash.append(i)
        C7R1_a1trash.append(i)
        C7R1_A1trash.append(i)
    elif C7max < row1_x1[i]:
        C7R1_x1trash.append(i)
        C7R1_y1trash.append(i)
        C7R1_t1trash.append(i)
        C7R1_a1trash.append(i)
        C7R1_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C7R1_x1 = np.delete(row1_x1, C7R1_x1trash)
C7R1_y1 = np.delete(row1_y1, C7R1_y1trash)
C7R1_t1 = np.delete(row1_t1, C7R1_t1trash)
C7R1_a1 = np.delete(row1_a1, C7R1_a1trash)
C7R1_A1 = np.delete(row1_A1, C7R1_A1trash)

# Sanity Check
print(len(row1_x1), len(C7R1_x1trash))
print('Point 7 Rate (Hz):', len(C7R1_x1))

# save rate in case it is needed later
rate_C7R1 = len(C7R1_x1)

# Initialize trash arrays
C8R1_x1trash = []
C8R1_y1trash = []
C8R1_t1trash = []
C8R1_a1trash = []
C8R1_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row1_x1) - 1):
    if C8min > row1_x1[i]:
        C8R1_x1trash.append(i)
        C8R1_y1trash.append(i)
        C8R1_t1trash.append(i)
        C8R1_a1trash.append(i)
        C8R1_A1trash.append(i)
    elif C8max < row1_x1[i]:
        C8R1_x1trash.append(i)
        C8R1_y1trash.append(i)
        C8R1_t1trash.append(i)
        C8R1_a1trash.append(i)
        C8R1_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C8R1_x1 = np.delete(row1_x1, C8R1_x1trash)
C8R1_y1 = np.delete(row1_y1, C8R1_y1trash)
C8R1_t1 = np.delete(row1_t1, C8R1_t1trash)
C8R1_a1 = np.delete(row1_a1, C8R1_a1trash)
C8R1_A1 = np.delete(row1_A1, C8R1_A1trash)

# Sanity Check
print(len(row1_x1), len(C8R1_x1trash))
print('Point 8 Rate (Hz):', len(C8R1_x1))

# save rate in case it is needed later
rate_C8R1 = len(C8R1_x1)

# Display Row 2

fig13, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 4))

h = ax1.hist2d(Grid_x1, Grid_y1, bins=C1max - C1min, range=[(C1min, C1max), (R2min, R2max)], norm=mpl.colors.LogNorm())
fig13.colorbar(h[3], ax=ax1)

h = ax2.hist2d(Grid_x1, Grid_y1, bins=C2max - C2min, range=[(C2min, C2max), (R2min, R2max)], norm=mpl.colors.LogNorm())
fig13.colorbar(h[3], ax=ax2)

h = ax3.hist2d(Grid_x1, Grid_y1, bins=C3max - C3min, range=[(C3min, C3max), (R2min, R2max)], norm=mpl.colors.LogNorm())
fig13.colorbar(h[3], ax=ax3)

h = ax4.hist2d(Grid_x1, Grid_y1, bins=C4max - C4min, range=[(C4min, C4max), (R2min, R2max)], norm=mpl.colors.LogNorm())
fig13.colorbar(h[3], ax=ax4)

fig13.tight_layout()
plt.show()

fig14, (ax5, ax6, ax7, ax8) = plt.subplots(ncols=4, figsize=(20, 4))

h = ax5.hist2d(Grid_x1, Grid_y1, bins=C5max - C5min, range=[(C5min, C5max), (R2min, R2max)], norm=mpl.colors.LogNorm())
fig14.colorbar(h[3], ax=ax5)

h = ax6.hist2d(Grid_x1, Grid_y1, bins=C6max - C6min, range=[(C6min, C6max), (R2min, R2max)], norm=mpl.colors.LogNorm())
fig14.colorbar(h[3], ax=ax6)

h = ax7.hist2d(Grid_x1, Grid_y1, bins=C7max - C7min, range=[(C7min, C7max), (R2min, R2max)], norm=mpl.colors.LogNorm())
fig14.colorbar(h[3], ax=ax7)

h = ax8.hist2d(Grid_x1, Grid_y1, bins=C8max - C8min, range=[(C8min, C8max), (R2min, R2max)], norm=mpl.colors.LogNorm())
fig14.colorbar(h[3], ax=ax8)

fig14.tight_layout()
plt.show()

# Sort arrays into row 2 array


# Initialize trash arrays
row2_x1trash = []
row2_y1trash = []
row2_t1trash = []
row2_a1trash = []
row2_A1trash = []

# Sort out points NOT in row, add to trash
for i in range(len(Grid_x1) - 1):
    if R2min > y1[i]:
        row2_x1trash.append(i)
        row2_y1trash.append(i)
        row2_t1trash.append(i)
        row2_a1trash.append(i)
        row2_A1trash.append(i)
    elif R2max < y1[i]:
        row2_x1trash.append(i)
        row2_y1trash.append(i)
        row2_t1trash.append(i)
        row2_a1trash.append(i)
        row2_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
row2_x1 = np.delete(Grid_x1, row2_x1trash)
row2_y1 = np.delete(Grid_y1, row2_y1trash)
row2_t1 = np.delete(Grid_t1, row2_t1trash)
row2_a1 = np.delete(Grid_a1, row2_a1trash)
row2_A1 = np.delete(Grid_A1, row2_A1trash)

# Sanity Check
print(len(Grid_x1), len(row2_x1trash), len(row2_x1))

# Sort Row 2 array into points


# Initialize trash arrays
C1R2_x1trash = []
C1R2_y1trash = []
C1R2_t1trash = []
C1R2_a1trash = []
C1R2_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row2_x1) - 1):
    if C1min > row2_x1[i]:
        C1R2_x1trash.append(i)
        C1R2_y1trash.append(i)
        C1R2_t1trash.append(i)
        C1R2_a1trash.append(i)
        C1R2_A1trash.append(i)
    elif C1max < row2_x1[i]:
        C1R2_x1trash.append(i)
        C1R2_y1trash.append(i)
        C1R2_t1trash.append(i)
        C1R2_a1trash.append(i)
        C1R2_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C1R2_x1 = np.delete(row2_x1, C1R2_x1trash)
C1R2_y1 = np.delete(row2_y1, C1R2_y1trash)
C1R2_t1 = np.delete(row2_t1, C1R2_t1trash)
C1R2_a1 = np.delete(row2_a1, C1R2_a1trash)
C1R2_A1 = np.delete(row2_A1, C1R2_A1trash)

# Sanity Check
print(len(row2_x1), len(C1R2_x1trash))
print('Point 1 Rate (Hz):', len(C1R2_x1))

# save rate in case it is needed later
rate_C1R2 = len(C1R2_x1)

# Initialize trash arrays
C2R2_x1trash = []
C2R2_y1trash = []
C2R2_t1trash = []
C2R2_a1trash = []
C2R2_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row2_x1) - 1):
    if C2min > row2_x1[i]:
        C2R2_x1trash.append(i)
        C2R2_y1trash.append(i)
        C2R2_t1trash.append(i)
        C2R2_a1trash.append(i)
        C2R2_A1trash.append(i)
    elif C2max < row2_x1[i]:
        C2R2_x1trash.append(i)
        C2R2_y1trash.append(i)
        C2R2_t1trash.append(i)
        C2R2_a1trash.append(i)
        C2R2_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C2R2_x1 = np.delete(row2_x1, C2R2_x1trash)
C2R2_y1 = np.delete(row2_y1, C2R2_y1trash)
C2R2_t1 = np.delete(row2_t1, C2R2_t1trash)
C2R2_a1 = np.delete(row2_a1, C2R2_a1trash)
C2R2_A1 = np.delete(row2_A1, C2R2_A1trash)

# Sanity Check
print(len(row2_x1), len(C2R2_x1trash))
print('Point 2 Rate (Hz):', len(C2R2_x1))

# save rate in case it is needed later
rate_C2R2 = len(C2R2_x1)

# Initialize trash arrays
C3R2_x1trash = []
C3R2_y1trash = []
C3R2_t1trash = []
C3R2_a1trash = []
C3R2_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row2_x1) - 1):
    if C3min > row2_x1[i]:
        C3R2_x1trash.append(i)
        C3R2_y1trash.append(i)
        C3R2_t1trash.append(i)
        C3R2_a1trash.append(i)
        C3R2_A1trash.append(i)
    elif C3max < row2_x1[i]:
        C3R2_x1trash.append(i)
        C3R2_y1trash.append(i)
        C3R2_t1trash.append(i)
        C3R2_a1trash.append(i)
        C3R2_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C3R2_x1 = np.delete(row2_x1, C3R2_x1trash)
C3R2_y1 = np.delete(row2_y1, C3R2_y1trash)
C3R2_t1 = np.delete(row2_t1, C3R2_t1trash)
C3R2_a1 = np.delete(row2_a1, C3R2_a1trash)
C3R2_A1 = np.delete(row2_A1, C3R2_A1trash)

# Sanity Check
print(len(row2_x1), len(C3R2_x1trash))
print('Point 3 Rate (Hz):', len(C3R2_x1))

# save rate in case it is needed later
rate_C3R2 = len(C3R2_x1)

# Initialize trash arrays
C4R2_x1trash = []
C4R2_y1trash = []
C4R2_t1trash = []
C4R2_a1trash = []
C4R2_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row2_x1) - 1):
    if C4min > row2_x1[i]:
        C4R2_x1trash.append(i)
        C4R2_y1trash.append(i)
        C4R2_t1trash.append(i)
        C4R2_a1trash.append(i)
        C4R2_A1trash.append(i)
    elif C4max < row2_x1[i]:
        C4R2_x1trash.append(i)
        C4R2_y1trash.append(i)
        C4R2_t1trash.append(i)
        C4R2_a1trash.append(i)
        C4R2_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C4R2_x1 = np.delete(row2_x1, C4R2_x1trash)
C4R2_y1 = np.delete(row2_y1, C4R2_y1trash)
C4R2_t1 = np.delete(row2_t1, C4R2_t1trash)
C4R2_a1 = np.delete(row2_a1, C4R2_a1trash)
C4R2_A1 = np.delete(row2_A1, C4R2_A1trash)

# Sanity Check
print(len(row2_x1), len(C4R2_x1trash))
print('Point 4 Rate (Hz):', len(C4R2_x1))

# save rate in case it is needed later
rate_C4R2 = len(C4R2_x1)

# Initialize trash arrays
C5R2_x1trash = []
C5R2_y1trash = []
C5R2_t1trash = []
C5R2_a1trash = []
C5R2_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row2_x1) - 1):
    if C5min > row2_x1[i]:
        C5R2_x1trash.append(i)
        C5R2_y1trash.append(i)
        C5R2_t1trash.append(i)
        C5R2_a1trash.append(i)
        C5R2_A1trash.append(i)
    elif C5max < row2_x1[i]:
        C5R2_x1trash.append(i)
        C5R2_y1trash.append(i)
        C5R2_t1trash.append(i)
        C5R2_a1trash.append(i)
        C5R2_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C5R2_x1 = np.delete(row2_x1, C5R2_x1trash)
C5R2_y1 = np.delete(row2_y1, C5R2_y1trash)
C5R2_t1 = np.delete(row2_t1, C5R2_t1trash)
C5R2_a1 = np.delete(row2_a1, C5R2_a1trash)
C5R2_A1 = np.delete(row2_A1, C5R2_A1trash)

# Sanity Check
print(len(row2_x1), len(C5R2_x1trash))
print('Point 5 Rate (Hz):', len(C5R2_x1))

# save rate in case it is needed later
rate_C5R2 = len(C5R2_x1)

# Initialize trash arrays
C6R2_x1trash = []
C6R2_y1trash = []
C6R2_t1trash = []
C6R2_a1trash = []
C6R2_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row2_x1) - 1):
    if C6min > row2_x1[i]:
        C6R2_x1trash.append(i)
        C6R2_y1trash.append(i)
        C6R2_t1trash.append(i)
        C6R2_a1trash.append(i)
        C6R2_A1trash.append(i)
    elif C6max < row2_x1[i]:
        C6R2_x1trash.append(i)
        C6R2_y1trash.append(i)
        C6R2_t1trash.append(i)
        C6R2_a1trash.append(i)
        C6R2_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C6R2_x1 = np.delete(row2_x1, C6R2_x1trash)
C6R2_y1 = np.delete(row2_y1, C6R2_y1trash)
C6R2_t1 = np.delete(row2_t1, C6R2_t1trash)
C6R2_a1 = np.delete(row2_a1, C6R2_a1trash)
C6R2_A1 = np.delete(row2_A1, C6R2_A1trash)

# Sanity Check
print(len(row2_x1), len(C6R2_x1trash))
print('Point 6 Rate (Hz):', len(C6R2_x1))

# save rate in case it is needed later
rate_C6R2 = len(C6R2_x1)

# Initialize trash arrays
C7R2_x1trash = []
C7R2_y1trash = []
C7R2_t1trash = []
C7R2_a1trash = []
C7R2_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row2_x1) - 1):
    if C7min > row2_x1[i]:
        C7R2_x1trash.append(i)
        C7R2_y1trash.append(i)
        C7R2_t1trash.append(i)
        C7R2_a1trash.append(i)
        C7R2_A1trash.append(i)
    elif C7max < row2_x1[i]:
        C7R2_x1trash.append(i)
        C7R2_y1trash.append(i)
        C7R2_t1trash.append(i)
        C7R2_a1trash.append(i)
        C7R2_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C7R2_x1 = np.delete(row2_x1, C7R2_x1trash)
C7R2_y1 = np.delete(row2_y1, C7R2_y1trash)
C7R2_t1 = np.delete(row2_t1, C7R2_t1trash)
C7R2_a1 = np.delete(row2_a1, C7R2_a1trash)
C7R2_A1 = np.delete(row2_A1, C7R2_A1trash)

# Sanity Check
print(len(row2_x1), len(C7R2_x1trash))
print('Point 7 Rate (Hz):', len(C7R2_x1))

# save rate in case it is needed later
rate_C7R2 = len(C7R2_x1)

# Initialize trash arrays
C8R2_x1trash = []
C8R2_y1trash = []
C8R2_t1trash = []
C8R2_a1trash = []
C8R2_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row2_x1) - 1):
    if C8min > row2_x1[i]:
        C8R2_x1trash.append(i)
        C8R2_y1trash.append(i)
        C8R2_t1trash.append(i)
        C8R2_a1trash.append(i)
        C8R2_A1trash.append(i)
    elif C8max < row2_x1[i]:
        C8R2_x1trash.append(i)
        C8R2_y1trash.append(i)
        C8R2_t1trash.append(i)
        C8R2_a1trash.append(i)
        C8R2_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C8R2_x1 = np.delete(row2_x1, C8R2_x1trash)
C8R2_y1 = np.delete(row2_y1, C8R2_y1trash)
C8R2_t1 = np.delete(row2_t1, C8R2_t1trash)
C8R2_a1 = np.delete(row2_a1, C8R2_a1trash)
C8R2_A1 = np.delete(row2_A1, C8R2_A1trash)

# Sanity Check
print(len(row2_x1), len(C8R2_x1trash))
print('Point 8 Rate (Hz):', len(C8R2_x1))

# save rate in case it is needed later
rate_C8R2 = len(C8R2_x1)

# Display Row 3

fig15, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 4))

h = ax1.hist2d(Grid_x1, Grid_y1, bins=C1max - C1min, range=[(C1min, C1max), (R3min, R3max)], norm=mpl.colors.LogNorm())
fig15.colorbar(h[3], ax=ax1)

h = ax2.hist2d(Grid_x1, Grid_y1, bins=C2max - C2min, range=[(C2min, C2max), (R3min, R3max)], norm=mpl.colors.LogNorm())
fig15.colorbar(h[3], ax=ax2)

h = ax3.hist2d(Grid_x1, Grid_y1, bins=C3max - C3min, range=[(C3min, C3max), (R3min, R3max)], norm=mpl.colors.LogNorm())
fig15.colorbar(h[3], ax=ax3)

h = ax4.hist2d(Grid_x1, Grid_y1, bins=C4max - C4min, range=[(C4min, C4max), (R3min, R3max)], norm=mpl.colors.LogNorm())
fig15.colorbar(h[3], ax=ax4)

fig15.tight_layout()
plt.show()

fig16, (ax5, ax6, ax7, ax8) = plt.subplots(ncols=4, figsize=(20, 4))

h = ax5.hist2d(Grid_x1, Grid_y1, bins=C5max - C5min, range=[(C5min, C5max), (R3min, R3max)], norm=mpl.colors.LogNorm())
fig16.colorbar(h[3], ax=ax5)

h = ax6.hist2d(Grid_x1, Grid_y1, bins=C6max - C6min, range=[(C6min, C6max), (R3min, R3max)], norm=mpl.colors.LogNorm())
fig16.colorbar(h[3], ax=ax6)

h = ax7.hist2d(Grid_x1, Grid_y1, bins=C7max - C7min, range=[(C7min, C7max), (R3min, R3max)], norm=mpl.colors.LogNorm())
fig16.colorbar(h[3], ax=ax7)

h = ax8.hist2d(Grid_x1, Grid_y1, bins=C8max - C8min, range=[(C8min, C8max), (R3min, R3max)], norm=mpl.colors.LogNorm())
fig16.colorbar(h[3], ax=ax8)

fig16.tight_layout()
plt.show()

# Sort arrays into row 3 array


# Initialize trash arrays
row3_x1trash = []
row3_y1trash = []
row3_t1trash = []
row3_a1trash = []
row3_A1trash = []

# Sort out points NOT in row, add to trash
for i in range(len(Grid_x1) - 1):
    if R3min > y1[i]:
        row3_x1trash.append(i)
        row3_y1trash.append(i)
        row3_t1trash.append(i)
        row3_a1trash.append(i)
        row3_A1trash.append(i)
    elif R3max < y1[i]:
        row3_x1trash.append(i)
        row3_y1trash.append(i)
        row3_t1trash.append(i)
        row3_a1trash.append(i)
        row3_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
row3_x1 = np.delete(Grid_x1, row3_x1trash)
row3_y1 = np.delete(Grid_y1, row3_y1trash)
row3_t1 = np.delete(Grid_t1, row3_t1trash)
row3_a1 = np.delete(Grid_a1, row3_a1trash)
row3_A1 = np.delete(Grid_A1, row3_A1trash)

# Sanity Check
print(len(Grid_x1), len(row3_x1trash), len(row3_x1))

# Sort Row 3 array into points


# Initialize trash arrays
C1R3_x1trash = []
C1R3_y1trash = []
C1R3_t1trash = []
C1R3_a1trash = []
C1R3_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row3_x1) - 1):
    if C1min > row3_x1[i]:
        C1R3_x1trash.append(i)
        C1R3_y1trash.append(i)
        C1R3_t1trash.append(i)
        C1R3_a1trash.append(i)
        C1R3_A1trash.append(i)
    elif C1max < row3_x1[i]:
        C1R3_x1trash.append(i)
        C1R3_y1trash.append(i)
        C1R3_t1trash.append(i)
        C1R3_a1trash.append(i)
        C1R3_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C1R3_x1 = np.delete(row3_x1, C1R3_x1trash)
C1R3_y1 = np.delete(row3_y1, C1R3_y1trash)
C1R3_t1 = np.delete(row3_t1, C1R3_t1trash)
C1R3_a1 = np.delete(row3_a1, C1R3_a1trash)
C1R3_A1 = np.delete(row3_A1, C1R3_A1trash)

# Sanity Check
print(len(row3_x1), len(C1R3_x1trash))
print('Point 1 Rate (Hz):', len(C1R3_x1))

# save rate in case it is needed later
rate_C1R3 = len(C1R3_x1)

# Initialize trash arrays
C2R3_x1trash = []
C2R3_y1trash = []
C2R3_t1trash = []
C2R3_a1trash = []
C2R3_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row3_x1) - 1):
    if C2min > row3_x1[i]:
        C2R3_x1trash.append(i)
        C2R3_y1trash.append(i)
        C2R3_t1trash.append(i)
        C2R3_a1trash.append(i)
        C2R3_A1trash.append(i)
    elif C2max < row3_x1[i]:
        C2R3_x1trash.append(i)
        C2R3_y1trash.append(i)
        C2R3_t1trash.append(i)
        C2R3_a1trash.append(i)
        C2R3_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C2R3_x1 = np.delete(row3_x1, C2R3_x1trash)
C2R3_y1 = np.delete(row3_y1, C2R3_y1trash)
C2R3_t1 = np.delete(row3_t1, C2R3_t1trash)
C2R3_a1 = np.delete(row3_a1, C2R3_a1trash)
C2R3_A1 = np.delete(row3_A1, C2R3_A1trash)

# Sanity Check
print(len(row3_x1), len(C2R3_x1trash))
print('Point 2 Rate (Hz):', len(C2R3_x1))

# save rate in case it is needed later
rate_C2R3 = len(C2R3_x1)

# Initialize trash arrays
C3R3_x1trash = []
C3R3_y1trash = []
C3R3_t1trash = []
C3R3_a1trash = []
C3R3_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row3_x1) - 1):
    if C3min > row3_x1[i]:
        C3R3_x1trash.append(i)
        C3R3_y1trash.append(i)
        C3R3_t1trash.append(i)
        C3R3_a1trash.append(i)
        C3R3_A1trash.append(i)
    elif C3max < row3_x1[i]:
        C3R3_x1trash.append(i)
        C3R3_y1trash.append(i)
        C3R3_t1trash.append(i)
        C3R3_a1trash.append(i)
        C3R3_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C3R3_x1 = np.delete(row3_x1, C3R3_x1trash)
C3R3_y1 = np.delete(row3_y1, C3R3_y1trash)
C3R3_t1 = np.delete(row3_t1, C3R3_t1trash)
C3R3_a1 = np.delete(row3_a1, C3R3_a1trash)
C3R3_A1 = np.delete(row3_A1, C3R3_A1trash)

# Sanity Check
print(len(row3_x1), len(C3R3_x1trash))
print('Point 3 Rate (Hz):', len(C3R3_x1))

# save rate in case it is needed later
rate_C3R3 = len(C3R3_x1)

# Initialize trash arrays
C4R3_x1trash = []
C4R3_y1trash = []
C4R3_t1trash = []
C4R3_a1trash = []
C4R3_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row3_x1) - 1):
    if C4min > row3_x1[i]:
        C4R3_x1trash.append(i)
        C4R3_y1trash.append(i)
        C4R3_t1trash.append(i)
        C4R3_a1trash.append(i)
        C4R3_A1trash.append(i)
    elif C4max < row3_x1[i]:
        C4R3_x1trash.append(i)
        C4R3_y1trash.append(i)
        C4R3_t1trash.append(i)
        C4R3_a1trash.append(i)
        C4R3_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C4R3_x1 = np.delete(row3_x1, C4R3_x1trash)
C4R3_y1 = np.delete(row3_y1, C4R3_y1trash)
C4R3_t1 = np.delete(row3_t1, C4R3_t1trash)
C4R3_a1 = np.delete(row3_a1, C4R3_a1trash)
C4R3_A1 = np.delete(row3_A1, C4R3_A1trash)

# Sanity Check
print(len(row3_x1), len(C4R3_x1trash))
print('Point 4 Rate (Hz):', len(C4R3_x1))

# save rate in case it is needed later
rate_C4R3 = len(C4R3_x1)

# Initialize trash arrays
C5R3_x1trash = []
C5R3_y1trash = []
C5R3_t1trash = []
C5R3_a1trash = []
C5R3_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row3_x1) - 1):
    if C5min > row3_x1[i]:
        C5R3_x1trash.append(i)
        C5R3_y1trash.append(i)
        C5R3_t1trash.append(i)
        C5R3_a1trash.append(i)
        C5R3_A1trash.append(i)
    elif C5max < row3_x1[i]:
        C5R3_x1trash.append(i)
        C5R3_y1trash.append(i)
        C5R3_t1trash.append(i)
        C5R3_a1trash.append(i)
        C5R3_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C5R3_x1 = np.delete(row3_x1, C5R3_x1trash)
C5R3_y1 = np.delete(row3_y1, C5R3_y1trash)
C5R3_t1 = np.delete(row3_t1, C5R3_t1trash)
C5R3_a1 = np.delete(row3_a1, C5R3_a1trash)
C5R3_A1 = np.delete(row3_A1, C5R3_A1trash)

# Sanity Check
print(len(row3_x1), len(C5R3_x1trash))
print('Point 5 Rate (Hz):', len(C5R3_x1))

# save rate in case it is needed later
rate_C5R3 = len(C5R3_x1)

# Initialize trash arrays
C6R3_x1trash = []
C6R3_y1trash = []
C6R3_t1trash = []
C6R3_a1trash = []
C6R3_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row3_x1) - 1):
    if C6min > row3_x1[i]:
        C6R3_x1trash.append(i)
        C6R3_y1trash.append(i)
        C6R3_t1trash.append(i)
        C6R3_a1trash.append(i)
        C6R3_A1trash.append(i)
    elif C6max < row3_x1[i]:
        C6R3_x1trash.append(i)
        C6R3_y1trash.append(i)
        C6R3_t1trash.append(i)
        C6R3_a1trash.append(i)
        C6R3_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C6R3_x1 = np.delete(row3_x1, C6R3_x1trash)
C6R3_y1 = np.delete(row3_y1, C6R3_y1trash)
C6R3_t1 = np.delete(row3_t1, C6R3_t1trash)
C6R3_a1 = np.delete(row3_a1, C6R3_a1trash)
C6R3_A1 = np.delete(row3_A1, C6R3_A1trash)

# Sanity Check
print(len(row3_x1), len(C6R3_x1trash))
print('Point 6 Rate (Hz):', len(C6R3_x1))

# save rate in case it is needed later
rate_C6R3 = len(C6R3_x1)

# Initialize trash arrays
C7R3_x1trash = []
C7R3_y1trash = []
C7R3_t1trash = []
C7R3_a1trash = []
C7R3_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row3_x1) - 1):
    if C7min > row3_x1[i]:
        C7R3_x1trash.append(i)
        C7R3_y1trash.append(i)
        C7R3_t1trash.append(i)
        C7R3_a1trash.append(i)
        C7R3_A1trash.append(i)
    elif C7max < row3_x1[i]:
        C7R3_x1trash.append(i)
        C7R3_y1trash.append(i)
        C7R3_t1trash.append(i)
        C7R3_a1trash.append(i)
        C7R3_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C7R3_x1 = np.delete(row3_x1, C7R3_x1trash)
C7R3_y1 = np.delete(row3_y1, C7R3_y1trash)
C7R3_t1 = np.delete(row3_t1, C7R3_t1trash)
C7R3_a1 = np.delete(row3_a1, C7R3_a1trash)
C7R3_A1 = np.delete(row3_A1, C7R3_A1trash)

# Sanity Check
print(len(row3_x1), len(C7R3_x1trash))
print('Point 7 Rate (Hz):', len(C7R3_x1))

# save rate in case it is needed later
rate_C7R3 = len(C7R3_x1)

# Initialize trash arrays
C8R3_x1trash = []
C8R3_y1trash = []
C8R3_t1trash = []
C8R3_a1trash = []
C8R3_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row3_x1) - 1):
    if C8min > row3_x1[i]:
        C8R3_x1trash.append(i)
        C8R3_y1trash.append(i)
        C8R3_t1trash.append(i)
        C8R3_a1trash.append(i)
        C8R3_A1trash.append(i)
    elif C8max < row3_x1[i]:
        C8R3_x1trash.append(i)
        C8R3_y1trash.append(i)
        C8R3_t1trash.append(i)
        C8R3_a1trash.append(i)
        C8R3_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C8R3_x1 = np.delete(row3_x1, C8R3_x1trash)
C8R3_y1 = np.delete(row3_y1, C8R3_y1trash)
C8R3_t1 = np.delete(row3_t1, C8R3_t1trash)
C8R3_a1 = np.delete(row3_a1, C8R3_a1trash)
C8R3_A1 = np.delete(row3_A1, C8R3_A1trash)

# Sanity Check
print(len(row3_x1), len(C8R3_x1trash))
print('Point 8 Rate (Hz):', len(C8R3_x1))

# save rate in case it is needed later
rate_C8R3 = len(C8R3_x1)

# Display Row 4

fig17, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 4))

h = ax1.hist2d(Grid_x1, Grid_y1, bins=C1max - C1min, range=[(C1min, C1max), (R4min, R4max)], norm=mpl.colors.LogNorm())
fig17.colorbar(h[3], ax=ax1)

h = ax2.hist2d(Grid_x1, Grid_y1, bins=C2max - C2min, range=[(C2min, C2max), (R4min, R4max)], norm=mpl.colors.LogNorm())
fig17.colorbar(h[3], ax=ax2)

h = ax3.hist2d(Grid_x1, Grid_y1, bins=C3max - C3min, range=[(C3min, C3max), (R4min, R4max)], norm=mpl.colors.LogNorm())
fig17.colorbar(h[3], ax=ax3)

h = ax4.hist2d(Grid_x1, Grid_y1, bins=C4max - C4min, range=[(C4min, C4max), (R4min, R4max)], norm=mpl.colors.LogNorm())
fig17.colorbar(h[3], ax=ax4)

fig17.tight_layout()
plt.show()

fig18, (ax5, ax6, ax7, ax8) = plt.subplots(ncols=4, figsize=(20, 4))

h = ax5.hist2d(Grid_x1, Grid_y1, bins=C5max - C5min, range=[(C5min, C5max), (R4min, R4max)], norm=mpl.colors.LogNorm())
fig18.colorbar(h[3], ax=ax5)

h = ax6.hist2d(Grid_x1, Grid_y1, bins=C6max - C6min, range=[(C6min, C6max), (R4min, R4max)], norm=mpl.colors.LogNorm())
fig18.colorbar(h[3], ax=ax6)

h = ax7.hist2d(Grid_x1, Grid_y1, bins=C7max - C7min, range=[(C7min, C7max), (R4min, R4max)], norm=mpl.colors.LogNorm())
fig18.colorbar(h[3], ax=ax7)

h = ax8.hist2d(Grid_x1, Grid_y1, bins=C8max - C8min, range=[(C8min, C8max), (R4min, R4max)], norm=mpl.colors.LogNorm())
fig18.colorbar(h[3], ax=ax8)

fig18.tight_layout()
plt.show()

# Sort arrays into row 4 array


# Initialize trash arrays
row4_x1trash = []
row4_y1trash = []
row4_t1trash = []
row4_a1trash = []
row4_A1trash = []

# Sort out points NOT in row, add to trash
for i in range(len(Grid_x1) - 1):
    if R4min > y1[i]:
        row4_x1trash.append(i)
        row4_y1trash.append(i)
        row4_t1trash.append(i)
        row4_a1trash.append(i)
        row4_A1trash.append(i)
    elif R4max < y1[i]:
        row4_x1trash.append(i)
        row4_y1trash.append(i)
        row4_t1trash.append(i)
        row4_a1trash.append(i)
        row4_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
row4_x1 = np.delete(Grid_x1, row4_x1trash)
row4_y1 = np.delete(Grid_y1, row4_y1trash)
row4_t1 = np.delete(Grid_t1, row4_t1trash)
row4_a1 = np.delete(Grid_a1, row4_a1trash)
row4_A1 = np.delete(Grid_A1, row4_A1trash)

# Sanity Check
print(len(Grid_x1), len(row4_x1trash), len(row4_x1))

# Sort Row 4 array into points


# Initialize trash arrays
C1R4_x1trash = []
C1R4_y1trash = []
C1R4_t1trash = []
C1R4_a1trash = []
C1R4_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row4_x1) - 1):
    if C1min > row4_x1[i]:
        C1R4_x1trash.append(i)
        C1R4_y1trash.append(i)
        C1R4_t1trash.append(i)
        C1R4_a1trash.append(i)
        C1R4_A1trash.append(i)
    elif C1max < row4_x1[i]:
        C1R4_x1trash.append(i)
        C1R4_y1trash.append(i)
        C1R4_t1trash.append(i)
        C1R4_a1trash.append(i)
        C1R4_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C1R4_x1 = np.delete(row4_x1, C1R4_x1trash)
C1R4_y1 = np.delete(row4_y1, C1R4_y1trash)
C1R4_t1 = np.delete(row4_t1, C1R4_t1trash)
C1R4_a1 = np.delete(row4_a1, C1R4_a1trash)
C1R4_A1 = np.delete(row4_A1, C1R4_A1trash)

# Sanity Check
print(len(row4_x1), len(C1R4_x1trash))
print('Point 1 Rate (Hz):', len(C1R4_x1))

# save rate in case it is needed later
rate_C1R4 = len(C1R4_x1)

# Initialize trash arrays
C2R4_x1trash = []
C2R4_y1trash = []
C2R4_t1trash = []
C2R4_a1trash = []
C2R4_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row4_x1) - 1):
    if C2min > row4_x1[i]:
        C2R4_x1trash.append(i)
        C2R4_y1trash.append(i)
        C2R4_t1trash.append(i)
        C2R4_a1trash.append(i)
        C2R4_A1trash.append(i)
    elif C2max < row4_x1[i]:
        C2R4_x1trash.append(i)
        C2R4_y1trash.append(i)
        C2R4_t1trash.append(i)
        C2R4_a1trash.append(i)
        C2R4_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C2R4_x1 = np.delete(row4_x1, C2R4_x1trash)
C2R4_y1 = np.delete(row4_y1, C2R4_y1trash)
C2R4_t1 = np.delete(row4_t1, C2R4_t1trash)
C2R4_a1 = np.delete(row4_a1, C2R4_a1trash)
C2R4_A1 = np.delete(row4_A1, C2R4_A1trash)

# Sanity Check
print(len(row4_x1), len(C2R4_x1trash))
print('Point 2 Rate (Hz):', len(C2R4_x1))

# save rate in case it is needed later
rate_C2R4 = len(C2R4_x1)

# Initialize trash arrays
C3R4_x1trash = []
C3R4_y1trash = []
C3R4_t1trash = []
C3R4_a1trash = []
C3R4_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row4_x1) - 1):
    if C3min > row4_x1[i]:
        C3R4_x1trash.append(i)
        C3R4_y1trash.append(i)
        C3R4_t1trash.append(i)
        C3R4_a1trash.append(i)
        C3R4_A1trash.append(i)
    elif C3max < row4_x1[i]:
        C3R4_x1trash.append(i)
        C3R4_y1trash.append(i)
        C3R4_t1trash.append(i)
        C3R4_a1trash.append(i)
        C3R4_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C3R4_x1 = np.delete(row4_x1, C3R4_x1trash)
C3R4_y1 = np.delete(row4_y1, C3R4_y1trash)
C3R4_t1 = np.delete(row4_t1, C3R4_t1trash)
C3R4_a1 = np.delete(row4_a1, C3R4_a1trash)
C3R4_A1 = np.delete(row4_A1, C3R4_A1trash)

# Sanity Check
print(len(row4_x1), len(C3R4_x1trash))
print('Point 3 Rate (Hz):', len(C3R4_x1))

# save rate in case it is needed later
rate_C3R4 = len(C3R4_x1)

# Initialize trash arrays
C4R4_x1trash = []
C4R4_y1trash = []
C4R4_t1trash = []
C4R4_a1trash = []
C4R4_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row4_x1) - 1):
    if C4min > row4_x1[i]:
        C4R4_x1trash.append(i)
        C4R4_y1trash.append(i)
        C4R4_t1trash.append(i)
        C4R4_a1trash.append(i)
        C4R4_A1trash.append(i)
    elif C4max < row4_x1[i]:
        C4R4_x1trash.append(i)
        C4R4_y1trash.append(i)
        C4R4_t1trash.append(i)
        C4R4_a1trash.append(i)
        C4R4_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C4R4_x1 = np.delete(row4_x1, C4R4_x1trash)
C4R4_y1 = np.delete(row4_y1, C4R4_y1trash)
C4R4_t1 = np.delete(row4_t1, C4R4_t1trash)
C4R4_a1 = np.delete(row4_a1, C4R4_a1trash)
C4R4_A1 = np.delete(row4_A1, C4R4_A1trash)

# Sanity Check
print(len(row4_x1), len(C4R4_x1trash))
print('Point 4 Rate (Hz):', len(C4R4_x1))

# save rate in case it is needed later
rate_C4R4 = len(C4R4_x1)

# Initialize trash arrays
C5R4_x1trash = []
C5R4_y1trash = []
C5R4_t1trash = []
C5R4_a1trash = []
C5R4_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row4_x1) - 1):
    if C5min > row4_x1[i]:
        C5R4_x1trash.append(i)
        C5R4_y1trash.append(i)
        C5R4_t1trash.append(i)
        C5R4_a1trash.append(i)
        C5R4_A1trash.append(i)
    elif C5max < row4_x1[i]:
        C5R4_x1trash.append(i)
        C5R4_y1trash.append(i)
        C5R4_t1trash.append(i)
        C5R4_a1trash.append(i)
        C5R4_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C5R4_x1 = np.delete(row4_x1, C5R4_x1trash)
C5R4_y1 = np.delete(row4_y1, C5R4_y1trash)
C5R4_t1 = np.delete(row4_t1, C5R4_t1trash)
C5R4_a1 = np.delete(row4_a1, C5R4_a1trash)
C5R4_A1 = np.delete(row4_A1, C5R4_A1trash)

# Sanity Check
print(len(row4_x1), len(C5R4_x1trash))
print('Point 5 Rate (Hz):', len(C5R4_x1))

# save rate in case it is needed later
rate_C5R4 = len(C5R4_x1)

# Initialize trash arrays
C6R4_x1trash = []
C6R4_y1trash = []
C6R4_t1trash = []
C6R4_a1trash = []
C6R4_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row4_x1) - 1):
    if C6min > row4_x1[i]:
        C6R4_x1trash.append(i)
        C6R4_y1trash.append(i)
        C6R4_t1trash.append(i)
        C6R4_a1trash.append(i)
        C6R4_A1trash.append(i)
    elif C6max < row4_x1[i]:
        C6R4_x1trash.append(i)
        C6R4_y1trash.append(i)
        C6R4_t1trash.append(i)
        C6R4_a1trash.append(i)
        C6R4_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C6R4_x1 = np.delete(row4_x1, C6R4_x1trash)
C6R4_y1 = np.delete(row4_y1, C6R4_y1trash)
C6R4_t1 = np.delete(row4_t1, C6R4_t1trash)
C6R4_a1 = np.delete(row4_a1, C6R4_a1trash)
C6R4_A1 = np.delete(row4_A1, C6R4_A1trash)

# Sanity Check
print(len(row4_x1), len(C6R4_x1trash))
print('Point 6 Rate (Hz):', len(C6R4_x1))

# save rate in case it is needed later
rate_C6R4 = len(C6R4_x1)

# Initialize trash arrays
C7R4_x1trash = []
C7R4_y1trash = []
C7R4_t1trash = []
C7R4_a1trash = []
C7R4_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row4_x1) - 1):
    if C7min > row4_x1[i]:
        C7R4_x1trash.append(i)
        C7R4_y1trash.append(i)
        C7R4_t1trash.append(i)
        C7R4_a1trash.append(i)
        C7R4_A1trash.append(i)
    elif C7max < row4_x1[i]:
        C7R4_x1trash.append(i)
        C7R4_y1trash.append(i)
        C7R4_t1trash.append(i)
        C7R4_a1trash.append(i)
        C7R4_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C7R4_x1 = np.delete(row4_x1, C7R4_x1trash)
C7R4_y1 = np.delete(row4_y1, C7R4_y1trash)
C7R4_t1 = np.delete(row4_t1, C7R4_t1trash)
C7R4_a1 = np.delete(row4_a1, C7R4_a1trash)
C7R4_A1 = np.delete(row4_A1, C7R4_A1trash)

# Sanity Check
print(len(row4_x1), len(C7R4_x1trash))
print('Point 7 Rate (Hz):', len(C7R4_x1))

# save rate in case it is needed later
rate_C7R4 = len(C7R4_x1)

# Initialize trash arrays
C8R4_x1trash = []
C8R4_y1trash = []
C8R4_t1trash = []
C8R4_a1trash = []
C8R4_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row4_x1) - 1):
    if C8min > row4_x1[i]:
        C8R4_x1trash.append(i)
        C8R4_y1trash.append(i)
        C8R4_t1trash.append(i)
        C8R4_a1trash.append(i)
        C8R4_A1trash.append(i)
    elif C8max < row4_x1[i]:
        C8R4_x1trash.append(i)
        C8R4_y1trash.append(i)
        C8R4_t1trash.append(i)
        C8R4_a1trash.append(i)
        C8R4_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C8R4_x1 = np.delete(row4_x1, C8R4_x1trash)
C8R4_y1 = np.delete(row4_y1, C8R4_y1trash)
C8R4_t1 = np.delete(row4_t1, C8R4_t1trash)
C8R4_a1 = np.delete(row4_a1, C8R4_a1trash)
C8R4_A1 = np.delete(row4_A1, C8R4_A1trash)

# Sanity Check
print(len(row4_x1), len(C8R4_x1trash))
print('Point 8 Rate (Hz):', len(C8R4_x1))

# save rate in case it is needed later
rate_C8R4 = len(C8R4_x1)

# Display Row 5

fig19, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 4))

h = ax1.hist2d(Grid_x1, Grid_y1, bins=C1max - C1min, range=[(C1min, C1max), (R5min, R5max)], norm=mpl.colors.LogNorm())
fig19.colorbar(h[3], ax=ax1)

h = ax2.hist2d(Grid_x1, Grid_y1, bins=C2max - C2min, range=[(C2min, C2max), (R5min, R5max)], norm=mpl.colors.LogNorm())
fig19.colorbar(h[3], ax=ax2)

h = ax3.hist2d(Grid_x1, Grid_y1, bins=C3max - C3min, range=[(C3min, C3max), (R5min, R5max)], norm=mpl.colors.LogNorm())
fig19.colorbar(h[3], ax=ax3)

h = ax4.hist2d(Grid_x1, Grid_y1, bins=C4max - C4min, range=[(C4min, C4max), (R5min, R5max)], norm=mpl.colors.LogNorm())
fig19.colorbar(h[3], ax=ax4)

fig19.tight_layout()
plt.show()

fig20, (ax5, ax6, ax7, ax8) = plt.subplots(ncols=4, figsize=(20, 4))

h = ax5.hist2d(Grid_x1, Grid_y1, bins=C5max - C5min, range=[(C5min, C5max), (R5min, R5max)], norm=mpl.colors.LogNorm())
fig20.colorbar(h[3], ax=ax5)

h = ax6.hist2d(Grid_x1, Grid_y1, bins=C6max - C6min, range=[(C6min, C6max), (R5min, R5max)], norm=mpl.colors.LogNorm())
fig20.colorbar(h[3], ax=ax6)

h = ax7.hist2d(Grid_x1, Grid_y1, bins=C7max - C7min, range=[(C7min, C7max), (R5min, R5max)], norm=mpl.colors.LogNorm())
fig20.colorbar(h[3], ax=ax7)

h = ax8.hist2d(Grid_x1, Grid_y1, bins=C8max - C8min, range=[(C8min, C8max), (R5min, R5max)], norm=mpl.colors.LogNorm())
fig20.colorbar(h[3], ax=ax8)

fig20.tight_layout()
plt.show()

# Sort arrays into row 5 array


# Initialize trash arrays
row5_x1trash = []
row5_y1trash = []
row5_t1trash = []
row5_a1trash = []
row5_A1trash = []

# Sort out points NOT in row, add to trash
for i in range(len(Grid_x1) - 1):
    if R5min > y1[i]:
        row5_x1trash.append(i)
        row5_y1trash.append(i)
        row5_t1trash.append(i)
        row5_a1trash.append(i)
        row5_A1trash.append(i)
    elif R5max < y1[i]:
        row5_x1trash.append(i)
        row5_y1trash.append(i)
        row5_t1trash.append(i)
        row5_a1trash.append(i)
        row5_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
row5_x1 = np.delete(Grid_x1, row5_x1trash)
row5_y1 = np.delete(Grid_y1, row5_y1trash)
row5_t1 = np.delete(Grid_t1, row5_t1trash)
row5_a1 = np.delete(Grid_a1, row5_a1trash)
row5_A1 = np.delete(Grid_A1, row5_A1trash)

# Sanity Check
print(len(Grid_x1), len(row5_x1trash), len(row5_x1))

# Sort Row 5 array into points


# Initialize trash arrays
C1R5_x1trash = []
C1R5_y1trash = []
C1R5_t1trash = []
C1R5_a1trash = []
C1R5_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row5_x1) - 1):
    if C1min > row5_x1[i]:
        C1R5_x1trash.append(i)
        C1R5_y1trash.append(i)
        C1R5_t1trash.append(i)
        C1R5_a1trash.append(i)
        C1R5_A1trash.append(i)
    elif C1max < row5_x1[i]:
        C1R5_x1trash.append(i)
        C1R5_y1trash.append(i)
        C1R5_t1trash.append(i)
        C1R5_a1trash.append(i)
        C1R5_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C1R5_x1 = np.delete(row5_x1, C1R5_x1trash)
C1R5_y1 = np.delete(row5_y1, C1R5_y1trash)
C1R5_t1 = np.delete(row5_t1, C1R5_t1trash)
C1R5_a1 = np.delete(row5_a1, C1R5_a1trash)
C1R5_A1 = np.delete(row5_A1, C1R5_A1trash)

# Sanity Check
print(len(row5_x1), len(C1R5_x1trash))
print('Point 1 Rate (Hz):', len(C1R5_x1))

# save rate in case it is needed later
rate_C1R5 = len(C1R5_x1)

# Initialize trash arrays
C2R5_x1trash = []
C2R5_y1trash = []
C2R5_t1trash = []
C2R5_a1trash = []
C2R5_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row5_x1) - 1):
    if C2min > row5_x1[i]:
        C2R5_x1trash.append(i)
        C2R5_y1trash.append(i)
        C2R5_t1trash.append(i)
        C2R5_a1trash.append(i)
        C2R5_A1trash.append(i)
    elif C2max < row5_x1[i]:
        C2R5_x1trash.append(i)
        C2R5_y1trash.append(i)
        C2R5_t1trash.append(i)
        C2R5_a1trash.append(i)
        C2R5_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C2R5_x1 = np.delete(row5_x1, C2R5_x1trash)
C2R5_y1 = np.delete(row5_y1, C2R5_y1trash)
C2R5_t1 = np.delete(row5_t1, C2R5_t1trash)
C2R5_a1 = np.delete(row5_a1, C2R5_a1trash)
C2R5_A1 = np.delete(row5_A1, C2R5_A1trash)

# Sanity Check
print(len(row5_x1), len(C2R5_x1trash))
print('Point 2 Rate (Hz):', len(C2R5_x1))

# save rate in case it is needed later
rate_C2R5 = len(C2R5_x1)

# Initialize trash arrays
C3R5_x1trash = []
C3R5_y1trash = []
C3R5_t1trash = []
C3R5_a1trash = []
C3R5_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row5_x1) - 1):
    if C3min > row5_x1[i]:
        C3R5_x1trash.append(i)
        C3R5_y1trash.append(i)
        C3R5_t1trash.append(i)
        C3R5_a1trash.append(i)
        C3R5_A1trash.append(i)
    elif C3max < row5_x1[i]:
        C3R5_x1trash.append(i)
        C3R5_y1trash.append(i)
        C3R5_t1trash.append(i)
        C3R5_a1trash.append(i)
        C3R5_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C3R5_x1 = np.delete(row5_x1, C3R5_x1trash)
C3R5_y1 = np.delete(row5_y1, C3R5_y1trash)
C3R5_t1 = np.delete(row5_t1, C3R5_t1trash)
C3R5_a1 = np.delete(row5_a1, C3R5_a1trash)
C3R5_A1 = np.delete(row5_A1, C3R5_A1trash)

# Sanity Check
print(len(row5_x1), len(C3R5_x1trash))
print('Point 3 Rate (Hz):', len(C3R5_x1))

# save rate in case it is needed later
rate_C3R5 = len(C3R5_x1)

# Initialize trash arrays
C4R5_x1trash = []
C4R5_y1trash = []
C4R5_t1trash = []
C4R5_a1trash = []
C4R5_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row5_x1) - 1):
    if C4min > row5_x1[i]:
        C4R5_x1trash.append(i)
        C4R5_y1trash.append(i)
        C4R5_t1trash.append(i)
        C4R5_a1trash.append(i)
        C4R5_A1trash.append(i)
    elif C4max < row5_x1[i]:
        C4R5_x1trash.append(i)
        C4R5_y1trash.append(i)
        C4R5_t1trash.append(i)
        C4R5_a1trash.append(i)
        C4R5_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C4R5_x1 = np.delete(row5_x1, C4R5_x1trash)
C4R5_y1 = np.delete(row5_y1, C4R5_y1trash)
C4R5_t1 = np.delete(row5_t1, C4R5_t1trash)
C4R5_a1 = np.delete(row5_a1, C4R5_a1trash)
C4R5_A1 = np.delete(row5_A1, C4R5_A1trash)

# Sanity Check
print(len(row5_x1), len(C4R5_x1trash))
print('Point 4 Rate (Hz):', len(C4R5_x1))

# save rate in case it is needed later
rate_C4R5 = len(C4R5_x1)

# Initialize trash arrays
C5R5_x1trash = []
C5R5_y1trash = []
C5R5_t1trash = []
C5R5_a1trash = []
C5R5_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row5_x1) - 1):
    if C5min > row5_x1[i]:
        C5R5_x1trash.append(i)
        C5R5_y1trash.append(i)
        C5R5_t1trash.append(i)
        C5R5_a1trash.append(i)
        C5R5_A1trash.append(i)
    elif C5max < row5_x1[i]:
        C5R5_x1trash.append(i)
        C5R5_y1trash.append(i)
        C5R5_t1trash.append(i)
        C5R5_a1trash.append(i)
        C5R5_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C5R5_x1 = np.delete(row5_x1, C5R5_x1trash)
C5R5_y1 = np.delete(row5_y1, C5R5_y1trash)
C5R5_t1 = np.delete(row5_t1, C5R5_t1trash)
C5R5_a1 = np.delete(row5_a1, C5R5_a1trash)
C5R5_A1 = np.delete(row5_A1, C5R5_A1trash)

# Sanity Check
print(len(row5_x1), len(C5R5_x1trash))
print('Point 5 Rate (Hz):', len(C5R5_x1))

# save rate in case it is needed later
rate_C5R5 = len(C5R5_x1)

# Initialize trash arrays
C6R5_x1trash = []
C6R5_y1trash = []
C6R5_t1trash = []
C6R5_a1trash = []
C6R5_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row5_x1) - 1):
    if C6min > row5_x1[i]:
        C6R5_x1trash.append(i)
        C6R5_y1trash.append(i)
        C6R5_t1trash.append(i)
        C6R5_a1trash.append(i)
        C6R5_A1trash.append(i)
    elif C6max < row5_x1[i]:
        C6R5_x1trash.append(i)
        C6R5_y1trash.append(i)
        C6R5_t1trash.append(i)
        C6R5_a1trash.append(i)
        C6R5_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C6R5_x1 = np.delete(row5_x1, C6R5_x1trash)
C6R5_y1 = np.delete(row5_y1, C6R5_y1trash)
C6R5_t1 = np.delete(row5_t1, C6R5_t1trash)
C6R5_a1 = np.delete(row5_a1, C6R5_a1trash)
C6R5_A1 = np.delete(row5_A1, C6R5_A1trash)

# Sanity Check
print(len(row5_x1), len(C6R5_x1trash))
print('Point 6 Rate (Hz):', len(C6R5_x1))

# save rate in case it is needed later
rate_C6R5 = len(C6R5_x1)

# Initialize trash arrays
C7R5_x1trash = []
C7R5_y1trash = []
C7R5_t1trash = []
C7R5_a1trash = []
C7R5_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row5_x1) - 1):
    if C7min > row5_x1[i]:
        C7R5_x1trash.append(i)
        C7R5_y1trash.append(i)
        C7R5_t1trash.append(i)
        C7R5_a1trash.append(i)
        C7R5_A1trash.append(i)
    elif C7max < row5_x1[i]:
        C7R5_x1trash.append(i)
        C7R5_y1trash.append(i)
        C7R5_t1trash.append(i)
        C7R5_a1trash.append(i)
        C7R5_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C7R5_x1 = np.delete(row5_x1, C7R5_x1trash)
C7R5_y1 = np.delete(row5_y1, C7R5_y1trash)
C7R5_t1 = np.delete(row5_t1, C7R5_t1trash)
C7R5_a1 = np.delete(row5_a1, C7R5_a1trash)
C7R5_A1 = np.delete(row5_A1, C7R5_A1trash)

# Sanity Check
print(len(row5_x1), len(C7R5_x1trash))
print('Point 7 Rate (Hz):', len(C7R5_x1))

# save rate in case it is needed later
rate_C7R5 = len(C7R5_x1)

# Initialize trash arrays
C8R5_x1trash = []
C8R5_y1trash = []
C8R5_t1trash = []
C8R5_a1trash = []
C8R5_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row5_x1) - 1):
    if C8min > row5_x1[i]:
        C8R5_x1trash.append(i)
        C8R5_y1trash.append(i)
        C8R5_t1trash.append(i)
        C8R5_a1trash.append(i)
        C8R5_A1trash.append(i)
    elif C8max < row5_x1[i]:
        C8R5_x1trash.append(i)
        C8R5_y1trash.append(i)
        C8R5_t1trash.append(i)
        C8R5_a1trash.append(i)
        C8R5_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C8R5_x1 = np.delete(row5_x1, C8R5_x1trash)
C8R5_y1 = np.delete(row5_y1, C8R5_y1trash)
C8R5_t1 = np.delete(row5_t1, C8R5_t1trash)
C8R5_a1 = np.delete(row5_a1, C8R5_a1trash)
C8R5_A1 = np.delete(row5_A1, C8R5_A1trash)

# Sanity Check
print(len(row5_x1), len(C8R5_x1trash))
print('Point 8 Rate (Hz):', len(C8R5_x1))

# save rate in case it is needed later
rate_C8R5 = len(C8R5_x1)

# Display Row 6

fig21, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 4))

h = ax1.hist2d(Grid_x1, Grid_y1, bins=C1max - C1min, range=[(C1min, C1max), (R6min, R6max)], norm=mpl.colors.LogNorm())
fig21.colorbar(h[3], ax=ax1)

h = ax2.hist2d(Grid_x1, Grid_y1, bins=C2max - C2min, range=[(C2min, C2max), (R6min, R6max)], norm=mpl.colors.LogNorm())
fig21.colorbar(h[3], ax=ax2)

h = ax3.hist2d(Grid_x1, Grid_y1, bins=C3max - C3min, range=[(C3min, C3max), (R6min, R6max)], norm=mpl.colors.LogNorm())
fig21.colorbar(h[3], ax=ax3)

h = ax4.hist2d(Grid_x1, Grid_y1, bins=C4max - C4min, range=[(C4min, C4max), (R6min, R6max)], norm=mpl.colors.LogNorm())
fig21.colorbar(h[3], ax=ax4)

fig21.tight_layout()
plt.show()

fig22, (ax5, ax6, ax7, ax8) = plt.subplots(ncols=4, figsize=(20, 4))

h = ax5.hist2d(Grid_x1, Grid_y1, bins=C5max - C5min, range=[(C5min, C5max), (R6min, R6max)], norm=mpl.colors.LogNorm())
fig22.colorbar(h[3], ax=ax5)

h = ax6.hist2d(Grid_x1, Grid_y1, bins=C6max - C6min, range=[(C6min, C6max), (R6min, R6max)], norm=mpl.colors.LogNorm())
fig22.colorbar(h[3], ax=ax6)

h = ax7.hist2d(Grid_x1, Grid_y1, bins=C7max - C7min, range=[(C7min, C7max), (R6min, R6max)], norm=mpl.colors.LogNorm())
fig22.colorbar(h[3], ax=ax7)

h = ax8.hist2d(Grid_x1, Grid_y1, bins=C8max - C8min, range=[(C8min, C8max), (R6min, R6max)], norm=mpl.colors.LogNorm())
fig22.colorbar(h[3], ax=ax8)

fig22.tight_layout()
plt.show()

# Sort arrays into row 6 array


# Initialize trash arrays
row6_x1trash = []
row6_y1trash = []
row6_t1trash = []
row6_a1trash = []
row6_A1trash = []

# Sort out points NOT in row, add to trash
for i in range(len(Grid_x1) - 1):
    if R6min > y1[i]:
        row6_x1trash.append(i)
        row6_y1trash.append(i)
        row6_t1trash.append(i)
        row6_a1trash.append(i)
        row6_A1trash.append(i)
    elif R6max < y1[i]:
        row6_x1trash.append(i)
        row6_y1trash.append(i)
        row6_t1trash.append(i)
        row6_a1trash.append(i)
        row6_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
row6_x1 = np.delete(Grid_x1, row6_x1trash)
row6_y1 = np.delete(Grid_y1, row6_y1trash)
row6_t1 = np.delete(Grid_t1, row6_t1trash)
row6_a1 = np.delete(Grid_a1, row6_a1trash)
row6_A1 = np.delete(Grid_A1, row6_A1trash)

# Sanity Check
print(len(Grid_x1), len(row6_x1trash), len(row6_x1))

# Sort Row 6 array into points


# Initialize trash arrays
C1R6_x1trash = []
C1R6_y1trash = []
C1R6_t1trash = []
C1R6_a1trash = []
C1R6_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row6_x1) - 1):
    if C1min > row6_x1[i]:
        C1R6_x1trash.append(i)
        C1R6_y1trash.append(i)
        C1R6_t1trash.append(i)
        C1R6_a1trash.append(i)
        C1R6_A1trash.append(i)
    elif C1max < row6_x1[i]:
        C1R6_x1trash.append(i)
        C1R6_y1trash.append(i)
        C1R6_t1trash.append(i)
        C1R6_a1trash.append(i)
        C1R6_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C1R6_x1 = np.delete(row6_x1, C1R6_x1trash)
C1R6_y1 = np.delete(row6_y1, C1R6_y1trash)
C1R6_t1 = np.delete(row6_t1, C1R6_t1trash)
C1R6_a1 = np.delete(row6_a1, C1R6_a1trash)
C1R6_A1 = np.delete(row6_A1, C1R6_A1trash)

# Sanity Check
print(len(row6_x1), len(C1R6_x1trash))
print('Point 1 Rate (Hz):', len(C1R6_x1))

# save rate in case it is needed later
rate_C1R6 = len(C1R6_x1)

# Initialize trash arrays
C2R6_x1trash = []
C2R6_y1trash = []
C2R6_t1trash = []
C2R6_a1trash = []
C2R6_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row6_x1) - 1):
    if C2min > row6_x1[i]:
        C2R6_x1trash.append(i)
        C2R6_y1trash.append(i)
        C2R6_t1trash.append(i)
        C2R6_a1trash.append(i)
        C2R6_A1trash.append(i)
    elif C2max < row6_x1[i]:
        C2R6_x1trash.append(i)
        C2R6_y1trash.append(i)
        C2R6_t1trash.append(i)
        C2R6_a1trash.append(i)
        C2R6_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C2R6_x1 = np.delete(row6_x1, C2R6_x1trash)
C2R6_y1 = np.delete(row6_y1, C2R6_y1trash)
C2R6_t1 = np.delete(row6_t1, C2R6_t1trash)
C2R6_a1 = np.delete(row6_a1, C2R6_a1trash)
C2R6_A1 = np.delete(row6_A1, C2R6_A1trash)

# Sanity Check
print(len(row6_x1), len(C2R6_x1trash))
print('Point 2 Rate (Hz):', len(C2R6_x1))

# save rate in case it is needed later
rate_C2R6 = len(C2R6_x1)

# Initialize trash arrays
C3R6_x1trash = []
C3R6_y1trash = []
C3R6_t1trash = []
C3R6_a1trash = []
C3R6_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row6_x1) - 1):
    if C3min > row6_x1[i]:
        C3R6_x1trash.append(i)
        C3R6_y1trash.append(i)
        C3R6_t1trash.append(i)
        C3R6_a1trash.append(i)
        C3R6_A1trash.append(i)
    elif C3max < row6_x1[i]:
        C3R6_x1trash.append(i)
        C3R6_y1trash.append(i)
        C3R6_t1trash.append(i)
        C3R6_a1trash.append(i)
        C3R6_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C3R6_x1 = np.delete(row6_x1, C3R6_x1trash)
C3R6_y1 = np.delete(row6_y1, C3R6_y1trash)
C3R6_t1 = np.delete(row6_t1, C3R6_t1trash)
C3R6_a1 = np.delete(row6_a1, C3R6_a1trash)
C3R6_A1 = np.delete(row6_A1, C3R6_A1trash)

# Sanity Check
print(len(row6_x1), len(C3R6_x1trash))
print('Point 3 Rate (Hz):', len(C3R6_x1))

# save rate in case it is needed later
rate_C3R6 = len(C3R6_x1)

# Initialize trash arrays
C4R6_x1trash = []
C4R6_y1trash = []
C4R6_t1trash = []
C4R6_a1trash = []
C4R6_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row6_x1) - 1):
    if C4min > row6_x1[i]:
        C4R6_x1trash.append(i)
        C4R6_y1trash.append(i)
        C4R6_t1trash.append(i)
        C4R6_a1trash.append(i)
        C4R6_A1trash.append(i)
    elif C4max < row6_x1[i]:
        C4R6_x1trash.append(i)
        C4R6_y1trash.append(i)
        C4R6_t1trash.append(i)
        C4R6_a1trash.append(i)
        C4R6_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C4R6_x1 = np.delete(row6_x1, C4R6_x1trash)
C4R6_y1 = np.delete(row6_y1, C4R6_y1trash)
C4R6_t1 = np.delete(row6_t1, C4R6_t1trash)
C4R6_a1 = np.delete(row6_a1, C4R6_a1trash)
C4R6_A1 = np.delete(row6_A1, C4R6_A1trash)

# Sanity Check
print(len(row6_x1), len(C4R6_x1trash))
print('Point 4 Rate (Hz):', len(C4R6_x1))

# save rate in case it is needed later
rate_C4R6 = len(C4R6_x1)

# Initialize trash arrays
C5R6_x1trash = []
C5R6_y1trash = []
C5R6_t1trash = []
C5R6_a1trash = []
C5R6_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row6_x1) - 1):
    if C5min > row6_x1[i]:
        C5R6_x1trash.append(i)
        C5R6_y1trash.append(i)
        C5R6_t1trash.append(i)
        C5R6_a1trash.append(i)
        C5R6_A1trash.append(i)
    elif C5max < row6_x1[i]:
        C5R6_x1trash.append(i)
        C5R6_y1trash.append(i)
        C5R6_t1trash.append(i)
        C5R6_a1trash.append(i)
        C5R6_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C5R6_x1 = np.delete(row6_x1, C5R6_x1trash)
C5R6_y1 = np.delete(row6_y1, C5R6_y1trash)
C5R6_t1 = np.delete(row6_t1, C5R6_t1trash)
C5R6_a1 = np.delete(row6_a1, C5R6_a1trash)
C5R6_A1 = np.delete(row6_A1, C5R6_A1trash)

# Sanity Check
print(len(row6_x1), len(C5R6_x1trash))
print('Point 5 Rate (Hz):', len(C5R6_x1))

# save rate in case it is needed later
rate_C5R6 = len(C5R6_x1)

# Initialize trash arrays
C6R6_x1trash = []
C6R6_y1trash = []
C6R6_t1trash = []
C6R6_a1trash = []
C6R6_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row6_x1) - 1):
    if C6min > row6_x1[i]:
        C6R6_x1trash.append(i)
        C6R6_y1trash.append(i)
        C6R6_t1trash.append(i)
        C6R6_a1trash.append(i)
        C6R6_A1trash.append(i)
    elif C6max < row6_x1[i]:
        C6R6_x1trash.append(i)
        C6R6_y1trash.append(i)
        C6R6_t1trash.append(i)
        C6R6_a1trash.append(i)
        C6R6_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C6R6_x1 = np.delete(row6_x1, C6R6_x1trash)
C6R6_y1 = np.delete(row6_y1, C6R6_y1trash)
C6R6_t1 = np.delete(row6_t1, C6R6_t1trash)
C6R6_a1 = np.delete(row6_a1, C6R6_a1trash)
C6R6_A1 = np.delete(row6_A1, C6R6_A1trash)

# Sanity Check
print(len(row6_x1), len(C6R6_x1trash))
print('Point 6 Rate (Hz):', len(C6R6_x1))

# save rate in case it is needed later
rate_C6R6 = len(C6R6_x1)

# Initialize trash arrays
C7R6_x1trash = []
C7R6_y1trash = []
C7R6_t1trash = []
C7R6_a1trash = []
C7R6_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row6_x1) - 1):
    if C7min > row6_x1[i]:
        C7R6_x1trash.append(i)
        C7R6_y1trash.append(i)
        C7R6_t1trash.append(i)
        C7R6_a1trash.append(i)
        C7R6_A1trash.append(i)
    elif C7max < row6_x1[i]:
        C7R6_x1trash.append(i)
        C7R6_y1trash.append(i)
        C7R6_t1trash.append(i)
        C7R6_a1trash.append(i)
        C7R6_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C7R6_x1 = np.delete(row6_x1, C7R6_x1trash)
C7R6_y1 = np.delete(row6_y1, C7R6_y1trash)
C7R6_t1 = np.delete(row6_t1, C7R6_t1trash)
C7R6_a1 = np.delete(row6_a1, C7R6_a1trash)
C7R6_A1 = np.delete(row6_A1, C7R6_A1trash)

# Sanity Check
print(len(row6_x1), len(C7R6_x1trash))
print('Point 7 Rate (Hz):', len(C7R6_x1))

# save rate in case it is needed later
rate_C7R6 = len(C7R6_x1)

# Initialize trash arrays
C8R6_x1trash = []
C8R6_y1trash = []
C8R6_t1trash = []
C8R6_a1trash = []
C8R6_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row6_x1) - 1):
    if C8min > row6_x1[i]:
        C8R6_x1trash.append(i)
        C8R6_y1trash.append(i)
        C8R6_t1trash.append(i)
        C8R6_a1trash.append(i)
        C8R6_A1trash.append(i)
    elif C8max < row6_x1[i]:
        C8R6_x1trash.append(i)
        C8R6_y1trash.append(i)
        C8R6_t1trash.append(i)
        C8R6_a1trash.append(i)
        C8R6_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C8R6_x1 = np.delete(row6_x1, C8R6_x1trash)
C8R6_y1 = np.delete(row6_y1, C8R6_y1trash)
C8R6_t1 = np.delete(row6_t1, C8R6_t1trash)
C8R6_a1 = np.delete(row6_a1, C8R6_a1trash)
C8R6_A1 = np.delete(row6_A1, C8R6_A1trash)

# Sanity Check
print(len(row6_x1), len(C8R6_x1trash))
print('Point 8 Rate (Hz):', len(C8R6_x1))

# save rate in case it is needed later
rate_C8R6 = len(C8R6_x1)

# Display Row 7

fig23, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 4))

h = ax1.hist2d(Grid_x1, Grid_y1, bins=C1max - C1min, range=[(C1min, C1max), (R7min, R7max)], norm=mpl.colors.LogNorm())
fig23.colorbar(h[3], ax=ax1)

h = ax2.hist2d(Grid_x1, Grid_y1, bins=C2max - C2min, range=[(C2min, C2max), (R7min, R7max)], norm=mpl.colors.LogNorm())
fig23.colorbar(h[3], ax=ax2)

h = ax3.hist2d(Grid_x1, Grid_y1, bins=C3max - C3min, range=[(C3min, C3max), (R7min, R7max)], norm=mpl.colors.LogNorm())
fig23.colorbar(h[3], ax=ax3)

h = ax4.hist2d(Grid_x1, Grid_y1, bins=C4max - C4min, range=[(C4min, C4max), (R7min, R7max)], norm=mpl.colors.LogNorm())
fig23.colorbar(h[3], ax=ax4)

fig23.tight_layout()
plt.show()

fig24, (ax5, ax6, ax7, ax8) = plt.subplots(ncols=4, figsize=(20, 4))

h = ax5.hist2d(Grid_x1, Grid_y1, bins=C5max - C5min, range=[(C5min, C5max), (R7min, R7max)], norm=mpl.colors.LogNorm())
fig24.colorbar(h[3], ax=ax5)

h = ax6.hist2d(Grid_x1, Grid_y1, bins=C6max - C6min, range=[(C6min, C6max), (R7min, R7max)], norm=mpl.colors.LogNorm())
fig24.colorbar(h[3], ax=ax6)

h = ax7.hist2d(Grid_x1, Grid_y1, bins=C7max - C7min, range=[(C7min, C7max), (R7min, R7max)], norm=mpl.colors.LogNorm())
fig24.colorbar(h[3], ax=ax7)

h = ax8.hist2d(Grid_x1, Grid_y1, bins=C8max - C8min, range=[(C8min, C8max), (R7min, R7max)], norm=mpl.colors.LogNorm())
fig24.colorbar(h[3], ax=ax8)
fig24.tight_layout()
plt.show()

# Sort arrays into row 7 array


# Initialize trash arrays
row7_x1trash = []
row7_y1trash = []
row7_t1trash = []
row7_a1trash = []
row7_A1trash = []

# Sort out points NOT in row, add to trash
for i in range(len(Grid_x1) - 1):
    if R7min > y1[i]:
        row7_x1trash.append(i)
        row7_y1trash.append(i)
        row7_t1trash.append(i)
        row7_a1trash.append(i)
        row7_A1trash.append(i)
    elif R7max < y1[i]:
        row7_x1trash.append(i)
        row7_y1trash.append(i)
        row7_t1trash.append(i)
        row7_a1trash.append(i)
        row7_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
row7_x1 = np.delete(Grid_x1, row7_x1trash)
row7_y1 = np.delete(Grid_y1, row7_y1trash)
row7_t1 = np.delete(Grid_t1, row7_t1trash)
row7_a1 = np.delete(Grid_a1, row7_a1trash)
row7_A1 = np.delete(Grid_A1, row7_A1trash)

# Sanity Check
print(len(Grid_x1), len(row7_x1trash), len(row7_x1))

# Sort Row 7 array into points


# Initialize trash arrays
C1R7_x1trash = []
C1R7_y1trash = []
C1R7_t1trash = []
C1R7_a1trash = []
C1R7_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row7_x1) - 1):
    if C1min > row7_x1[i]:
        C1R7_x1trash.append(i)
        C1R7_y1trash.append(i)
        C1R7_t1trash.append(i)
        C1R7_a1trash.append(i)
        C1R7_A1trash.append(i)
    elif C1max < row7_x1[i]:
        C1R7_x1trash.append(i)
        C1R7_y1trash.append(i)
        C1R7_t1trash.append(i)
        C1R7_a1trash.append(i)
        C1R7_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C1R7_x1 = np.delete(row7_x1, C1R7_x1trash)
C1R7_y1 = np.delete(row7_y1, C1R7_y1trash)
C1R7_t1 = np.delete(row7_t1, C1R7_t1trash)
C1R7_a1 = np.delete(row7_a1, C1R7_a1trash)
C1R7_A1 = np.delete(row7_A1, C1R7_A1trash)

# Sanity Check
print(len(row7_x1), len(C1R7_x1trash))
print('Point 1 Rate (Hz):', len(C1R7_x1))

# save rate in case it is needed later
rate_C1R7 = len(C1R7_x1)

# Initialize trash arrays
C2R7_x1trash = []
C2R7_y1trash = []
C2R7_t1trash = []
C2R7_a1trash = []
C2R7_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row7_x1) - 1):
    if C2min > row7_x1[i]:
        C2R7_x1trash.append(i)
        C2R7_y1trash.append(i)
        C2R7_t1trash.append(i)
        C2R7_a1trash.append(i)
        C2R7_A1trash.append(i)
    elif C2max < row7_x1[i]:
        C2R7_x1trash.append(i)
        C2R7_y1trash.append(i)
        C2R7_t1trash.append(i)
        C2R7_a1trash.append(i)
        C2R7_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C2R7_x1 = np.delete(row7_x1, C2R7_x1trash)
C2R7_y1 = np.delete(row7_y1, C2R7_y1trash)
C2R7_t1 = np.delete(row7_t1, C2R7_t1trash)
C2R7_a1 = np.delete(row7_a1, C2R7_a1trash)
C2R7_A1 = np.delete(row7_A1, C2R7_A1trash)

# Sanity Check
print(len(row7_x1), len(C2R7_x1trash))
print('Point 2 Rate (Hz):', len(C2R7_x1))

# save rate in case it is needed later
rate_C2R7 = len(C2R7_x1)

# Initialize trash arrays
C3R7_x1trash = []
C3R7_y1trash = []
C3R7_t1trash = []
C3R7_a1trash = []
C3R7_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row7_x1) - 1):
    if C3min > row7_x1[i]:
        C3R7_x1trash.append(i)
        C3R7_y1trash.append(i)
        C3R7_t1trash.append(i)
        C3R7_a1trash.append(i)
        C3R7_A1trash.append(i)
    elif C3max < row7_x1[i]:
        C3R7_x1trash.append(i)
        C3R7_y1trash.append(i)
        C3R7_t1trash.append(i)
        C3R7_a1trash.append(i)
        C3R7_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C3R7_x1 = np.delete(row7_x1, C3R7_x1trash)
C3R7_y1 = np.delete(row7_y1, C3R7_y1trash)
C3R7_t1 = np.delete(row7_t1, C3R7_t1trash)
C3R7_a1 = np.delete(row7_a1, C3R7_a1trash)
C3R7_A1 = np.delete(row7_A1, C3R7_A1trash)

# Sanity Check
print(len(row7_x1), len(C3R7_x1trash))
print('Point 3 Rate (Hz):', len(C3R7_x1))

# save rate in case it is needed later
rate_C3R7 = len(C3R7_x1)

# Initialize trash arrays
C4R7_x1trash = []
C4R7_y1trash = []
C4R7_t1trash = []
C4R7_a1trash = []
C4R7_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row7_x1) - 1):
    if C4min > row7_x1[i]:
        C4R7_x1trash.append(i)
        C4R7_y1trash.append(i)
        C4R7_t1trash.append(i)
        C4R7_a1trash.append(i)
        C4R7_A1trash.append(i)
    elif C4max < row7_x1[i]:
        C4R7_x1trash.append(i)
        C4R7_y1trash.append(i)
        C4R7_t1trash.append(i)
        C4R7_a1trash.append(i)
        C4R7_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C4R7_x1 = np.delete(row7_x1, C4R7_x1trash)
C4R7_y1 = np.delete(row7_y1, C4R7_y1trash)
C4R7_t1 = np.delete(row7_t1, C4R7_t1trash)
C4R7_a1 = np.delete(row7_a1, C4R7_a1trash)
C4R7_A1 = np.delete(row7_A1, C4R7_A1trash)

# Sanity Check
print(len(row7_x1), len(C4R7_x1trash))
print('Point 4 Rate (Hz):', len(C4R7_x1))

# save rate in case it is needed later
rate_C4R7 = len(C4R7_x1)

# Initialize trash arrays
C5R7_x1trash = []
C5R7_y1trash = []
C5R7_t1trash = []
C5R7_a1trash = []
C5R7_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row7_x1) - 1):
    if C5min > row7_x1[i]:
        C5R7_x1trash.append(i)
        C5R7_y1trash.append(i)
        C5R7_t1trash.append(i)
        C5R7_a1trash.append(i)
        C5R7_A1trash.append(i)
    elif C5max < row7_x1[i]:
        C5R7_x1trash.append(i)
        C5R7_y1trash.append(i)
        C5R7_t1trash.append(i)
        C5R7_a1trash.append(i)
        C5R7_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C5R7_x1 = np.delete(row7_x1, C5R7_x1trash)
C5R7_y1 = np.delete(row7_y1, C5R7_y1trash)
C5R7_t1 = np.delete(row7_t1, C5R7_t1trash)
C5R7_a1 = np.delete(row7_a1, C5R7_a1trash)
C5R7_A1 = np.delete(row7_A1, C5R7_A1trash)

# Sanity Check
print(len(row7_x1), len(C5R7_x1trash))
print('Point 5 Rate (Hz):', len(C5R7_x1))

# save rate in case it is needed later
rate_C5R7 = len(C5R7_x1)

# Initialize trash arrays
C6R7_x1trash = []
C6R7_y1trash = []
C6R7_t1trash = []
C6R7_a1trash = []
C6R7_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row7_x1) - 1):
    if C6min > row7_x1[i]:
        C6R7_x1trash.append(i)
        C6R7_y1trash.append(i)
        C6R7_t1trash.append(i)
        C6R7_a1trash.append(i)
        C6R7_A1trash.append(i)
    elif C6max < row7_x1[i]:
        C6R7_x1trash.append(i)
        C6R7_y1trash.append(i)
        C6R7_t1trash.append(i)
        C6R7_a1trash.append(i)
        C6R7_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C6R7_x1 = np.delete(row7_x1, C6R7_x1trash)
C6R7_y1 = np.delete(row7_y1, C6R7_y1trash)
C6R7_t1 = np.delete(row7_t1, C6R7_t1trash)
C6R7_a1 = np.delete(row7_a1, C6R7_a1trash)
C6R7_A1 = np.delete(row7_A1, C6R7_A1trash)

# Sanity Check
print(len(row7_x1), len(C6R7_x1trash))
print('Point 6 Rate (Hz):', len(C6R7_x1))

# save rate in case it is needed later
rate_C6R7 = len(C6R7_x1)

# Initialize trash arrays
C7R7_x1trash = []
C7R7_y1trash = []
C7R7_t1trash = []
C7R7_a1trash = []
C7R7_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row7_x1) - 1):
    if C7min > row7_x1[i]:
        C7R7_x1trash.append(i)
        C7R7_y1trash.append(i)
        C7R7_t1trash.append(i)
        C7R7_a1trash.append(i)
        C7R7_A1trash.append(i)
    elif C7max < row7_x1[i]:
        C7R7_x1trash.append(i)
        C7R7_y1trash.append(i)
        C7R7_t1trash.append(i)
        C7R7_a1trash.append(i)
        C7R7_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C7R7_x1 = np.delete(row7_x1, C7R7_x1trash)
C7R7_y1 = np.delete(row7_y1, C7R7_y1trash)
C7R7_t1 = np.delete(row7_t1, C7R7_t1trash)
C7R7_a1 = np.delete(row7_a1, C7R7_a1trash)
C7R7_A1 = np.delete(row7_A1, C7R7_A1trash)

# Sanity Check
print(len(row7_x1), len(C7R7_x1trash))
print('Point 7 Rate (Hz):', len(C7R7_x1))

# save rate in case it is needed later
rate_C7R7 = len(C7R7_x1)

# Initialize trash arrays
C8R7_x1trash = []
C8R7_y1trash = []
C8R7_t1trash = []
C8R7_a1trash = []
C8R7_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row7_x1) - 1):
    if C8min > row7_x1[i]:
        C8R7_x1trash.append(i)
        C8R7_y1trash.append(i)
        C8R7_t1trash.append(i)
        C8R7_a1trash.append(i)
        C8R7_A1trash.append(i)
    elif C8max < row7_x1[i]:
        C8R7_x1trash.append(i)
        C8R7_y1trash.append(i)
        C8R7_t1trash.append(i)
        C8R7_a1trash.append(i)
        C8R7_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C8R7_x1 = np.delete(row7_x1, C8R7_x1trash)
C8R7_y1 = np.delete(row7_y1, C8R7_y1trash)
C8R7_t1 = np.delete(row7_t1, C8R7_t1trash)
C8R7_a1 = np.delete(row7_a1, C8R7_a1trash)
C8R7_A1 = np.delete(row7_A1, C8R7_A1trash)

# Sanity Check
print(len(row7_x1), len(C8R7_x1trash))
print('Point 8 Rate (Hz):', len(C8R7_x1))

# save rate in case it is needed later
rate_C8R7 = len(C8R7_x1)

# Display Row 8

fig25, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 4))

h = ax1.hist2d(Grid_x1, Grid_y1, bins=C1max - C1min, range=[(C1min, C1max), (R8min, R8max)], norm=mpl.colors.LogNorm())
fig25.colorbar(h[3], ax=ax1)

h = ax2.hist2d(Grid_x1, Grid_y1, bins=C2max - C2min, range=[(C2min, C2max), (R8min, R8max)], norm=mpl.colors.LogNorm())
fig25.colorbar(h[3], ax=ax2)

h = ax3.hist2d(Grid_x1, Grid_y1, bins=C3max - C3min, range=[(C3min, C3max), (R8min, R8max)], norm=mpl.colors.LogNorm())
fig25.colorbar(h[3], ax=ax3)

h = ax4.hist2d(Grid_x1, Grid_y1, bins=C4max - C4min, range=[(C4min, C4max), (R8min, R8max)], norm=mpl.colors.LogNorm())
fig25.colorbar(h[3], ax=ax4)

fig25.tight_layout()
plt.show()

fig26, (ax5, ax6, ax7, ax8) = plt.subplots(ncols=4, figsize=(20, 4))

h = ax5.hist2d(Grid_x1, Grid_y1, bins=C5max - C5min, range=[(C5min, C5max), (R8min, R8max)], norm=mpl.colors.LogNorm())
fig26.colorbar(h[3], ax=ax5)

h = ax6.hist2d(Grid_x1, Grid_y1, bins=C6max - C6min, range=[(C6min, C6max), (R8min, R8max)], norm=mpl.colors.LogNorm())
fig26.colorbar(h[3], ax=ax6)

h = ax7.hist2d(Grid_x1, Grid_y1, bins=C7max - C7min, range=[(C7min, C7max), (R8min, R8max)], norm=mpl.colors.LogNorm())
fig26.colorbar(h[3], ax=ax7)

h = ax8.hist2d(Grid_x1, Grid_y1, bins=C8max - C8min, range=[(C8min, C8max), (R8min, R8max)], norm=mpl.colors.LogNorm())
fig26.colorbar(h[3], ax=ax8)

fig26.tight_layout()
plt.show()

# Sort arrays into row 8 array


# Initialize trash arrays
row8_x1trash = []
row8_y1trash = []
row8_t1trash = []
row8_a1trash = []
row8_A1trash = []

# Sort out points NOT in row, add to trash
for i in range(len(Grid_x1) - 1):
    if R8min > y1[i]:
        row8_x1trash.append(i)
        row8_y1trash.append(i)
        row8_t1trash.append(i)
        row8_a1trash.append(i)
        row8_A1trash.append(i)
    elif R8max < y1[i]:
        row8_x1trash.append(i)
        row8_y1trash.append(i)
        row8_t1trash.append(i)
        row8_a1trash.append(i)
        row8_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
row8_x1 = np.delete(Grid_x1, row8_x1trash)
row8_y1 = np.delete(Grid_y1, row8_y1trash)
row8_t1 = np.delete(Grid_t1, row8_t1trash)
row8_a1 = np.delete(Grid_a1, row8_a1trash)
row8_A1 = np.delete(Grid_A1, row8_A1trash)

# Sanity Check
print(len(Grid_x1), len(row8_x1trash), len(row8_x1))

# Sort Row 8 array into points


# Initialize trash arrays
C1R8_x1trash = []
C1R8_y1trash = []
C1R8_t1trash = []
C1R8_a1trash = []
C1R8_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row8_x1) - 1):
    if C1min > row8_x1[i]:
        C1R8_x1trash.append(i)
        C1R8_y1trash.append(i)
        C1R8_t1trash.append(i)
        C1R8_a1trash.append(i)
        C1R8_A1trash.append(i)
    elif C1max < row8_x1[i]:
        C1R8_x1trash.append(i)
        C1R8_y1trash.append(i)
        C1R8_t1trash.append(i)
        C1R8_a1trash.append(i)
        C1R8_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C1R8_x1 = np.delete(row8_x1, C1R8_x1trash)
C1R8_y1 = np.delete(row8_y1, C1R8_y1trash)
C1R8_t1 = np.delete(row8_t1, C1R8_t1trash)
C1R8_a1 = np.delete(row8_a1, C1R8_a1trash)
C1R8_A1 = np.delete(row8_A1, C1R8_A1trash)

# Sanity Check
print(len(row8_x1), len(C1R8_x1trash))
print('Point 1 Rate (Hz):', len(C1R8_x1))

# save rate in case it is needed later
rate_C1R8 = len(C1R8_x1)

# Initialize trash arrays
C2R8_x1trash = []
C2R8_y1trash = []
C2R8_t1trash = []
C2R8_a1trash = []
C2R8_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row8_x1) - 1):
    if C2min > row8_x1[i]:
        C2R8_x1trash.append(i)
        C2R8_y1trash.append(i)
        C2R8_t1trash.append(i)
        C2R8_a1trash.append(i)
        C2R8_A1trash.append(i)
    elif C2max < row8_x1[i]:
        C2R8_x1trash.append(i)
        C2R8_y1trash.append(i)
        C2R8_t1trash.append(i)
        C2R8_a1trash.append(i)
        C2R8_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C2R8_x1 = np.delete(row8_x1, C2R8_x1trash)
C2R8_y1 = np.delete(row8_y1, C2R8_y1trash)
C2R8_t1 = np.delete(row8_t1, C2R8_t1trash)
C2R8_a1 = np.delete(row8_a1, C2R8_a1trash)
C2R8_A1 = np.delete(row8_A1, C2R8_A1trash)

# Sanity Check
print(len(row8_x1), len(C2R8_x1trash))
print('Point 2 Rate (Hz):', len(C2R8_x1))

# save rate in case it is needed later
rate_C2R8 = len(C2R8_x1)

# Initialize trash arrays
C3R8_x1trash = []
C3R8_y1trash = []
C3R8_t1trash = []
C3R8_a1trash = []
C3R8_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row8_x1) - 1):
    if C3min > row8_x1[i]:
        C3R8_x1trash.append(i)
        C3R8_y1trash.append(i)
        C3R8_t1trash.append(i)
        C3R8_a1trash.append(i)
        C3R8_A1trash.append(i)
    elif C3max < row8_x1[i]:
        C3R8_x1trash.append(i)
        C3R8_y1trash.append(i)
        C3R8_t1trash.append(i)
        C3R8_a1trash.append(i)
        C3R8_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C3R8_x1 = np.delete(row8_x1, C3R8_x1trash)
C3R8_y1 = np.delete(row8_y1, C3R8_y1trash)
C3R8_t1 = np.delete(row8_t1, C3R8_t1trash)
C3R8_a1 = np.delete(row8_a1, C3R8_a1trash)
C3R8_A1 = np.delete(row8_A1, C3R8_A1trash)

# Sanity Check
print(len(row8_x1), len(C3R8_x1trash))
print('Point 3 Rate (Hz):', len(C3R8_x1))

# save rate in case it is needed later
rate_C3R8 = len(C3R8_x1)

# Initialize trash arrays
C4R8_x1trash = []
C4R8_y1trash = []
C4R8_t1trash = []
C4R8_a1trash = []
C4R8_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row8_x1) - 1):
    if C4min > row8_x1[i]:
        C4R8_x1trash.append(i)
        C4R8_y1trash.append(i)
        C4R8_t1trash.append(i)
        C4R8_a1trash.append(i)
        C4R8_A1trash.append(i)
    elif C4max < row8_x1[i]:
        C4R8_x1trash.append(i)
        C4R8_y1trash.append(i)
        C4R8_t1trash.append(i)
        C4R8_a1trash.append(i)
        C4R8_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C4R8_x1 = np.delete(row8_x1, C4R8_x1trash)
C4R8_y1 = np.delete(row8_y1, C4R8_y1trash)
C4R8_t1 = np.delete(row8_t1, C4R8_t1trash)
C4R8_a1 = np.delete(row8_a1, C4R8_a1trash)
C4R8_A1 = np.delete(row8_A1, C4R8_A1trash)

# Sanity Check
print(len(row8_x1), len(C4R8_x1trash))
print('Point 4 Rate (Hz):', len(C4R8_x1))

# save rate in case it is needed later
rate_C4R8 = len(C4R8_x1)

# Initialize trash arrays
C5R8_x1trash = []
C5R8_y1trash = []
C5R8_t1trash = []
C5R8_a1trash = []
C5R8_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row8_x1) - 1):
    if C5min > row8_x1[i]:
        C5R8_x1trash.append(i)
        C5R8_y1trash.append(i)
        C5R8_t1trash.append(i)
        C5R8_a1trash.append(i)
        C5R8_A1trash.append(i)
    elif C5max < row8_x1[i]:
        C5R8_x1trash.append(i)
        C5R8_y1trash.append(i)
        C5R8_t1trash.append(i)
        C5R8_a1trash.append(i)
        C5R8_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C5R8_x1 = np.delete(row8_x1, C5R8_x1trash)
C5R8_y1 = np.delete(row8_y1, C5R8_y1trash)
C5R8_t1 = np.delete(row8_t1, C5R8_t1trash)
C5R8_a1 = np.delete(row8_a1, C5R8_a1trash)
C5R8_A1 = np.delete(row8_A1, C5R8_A1trash)

# Sanity Check
print(len(row8_x1), len(C5R8_x1trash))
print('Point 5 Rate (Hz):', len(C5R8_x1))

# save rate in case it is needed later
rate_C5R8 = len(C5R8_x1)

# Initialize trash arrays
C6R8_x1trash = []
C6R8_y1trash = []
C6R8_t1trash = []
C6R8_a1trash = []
C6R8_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row8_x1) - 1):
    if C6min > row8_x1[i]:
        C6R8_x1trash.append(i)
        C6R8_y1trash.append(i)
        C6R8_t1trash.append(i)
        C6R8_a1trash.append(i)
        C6R8_A1trash.append(i)
    elif C6max < row8_x1[i]:
        C6R8_x1trash.append(i)
        C6R8_y1trash.append(i)
        C6R8_t1trash.append(i)
        C6R8_a1trash.append(i)
        C6R8_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C6R8_x1 = np.delete(row8_x1, C6R8_x1trash)
C6R8_y1 = np.delete(row8_y1, C6R8_y1trash)
C6R8_t1 = np.delete(row8_t1, C6R8_t1trash)
C6R8_a1 = np.delete(row8_a1, C6R8_a1trash)
C6R8_A1 = np.delete(row8_A1, C6R8_A1trash)

# Sanity Check
print(len(row8_x1), len(C6R8_x1trash))
print('Point 6 Rate (Hz):', len(C6R8_x1))

# save rate in case it is needed later
rate_C6R8 = len(C6R8_x1)

# Initialize trash arrays
C7R8_x1trash = []
C7R8_y1trash = []
C7R8_t1trash = []
C7R8_a1trash = []
C7R8_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row8_x1) - 1):
    if C7min > row8_x1[i]:
        C7R8_x1trash.append(i)
        C7R8_y1trash.append(i)
        C7R8_t1trash.append(i)
        C7R8_a1trash.append(i)
        C7R8_A1trash.append(i)
    elif C7max < row8_x1[i]:
        C7R8_x1trash.append(i)
        C7R8_y1trash.append(i)
        C7R8_t1trash.append(i)
        C7R8_a1trash.append(i)
        C7R8_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C7R8_x1 = np.delete(row8_x1, C7R8_x1trash)
C7R8_y1 = np.delete(row8_y1, C7R8_y1trash)
C7R8_t1 = np.delete(row8_t1, C7R8_t1trash)
C7R8_a1 = np.delete(row8_a1, C7R8_a1trash)
C7R8_A1 = np.delete(row8_A1, C7R8_A1trash)

# Sanity Check
print(len(row8_x1), len(C7R8_x1trash))
print('Point 7 Rate (Hz):', len(C7R8_x1))

# save rate in case it is needed later
rate_C7R8 = len(C7R8_x1)

# Initialize trash arrays
C8R8_x1trash = []
C8R8_y1trash = []
C8R8_t1trash = []
C8R8_a1trash = []
C8R8_A1trash = []

# Sort out points NOT in spot, add to trash
for i in range(len(row8_x1) - 1):
    if C8min > row8_x1[i]:
        C8R8_x1trash.append(i)
        C8R8_y1trash.append(i)
        C8R8_t1trash.append(i)
        C8R8_a1trash.append(i)
        C8R8_A1trash.append(i)
    elif C8max < row8_x1[i]:
        C8R8_x1trash.append(i)
        C8R8_y1trash.append(i)
        C8R8_t1trash.append(i)
        C8R8_a1trash.append(i)
        C8R8_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
C8R8_x1 = np.delete(row8_x1, C8R8_x1trash)
C8R8_y1 = np.delete(row8_y1, C8R8_y1trash)
C8R8_t1 = np.delete(row8_t1, C8R8_t1trash)
C8R8_a1 = np.delete(row8_a1, C8R8_a1trash)
C8R8_A1 = np.delete(row8_A1, C8R8_A1trash)

# Sanity Check
print(len(row8_x1), len(C8R8_x1trash))
print('Point 8 Rate (Hz):', len(C8R8_x1))

# save rate in case it is needed later
rate_C8R8 = len(C8R8_x1)

RateCol1 = [rate_C1R1, rate_C1R2, rate_C1R3, rate_C1R4, rate_C1R5, rate_C1R6, rate_C1R7, rate_C1R8]
AvgCol1 = sum(RateCol1) / 8
RateCol2 = [rate_C2R1, rate_C2R2, rate_C2R3, rate_C2R4, rate_C2R5, rate_C2R6, rate_C2R7, rate_C2R8]
AvgCol2 = sum(RateCol2) / 8
RateCol3 = [rate_C3R1, rate_C3R2, rate_C3R3, rate_C3R4, rate_C3R5, rate_C3R6, rate_C3R7, rate_C3R8]
AvgCol3 = sum(RateCol3) / 8
RateCol4 = [rate_C4R1, rate_C4R2, rate_C4R3, rate_C4R4, rate_C4R5, rate_C4R6, rate_C4R7, rate_C4R8]
AvgCol4 = sum(RateCol4) / 8
RateCol5 = [rate_C5R1, rate_C5R2, rate_C5R3, rate_C5R4, rate_C5R5, rate_C5R6, rate_C5R7, rate_C5R8]
AvgCol5 = sum(RateCol5) / 8
RateCol6 = [rate_C6R1, rate_C6R2, rate_C6R3, rate_C6R4, rate_C6R5, rate_C6R6, rate_C6R7, rate_C6R8]
AvgCol6 = sum(RateCol6) / 8
RateCol7 = [rate_C7R1, rate_C7R2, rate_C7R3, rate_C7R4, rate_C7R5, rate_C7R6, rate_C7R7, rate_C7R8]
AvgCol7 = sum(RateCol7) / 8
RateCol8 = [rate_C8R1, rate_C8R2, rate_C8R3, rate_C8R4, rate_C8R5, rate_C8R6, rate_C8R7, rate_C8R8]
AvgCol8 = sum(RateCol8) / 8

AvgCol = [AvgCol1, AvgCol2, AvgCol3, AvgCol4, AvgCol5, AvgCol6, AvgCol7, AvgCol8]
print(AvgCol)

print(RateCol1, RateCol2, RateCol3, RateCol4, RateCol5, RateCol6, RateCol7, RateCol8)

RateRow1 = [rate_C1R1, rate_C2R1, rate_C3R1, rate_C4R1, rate_C5R1, rate_C6R1, rate_C6R1, rate_C8R1]
AvgRow1 = sum(RateRow1) / 8
RateRow2 = [rate_C1R2, rate_C2R2, rate_C3R2, rate_C4R2, rate_C5R2, rate_C6R2, rate_C6R2, rate_C8R2]
AvgRow2 = sum(RateRow2) / 8
RateRow3 = [rate_C1R3, rate_C2R3, rate_C3R3, rate_C4R3, rate_C5R3, rate_C6R3, rate_C6R3, rate_C8R3]
AvgRow3 = sum(RateRow3) / 8
RateRow4 = [rate_C1R4, rate_C2R4, rate_C3R4, rate_C4R4, rate_C5R4, rate_C6R4, rate_C6R4, rate_C8R4]
AvgRow4 = sum(RateRow4) / 8
RateRow5 = [rate_C1R5, rate_C2R5, rate_C3R5, rate_C4R5, rate_C5R5, rate_C6R5, rate_C6R5, rate_C8R5]
AvgRow5 = sum(RateRow5) / 8
RateRow6 = [rate_C1R6, rate_C2R6, rate_C3R6, rate_C4R6, rate_C5R6, rate_C6R6, rate_C6R6, rate_C8R6]
AvgRow6 = sum(RateRow6) / 8
RateRow7 = [rate_C1R7, rate_C2R7, rate_C3R7, rate_C4R7, rate_C5R7, rate_C6R7, rate_C6R7, rate_C8R7]
AvgRow7 = sum(RateRow7) / 8
RateRow8 = [rate_C1R8, rate_C2R8, rate_C3R8, rate_C4R8, rate_C5R8, rate_C6R8, rate_C6R8, rate_C8R8]
AvgRow8 = sum(RateRow8) / 8

AvgRow = [AvgRow1, AvgRow2, AvgRow3, AvgRow4, AvgRow5, AvgRow6, AvgRow7, AvgRow8]
print(AvgRow)

# define tmin, tmax
tMin0 = 0.9990e9
tMax0 = 0.999125e9

# Sort arrays to exclude points outside Grid

# Converted time array
t1_conv = []
t1_conv = t1 / 4096. * 25

# Initialize trash arrays
t1_0_Grid_x1trash = []
t1_0_Grid_y1trash = []
t1_0_Grid_t1trash = []
t1_0_Grid_a1trash = []
t1_0_Grid_A1trash = []

for i in range(len(x1) - 1):
    if t1_conv[i] < tMin0:
        t1_0_Grid_x1trash.append(i)
        t1_0_Grid_y1trash.append(i)
        t1_0_Grid_t1trash.append(i)
        t1_0_Grid_a1trash.append(i)
        t1_0_Grid_A1trash.append(i)
    elif t1_conv[i] > tMax0:
        t1_0_Grid_x1trash.append(i)
        t1_0_Grid_y1trash.append(i)
        t1_0_Grid_t1trash.append(i)
        t1_0_Grid_a1trash.append(i)
        t1_0_Grid_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
t1_0_Grid_x1 = np.delete(x1, t1_0_Grid_x1trash)
t1_0_Grid_y1 = np.delete(y1, t1_0_Grid_y1trash)
t1_0_Grid_t1 = np.delete(t1, t1_0_Grid_t1trash)
t1_0_Grid_a1 = np.delete(a1, t1_0_Grid_a1trash)
t1_0_Grid_A1 = np.delete(A1, t1_0_Grid_A1trash)

# Sanity Check
print(len(x1), len(t1_conv), len(t1_0_Grid_x1trash), len(t1_0_Grid_x1), ", tMin0: {}".format(tMin0),
      ", tMax0: {}".format(tMax0))

GridX1min = 60
GridX1max = 105
GridY1min = 155
GridY1max = 198

fig27, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9.5, 4))

h = ax0.hist2d(t1_0_Grid_x1, t1_0_Grid_y1, bins=GridX1max - GridX1min,
               range=[(GridX1min, GridX1max), (GridY1min, GridY1max)])
fig27.colorbar(h[3], ax=ax0)

h = ax1.hist2d(t1_0_Grid_x1, t1_0_Grid_y1, bins=GridX1max - GridX1min,
               range=[(GridX1min, GridX1max), (GridY1min, GridY1max)], norm=mpl.colors.LogNorm())
fig27.colorbar(h[3], ax=ax1)

fig27.tight_layout()
plt.show()

# define tmin, tmax
tStep = tMax0 - tMin0
tMin1 = tMax0
tMax1 = tMin1 + tStep

# Sort arrays to exclude points outside Grid

# #Converted time array
# t1_conv =[];
# t1_conv = t1/4096.*25;

# Initialize trash arrays
t1_1_Grid_x1trash = []
t1_1_Grid_y1trash = []
t1_1_Grid_t1trash = []
t1_1_Grid_a1trash = []
t1_1_Grid_A1trash = []

for i in range(len(x1) - 1):
    if t1_conv[i] < tMin1:
        t1_1_Grid_x1trash.append(i)
        t1_1_Grid_y1trash.append(i)
        t1_1_Grid_t1trash.append(i)
        t1_1_Grid_a1trash.append(i)
        t1_1_Grid_A1trash.append(i)
    elif t1_conv[i] > tMax1:
        t1_1_Grid_x1trash.append(i)
        t1_1_Grid_y1trash.append(i)
        t1_1_Grid_t1trash.append(i)
        t1_1_Grid_a1trash.append(i)
        t1_1_Grid_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
t1_1_Grid_x1 = np.delete(x1, t1_1_Grid_x1trash)
t1_1_Grid_y1 = np.delete(y1, t1_1_Grid_y1trash)
t1_1_Grid_t1 = np.delete(t1, t1_1_Grid_t1trash)
t1_1_Grid_a1 = np.delete(a1, t1_1_Grid_a1trash)
t1_1_Grid_A1 = np.delete(A1, t1_1_Grid_A1trash)

# Sanity Check
print(len(x1), len(t1_conv), len(t1_1_Grid_x1trash), len(t1_1_Grid_x1), ", tMin1: {}".format(tMin1),
      ", tMax1: {}".format(tMax1), ", tStep: {}".format(tStep))

GridX1min = 60
GridX1max = 105
GridY1min = 155
GridY1max = 198

fig28, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9.5, 4))

h = ax0.hist2d(t1_1_Grid_x1, t1_1_Grid_y1, bins=GridX1max - GridX1min,
               range=[(GridX1min, GridX1max), (GridY1min, GridY1max)])
fig28.colorbar(h[3], ax=ax0)

h = ax1.hist2d(t1_1_Grid_x1, t1_1_Grid_y1, bins=GridX1max - GridX1min,
               range=[(GridX1min, GridX1max), (GridY1min, GridY1max)], norm=mpl.colors.LogNorm())
fig28.colorbar(h[3], ax=ax1)

fig28.tight_layout()

# define tmin, tmax
tMin2 = tMax1
tMax2 = tMin2 + tStep

# Sort arrays to exclude points outside Grid

# #Converted time array
# t1_conv =[];
# t1_conv = t1/4096.*25;

# Initialize trash arrays
t1_2_Grid_x1trash = []
t1_2_Grid_y1trash = []
t1_2_Grid_t1trash = []
t1_2_Grid_a1trash = []
t1_2_Grid_A1trash = []

for i in range(len(x1) - 1):
    if t1_conv[i] < tMin1:
        t1_2_Grid_x1trash.append(i)
        t1_2_Grid_y1trash.append(i)
        t1_2_Grid_t1trash.append(i)
        t1_2_Grid_a1trash.append(i)
        t1_2_Grid_A1trash.append(i)
    elif t1_conv[i] > tMax1:
        t1_2_Grid_x1trash.append(i)
        t1_2_Grid_y1trash.append(i)
        t1_2_Grid_t1trash.append(i)
        t1_2_Grid_a1trash.append(i)
        t1_2_Grid_A1trash.append(i)

# Generate new arrays by deleting trash selections from next step up
t1_2_Grid_x1 = np.delete(x1, t1_2_Grid_x1trash)
t1_2_Grid_y1 = np.delete(y1, t1_2_Grid_y1trash)
t1_2_Grid_t1 = np.delete(t1, t1_2_Grid_t1trash)
t1_2_Grid_a1 = np.delete(a1, t1_2_Grid_a1trash)
t1_2_Grid_A1 = np.delete(A1, t1_2_Grid_A1trash)

# Sanity Check
print(len(x1), len(t1_conv), len(t1_2_Grid_x1trash), len(t1_2_Grid_x1), ", tMin2: {}".format(tMin2),
      ", tMax2: {}".format(tMax2))

GridX1min = 60
GridX1max = 105
GridY1min = 155
GridY1max = 198

fig29, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9.5, 4))

h = ax0.hist2d(t1_2_Grid_x1, t1_2_Grid_y1, bins=GridX1max - GridX1min,
               range=[(GridX1min, GridX1max), (GridY1min, GridY1max)])
fig29.colorbar(h[3], ax=ax0)

h = ax1.hist2d(t1_2_Grid_x1, t1_2_Grid_y1, bins=GridX1max - GridX1min,
               range=[(GridX1min, GridX1max), (GridY1min, GridY1max)], norm=mpl.colors.LogNorm())
fig29.colorbar(h[3], ax=ax1)

fig29.tight_layout()

fig30, ax0 = plt.subplots(ncols=1, figsize=(10, 4))
plt.hist(C1R1_t1 / 4096. * 25., bins=1000, color='r', ec='r', range=(.998e9, 1e9))
plt.title("TOA", fontsize=12)  # change the title
plt.xlabel('TOA, ns', fontsize=12)
plt.ylabel('counts', fontsize=12)
plt.show()

# fig, ax0 = plt.subplots(ncols=1, figsize=(10, 4))
# plt.hist(tt/4096.*25., bins = 100, color = 'r', ec = 'k')
# plt.title("TOA", fontsize = 12) # change the title
# plt.xlabel('TOA, ns',fontsize = 12)
# plt.ylabel('counts',fontsize = 12)
# plt.show()

fig31, ax0 = plt.subplots(ncols=1, figsize=(10, 4))
# plt.hist(t/4096.*25., bins = 10, range = (0.5E+9, 1.5E+9), color = 'r', ec = 'k')
plt.hist(C1R1_t1 / 4096. * 25., bins=1000, range=(-1.E+9, 1.E+9), color='r', ec='r')
plt.title("TOA", fontsize=12)  # change the title
plt.xlabel('TOA, ns', fontsize=12)
plt.ylabel('counts', fontsize=12)
plt.show()

fig32, ax0 = plt.subplots(ncols=1, figsize=(10, 4))
plt.hist(C4R3_t1 / 4096. * 25., bins=1000, color='r', ec='r', range=(.998e9, 1e9))
plt.title("TOA", fontsize=12)  # change the title
plt.xlabel('TOA, ns', fontsize=12)
plt.ylabel('counts', fontsize=12)
plt.show()

# fig, ax0 = plt.subplots(ncols=1, figsize=(10, 4))
# plt.hist(tt/4096.*25., bins = 100, color = 'r', ec = 'k')
# plt.title("TOA", fontsize = 12) # change the title
# plt.xlabel('TOA, ns',fontsize = 12)
# plt.ylabel('counts',fontsize = 12)
# plt.show()

fig33, ax0 = plt.subplots(ncols=1, figsize=(10, 4))
# plt.hist(t/4096.*25., bins = 10, range = (0.5E+9, 1.5E+9), color = 'r', ec = 'k')
plt.hist(C4R3_t1 / 4096. * 25., bins=1000, range=(-1.E+9, 1.E+9), color='r', ec='r')
plt.title("TOA", fontsize=12)  # change the title
plt.xlabel('TOA, ns', fontsize=12)
plt.ylabel('counts', fontsize=12)
plt.show()
