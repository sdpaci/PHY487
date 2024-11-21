import os
import sys
import pandas as pd
import numpy as np
import math

#print(np)
#print(os)
#print(sys.path)
#print(pd)

def find(name, path):  #find first match
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

file = find('pbtio3x10.xyz', 'C:\PHY487')
print(file)

#read file into a DataFrame and select val at [0][0]
# natoms = pd.read_table(file, header=None, delim_whitespace=True, nrows=1)[0][0]
#
# print(natoms)
# df = pd.read_table(file)
# #now we have natoms now we read whole file adn remove space and create step
#
# df = pd.read_table(file)
# print(df)
# print(len(df))
# df= pd.DataFrame()
# print(df)
# drop_list=[0,1]
# df = pd.read_table(file, header=None, skiprows=drop_list, delim_whitespace=True)
# # assign steps, use index mod for step, mod290
# df = df.dropna()
# df.columns = ['ATOM', 'X', 'Y', 'Z']
# df['STEP'] = df.reset_index().index / natoms + 1
# df2 = pd.read_table(file, skip_blank_lines=True)
# print(df2)
# print(len(df2))
# df3= pd.read_table(file, skiprows = (2))
# print(df3)
# df4= pd.read_table(file, skiprows = (0,2))
# print(len(df3))
# print(df.index, df2.index, df3.index)
# count_row=df.shape[0]
# count_row2=df2.shape[0]
# count_row3=df3.shape[0]
# print(count_row)
# print(count_row2)
# print(count_row3)
# #count all empty rows
# nullDf = df.isnull()    #create null DataFRame with True & False data
# print(nullDf)
# print(nullDf.sum().sum())      #give count of total NaN in DataFrame
# #print((len(df)+(nullDf.sum())))    #NaNs plus of number elements should give total number of rows
# print(df.dropna(inplace=True))
# def read_ani(file_path, natoms=275):
#     natoms = pd.read_table(file_path, header=None, delim_whitespace=True, nrows=1)[0][0]
#     print("Atoms: {}".format(natoms))
#     #there is one blank row after natoms designation per time step
#     #we can drop ions at the end
#     df = pd.DataFrame()
#     #drop first and second row to get columns right, can drop nans later
#     drop_list=[0,1]
#     df = pd.read_table(file_path, header=None, skiprows=drop_list, delim_whitespace=True)
#     #assign steps, use index mod for step, mod290
#     df = df.dropna()
#     df.columns = ['ATOM', 'X', 'Y', 'Z']
#     df['STEP'] = df.reset_index().index/natoms+1
#     return df
# f = read_ani('C:\PHY487\pbtio3x10.xyz')
# print(f)

#read file into a DataFrame and select val at [0][0]
natoms = pd.read_table(file, header=None, delim_whitespace=True, nrows=1)[0][0]
df= pd.DataFrame()
drop_list=[0,1]
df = pd.read_table(file, header=None, skiprows=drop_list, delim_whitespace=True)
df = df.dropna()
df.columns = ['ATOM', 'X', 'Y', 'Z']
df['STEP'] = df.reset_index().index // natoms + 1
#print(df.index)
print("Number of Atoms: {}".format(natoms))
# b=df.index
s = set(df['STEP'])
print("Number of Time Steps {}".format(len(s)))


# print(df.loc[0, :].unique())
# K=df.loc[0, :].unique()
# print(df.loc[0]['ATOM':'Z'])
# G=df.loc[0]['ATOM':'Z']
# print(G)
# # print(G.loc[0, [1,3]].unique())
# # print(G.loc['Pb']['X'].unique())
# # print("Atom: {}".format(G.loc[1][1].unique()))
# print(natoms)
# # def d_euclidean(x, y, z):
#
# # Brute force Python 3 program
# # to calculate the maximum
# # absolute difference of an array.
#
#
# def calculateDiff(i, j, arr):
#     # Utility function to calculate
#     # the value of absolute difference
#     # for the pair (i, j).
#     print("abs(arr[i] - arr[j]) {}".format(abs(arr[i] - arr[j])))
#     print("abs(i - j) {}".format(abs(i - j)))
#     print("abs(arr[i] - arr[j]) + abs(i - j) {}".format(abs(arr[i] - arr[j]) + abs(i - j)))
#
#     return abs(arr[i] - arr[j])
#
#
# # Function to return maximum
# # absolute difference in
# # brute force.
# def maxDistance(arr, n, arrayOfDiffs):
#     # Variable for storing the
#     # maximum absolute distance
#     # throughout the traversal
#     # of loops.
#     result = 0
#
#     # Iterate through all pairs.
#     for i in range(0, n):
#         for j in range(i, n):
#
#             # If the absolute difference of
#             # current pair (i, j) is greater
#             # than the maximum difference
#             # calculated till now, update
#             # the value of result.
#             arrayOfDiffs.append(calculateDiff(i, j, arr))
#             if (calculateDiff(i, j, arr) > result):
#                 result = calculateDiff(i, j, arr)
#
#     return result, arrayOfDiffs
#
#
# # Driver program
# arr1 = [-70, -64, -6, -56, 64,
#         61, -57, 16, 48, -98]
# arr2 = [5, -77, -8, 55, 60,
#         43, -10, 49, 72, -37]
#take in
# n = len(arr)
# arrayOfDiffs = []
#pwrLst = []
OfromH2ODict = {}
OfromSlabDict = {}
timeDict = {}

#zippedList = []

# print("maxDistance(arr, n) {}".format(maxDistance(arr, n, arrayOfDiffs)[0]))
# print("arrayOfDiffs {}".format(maxDistance(arr, n, arrayOfDiffs)[1]))

# uniqueh20 = []
# h20Master = []
# for k in atomDict:
#     if "tags O and H" in k:
#         if atomDict[k][2][0] == H20_L:
#             h20Dict[k] = atomDict[k]
# for k in h20Dict:
#     for j in h20Dict:
#         if h20Dict[k][3][0] == h20Dict[j][3][0]:
#             uniqueh20[h20Dict[k][3][0]] = h20Dict[k]
# for k in uniqueh20:
#     if len(uniqueh20[k]) >= 2:
#             h20Master[h20Dict[k][3][0]] = uniqueh20[k]

#Boundry Conditions

bcX = [11.087434329,    0.0000000000,  0.000000]    #diff between part 1 in zipped list and ..
bcY = [0.0000000000,    11.087434329,   0.000000]
bcZ = [0.0000000000,    0.0000000000, 27.202644]

LENGTH = 27.202644
WIDTH = 11.087434329

H20_L = 1.00

def h2oSearch(atomdict):
    hoPairs = {}
    for k in atomdict:
        if (atomdict[k][5][0] == 'H') and (atomdict[k][6][0] == 'O'): #make this a slice somehow or import tags as element and look through them
            hoPairs[k] = atomdict[k]
    HOnums = []
    for k in hoPairs:
        HOnums.append(hoPairs[k][3])
    hoSet = set(HOnums)
    hopairsByAtom = {}
    for n in hoSet:
        hopairsByAtom[n] = []
    for n in hopairsByAtom:
        for k in hoPairs:
            if n == hoPairs[k][3]:
                hopairsByAtom[n].append(hoPairs[k])
    for k in hopairsByAtom:
        l = len(hopairsByAtom[k])
        for i in range(len(hopairsByAtom[k])):
            r = hopairsByAtom[k]
            t = hopairsByAtom[k][i]
            y = hopairsByAtom[k][i][2]
            g = hopairsByAtom[k][i][2][0]
            if hopairsByAtom[k][i][2][0] > H20_L:
                hopairsByAtom[k].pop(i)
    H2ODict = {}
    for k in hopairsByAtom:
        if len(hopairsByAtom[k]) >= 2:
            # Then it's an H2O
            H2ODict[k] = hopairsByAtom[k]
    OfromH2O = []
    for k in H2ODict:
        for i in range(len(H2ODict[k])):
            OfromH2O.append(H2ODict[k][i][4])
    # for k in hoPairs:
    #     if hoPairs[k][2][0] > H20_L:
    #         hoPairs.pop(k)
    # hopairsByAtom = {}
    # HOnums = []
    # dictList = []
    # for k in hoPairs:
    #     HOnums.append(hoPairs[k][3][0])
    # hoSet = set(HOnums)
    # # for n in hoSet:
    # #     hopairsByAtom[n] = []
    # hopairsByAtom = {dictList: [] for dictList in range(len(hoSet))}
    # for n in hoSet:
    #     for k in hoPairs:
    #         if n == hoPairs[k][3][0]:
    #             hopairsByAtom[n] = dictList.append(k)
    # H2ODict = {}
    # for k in hopairsByAtom:
    #     if len(hopairsByAtom[k]) >= 2:
    #         H2ODict[k] = True
    return H2ODict, OfromH2O

def boundary_conditions(vectorList):
    #The slab is six sided so there are six mirror particles... but there are also parst in the corners (4) plus 6 sides = 10 image
    #diff between point and point being sub. and all it's mirror parts
    #veclistP = [x,y,z]
    for i in range(len(vectorList)):
        if i == 2:
            if vectorList[i] > LENGTH/2:
                vectorList[i] = (vectorList[i] - LENGTH)
            elif vectorList[i] < -(LENGTH/2):
                vectorList[i] = (vectorList[i] + LENGTH)
        else:
            if vectorList[i] > WIDTH/2:
                vectorList[i] = (vectorList[i] - WIDTH)
            elif vectorList[i] < -(WIDTH/2):
                vectorList[i] = (vectorList[i] + WIDTH)
    return vectorList


# func to take pos. and cal. euclidean dist.
# should calc. the distance. btwn. every particle for  given time step
# grab each three x,y,z from each col of times step and calc dist between each col(aka coords)
def e_dist(zippedList, i, j, iTag, jTag):
    pwrLst = []
    vecList = []
    vecListP = []
    vecListN = []
    vecListM =[]
    #e_dist = sqrt(pwr(x2-x1)^2+...)
    #math.sqrt(math.pow(x2-x1,2)+...)
    for (a, b) in zippedList:  # (x1,x2),(y1,y2),(z1,z2)
        vecListP.append(a-b)
    for (a, b) in zippedList:  # (x1,x2),(y1,y2),(z1,z2)
        vecListN.append(b-a)
    vecList.append(boundary_conditions(vecListP))
    vecList.append(boundary_conditions(vecListN))
    for (a,b) in zippedList:
           pwrLst.append(math.pow(b-a, 2))
    vecListM.append(math.sqrt(sum(pwrLst)))
    vecList.append(vecListM)
    vecList.append(i)
    vecList.append(j)
    vecList.append(iTag)
    vecList.append(jTag)
    return vecList

# for every col(of x,y,z) in DataFrame() first time step, use double nested for loop to calc. dists & store in dict
def coord_zipper(nStart, nStop):
    eDistDict = {}
    for i in range(nStart, nStop): #(0 to 274)
        for j in range(i, nStop): #(i to 274)
            # zip xyz at i with subsequent xyz at j
            rowI = df.loc[i, :].unique()
            valsI = rowI[1:4]
            iTag = rowI[0]
            rowJ = df.loc[j, :].unique()
            valsJ = rowJ[1:4]
            jTag = rowJ[0]
            # take comparing cols slice xyz's, zip them into a list
            zippedList = list(zip(valsI, valsJ))
            eDistDict["Distance between atoms {} and {}; tags {} and {}, List A: Vector from {} -> {} and List B: Vector from {} -> {}".format(i, j, iTag, jTag, i, j, j, i)] = e_dist(zippedList, i, j, iTag, jTag) #boundary_conditions(e_dist(zippedList))
    print(eDistDict)
    print(len(eDistDict))
    h20Return = h2oSearch(eDistDict)
    H2ODict = h20Return[0]
    OfromH2O = h20Return[1]
    OfromSlab = []
    for i in range(nStart, nStop):
        if ((i) != (OfromH2O[i])) and ((df.loc[i, 0].unique()) == 'O'):
            OfromSlab.append(i)
    return eDistDict, OfromH2O, OfromSlab

#for every time step run 275 times
for i in s:
    #from index 276 to 550; from natoms*i-natoms to natoms*i  3:552->826 4:828
    #((natoms*i)-(natoms)+(i-1)) to n+(i-1)
    n = natoms*i
    nStart = (((n)-(natoms))+(i-1))
    nStop = (n+(i-1))
    coordReturn = coord_zipper(nStart, nStop)
    distDict = coordReturn[0]
    OfromH2O = coordReturn[1]
    OfromSlab = coordReturn[2]
    timeDict["Distance dict for time step {}".format(i)] = distDict
    OfromH2ODict["O from H2O list for time step {}".format(i)] = OfromH2O
    OfromSlabDict["O from Slab list for time step {}".format(i)] = OfromSlab




# for i in dist:
#     dist[i] = min_dist(i)


