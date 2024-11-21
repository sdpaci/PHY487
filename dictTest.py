import os
import sys
import pandas as pd
import numpy as np
import math

vecTo1 = [21, 14, 5]
vecFrom1 = [6, 41, 18]
mag1 = [23]
i1 = [273]

vecList1 = []
vecList1.append(vecTo1)
vecList1.append(vecFrom1)
vecList1.append(mag1)
vecList1.append(i1)

vecTo2 = [21, 14, 5]
vecFrom2 = [6, 41, 18]
mag2 = [23]
i2 = [273]

vecList2 = []
vecList2.append(vecTo2)
vecList2.append(vecFrom2)
vecList2.append(mag2)
vecList2.append(i2)

dict = {}
newList = []
vecList = []
vecList.append(vecList1)
vecList.append(vecList2)

for i in range(2):
    dict["Element {}".format(i)] = vecList[i]

print(dict)

for k in dict:
    newList.append(dict[k][0][2])

print(newList)

# def set_Linage(nStart, nStop, Oindex, Hindex):
#     for k in range(nStart, nStop):
#         if (k != Oindex) and (df.loc[k, 'ATOM'] == 'O') and (pd.isnull(df.loc[k, 'Molecule']) == True):
#             O2index = k
#             # Olist2 = []
#             # Olist2.append(df.loc[k, 'X'])
#             # Olist2.append(df.loc[k, 'Y'])
#             # Olist2.append(df.loc[k, 'Z'])
#             if dist(Hindex, O2index) <= HO_L:
#                 # we can say O(i) and O(k) belong to H2O or H3O
#                 # O(i) and O(k) are O from H2O
#                 df.at[Oindex, 'Molecule'] = "O from H2O"
#                 df.at[O2index, 'Molecule'] = "O from H2O"
#                 return


# def df_search():
#     for i in range(nStart, nStop):  # (0 to 274)
#         if (df.loc[i, 'ATOM'] == 'O') and (pd.isnull(df.loc[i, 'Molecule']) == True):
#             Oindex = i
#             # Olist1 = []
#             # Olist1.append(df.loc[i, 'X'])
#             # Olist1.append(df.loc[i, 'Y'])
#             # Olist1.append(df.loc[i, 'Z'])
#             for j in range(nStart, nStop):
#                 if df.loc[j, 'ATOM'] == 'H':
#                     Hindex = j
#                     # Hlist = []
#                     # Hlist.append(df.loc[j, 'X'])
#                     # Hlist.append(df.loc[j, 'Y'])
#                     # Hlist.append(df.loc[j, 'Z'])
#                     # calculate the distance between the O and H
#                     if dist(Oindex, Hindex) <= HO_L:
#                         # we can say O(i) belongs to Hydroxyl or H2O or H30
#                         set_Linage(nStart, nStop, Oindex, Hindex)
#                         break
#     else:
#         for i in range(nStart, nStop):  # (0 to 274)
#             if (df.loc[i, 'ATOM'] == 'O') and (pd.isnull(df.loc[i, 'Molecule']) == True):
#                 df[i, 'Molecule'] = "O from Slab"
#                 # for k in range(nStart, nStop):
#                 #     if (k != Oindex) and (df.loc[k, 0] == 'O') and (pd.isnull(df.loc[k, 5]) == True):
#                 #         O2index = k
#                 #         Olist2 = []
#                 #         Olist2.append(df.loc[k, 1])
#                 #         Olist2.append(df.loc[k, 2])
#                 #         Olist2.append(df.loc[k, 3])
#                 #         if dist(Hlist, Olist2) <= HO_L:
#                 #             #we can say O(i) and O(k) belong to H2O or H3O
#                 #             #O(i) and O(k) are O from H2O
#                 #             df.set_value(i, 'Molecule', "O from H2O")
#                 #             df.set_value(k, 'Molecule', "O from H2O")
#                 #             break

def find_molecule_user_choice(ts_start, ts_stop, x, y, z):
    """For coordinates of atom in the time step this function checks for other atoms which may be in it's proximity to
        determine to which molecule it may belong """
    # get atom type at current index
    atom_index_and_type = get_atom_xyz(ts_start, ts_stop, x, y, z)
    e = atom_index_and_type[0]
    atomtype = atom_index_and_type[1]
    if atomtype == 'not found':
        print('Atom not found')
    # search cases of which given atom type may be a part

    if atomtype == 'O':
        same_type_lst = []
        atom_lst = []
        atom_lst.append(e)

        Hindex = compare_atoms(e, 'H', HO_L)
        if Hindex == "not found":
            # Lead Titanate 1 Ti surrounded by 3 O's and 1 Pb
            Tiindex = compare_atoms(e, 'Ti', TiO_L)
            if Tiindex == 'not found':
                print('Atom not found')
            else:
                atom_lst.append(Tiindex)
            Pbindex = compare_atoms(Tiindex, 'Pb', TiPb_L)
            if Pbindex == 'not found':
                print('Atom not found')
            else:
                atom_lst.append(Pbindex)
                same_type_lst.append(e)
            O2index = compare_atoms_same_type(Tiindex, 'O', TiO_L, same_type_lst)
            if O2index == 'not found':
                print('Atom not found')
            else:
                atom_lst.append(O2index)
                same_type_lst.append(O2index)
            O3index = compare_atoms_same_type(Tiindex, 'O', TiO_L, same_type_lst)
            if O3index == 'not found':
                print('Atom not found')
            else:
                atom_lst.append(O3index)
                setter(atom_lst)
        else:
            atom_lst.append(Hindex)
            same_type_lst.append(Hindex)

        H2index = compare_atoms_same_type(e, 'H', HO_L, same_type_lst)
        if H2index == "not found":
            setter(atom_lst)
            print('Atom not found')
        else:
            atom_lst.append(H2index)
            same_type_lst.append(H2index)

        H3index = compare_atoms_same_type(e, 'H', HO_L, same_type_lst)
        if H3index == "not found":
            setter(atom_lst)
            print('Atom not found')
        else:
            atom_lst.append(H3index)
            setter(atom_lst)

    if atomtype == 'H':
        same_type_lst = []
        atom_lst = []
        atom_lst.append(e)
        same_type_lst.append(e)

        Oindex = compare_atoms(e, 'O', HO_L)
        if Oindex == "not found":
            print('Atom not found')
        else:
            atom_lst.append(Oindex)

        H2index = compare_atoms_same_type(Oindex, 'H', HO_L, same_type_lst)
        if H2index == "not found":
            setter(atom_lst)
            print('Atom not found')
        else:
            atom_lst.append(H2index)
            same_type_lst.append(H2index)

        H3index = compare_atoms_same_type(Oindex, 'H', HO_L, same_type_lst)
        if H3index == "not found":
            setter(atom_lst)
            print('Atom not found')
        else:
            atom_lst.append(H3index)
            setter(atom_lst)

    if atomtype == 'Pb':
        same_type_lst = []
        atom_lst = []
        atom_lst.append(e)

        Tiindex = compare_atoms(e, 'Ti', TiPb_L)
        if Tiindex == "not found":
            print('Atom not found')
        else:
            atom_lst.append(Tiindex)

        Oindex = compare_atoms(Tiindex, 'O', TiO_L)
        if Oindex == "not found":
            print('Atom not found')
        else:
            atom_lst.append(Oindex)
            same_type_lst.append(Oindex)

        O2index = compare_atoms_same_type(Tiindex, 'O', TiO_L, same_type_lst)
        if O2index == "not found":
            print('Atom not found')
        else:
            atom_lst.append(O2index)
            same_type_lst.append(O2index)

        O3index = compare_atoms_same_type(Tiindex, 'O', TiO_L, same_type_lst)
        if O3index == "not found":
            print('Atom not found')
        else:
            atom_lst.append(O3index)
            setter(atom_lst)

    if atomtype == 'Ti':
        same_type_lst = []
        atom_lst = []
        atom_lst.append(e)

        Pbindex = compare_atoms(e, 'Pb', TiPb_L)
        if Pbindex == "not found":
            print('Atom not found')
        else:
            atom_lst.append(Pbindex)

        Oindex = compare_atoms(e, 'O', TiO_L)
        if Oindex == "not found":
            print('Atom not found')
        else:
            atom_lst.append(Oindex)
            same_type_lst.append(Oindex)

        O2index = compare_atoms_same_type(e, 'O', TiO_L, same_type_lst)
        if O2index == "not found":
            print('Atom not found')
        else:
            atom_lst.append(O2index)

        O3index = compare_atoms_same_type(e, 'O', TiO_L, same_type_lst)
        if O3index == "not found":
            print('Atom not found')
        else:
            atom_lst.append(O3index)
            setter(atom_lst)

def surface_bulk():
    """Radius from center atom in molecule is used to find proximity to centers of *other atoms* and
            determine if from Surface/Bulk (For HO/H2O/H3O center atom type is 'O' w/Molecule as HO/H2O/H3O.
            for PbTiO3 center atom type is Ti w/Molecule PbTiO3.)
            *Other Atoms* used to define Surface/Bulk are PbTiO3"""
    """For molecules TiHO2 and PbOH the presence of the bond to the Surface by checking for TiHO2, if an OH is in 
    the necessary proximity to 2 Ti with z-component < 0.5 Angstrom and for PbOH if and OH is in the necessary 
    proximity to 3 Pb with z-component < 0.5 Angstrom"""
    global PbOHGlobalIndicies
    global TiHO2GlobalIndicies
    for molecule in range(nStart, nStop):
        if ((df.loc[molecule, 'Molecule'] == 'OH') or (df.loc[molecule, 'Molecule'] == 'H2O') or (
                df.loc[molecule, 'Molecule'] == 'H3O')) and (df.loc[molecule, 'ATOM'] == 'O'):
            compare_mol(molecule)
            # Is equivalent to:
            surface_atom_index = compare_atoms(molecule, 'Ti', SURFACE_RADIUS)
            if surface_atom_index == "not found":
                setter_surface(get_others(molecule), False)
            else:
                setter_surface(get_others(molecule), True)
        elif (df.loc[molecule, 'Molecule'] == 'PbTiO3') and (df.loc[molecule, 'ATOM'] == 'Ti'):
            compare_mol(molecule)
        elif (df.loc[molecule, 'Molecule'] == 'TiHO2') and (df.loc[molecule, 'ATOM'] == 'O'):
            # Then look for another Ti distance TiO_L and then make sure one has a z component < 0.5
            # Atom species corresponding to indicies contained in list TiHO2GlobalIndicies: [O,H,Ti,O]
            t = [TiHO2GlobalIndicies[2]]
            Tiindex = compare_atoms_same_type(TiHO2GlobalIndicies[0], 'Ti', TiO_L, t)
            if Tiindex == 'not found':
                # If 2nd Ti not found at distance TiO_L from O in TiHO2 then it is TiHO2 from Bulk
                # for atom_index in TiHO2GlobalIndicies:
                setter_surface(TiHO2GlobalIndicies, 'Bulk')
            # Check condition to determine TiHO2 Surface or Bulk
            elif (dist(TiHO2GlobalIndicies[3], TiHO2GlobalIndicies[2])[1] < 0.5) and (
                    dist(TiHO2GlobalIndicies[3], Tiindex)[1] < 0.5):
                # for atom_index in TiHO2GlobalIndicies:
                setter_surface(TiHO2GlobalIndicies, 'Surface')
            else:
                # for atom_index in TiHO2GlobalIndicies:
                setter_surface(TiHO2GlobalIndicies, 'Bulk')
        elif (df.loc[molecule, 'Molecule'] == 'PbOH') and (df.loc[molecule, 'ATOM'] == 'O'):
            # Then look for 2 other Pb distance PbO and then make sure one has a z component < 0.5
            # Atom species corresponding to indicies contained in list PbOHGlobalIndicies: [O,H,Pb]
            p = [PbOHGlobalIndicies[2]]
            Pbindex = compare_atoms_same_type(PbOHGlobalIndicies[0], 'Pb', PbOH_pbO, p)
            p.append(Pbindex)
            Pbindex2 = compare_atoms_same_type(PbOHGlobalIndicies[0], 'Pb', PbOH_pbO, p)
            if Pbindex2 == 'not found':
                # If 3rd Pb not found at distance PbOH_pbO from O in PbOH then it is PbOH from Bulk
                # for atom_index in PbOHGlobalIndicies:
                setter_surface(PbOHGlobalIndicies, 'Bulk')
            # Check condition to determine PbOH Surface or Bulk
            elif (dist(PbOHGlobalIndicies[0], PbOHGlobalIndicies[2])[1] < 0.5) and (
                    dist(PbOHGlobalIndicies[0], Pbindex)[1] < 0.5) and (dist(PbOHGlobalIndicies[0], Pbindex2)[1] < 0.5):
                # for atom_index in PbOHGlobalIndicies:
                setter_surface(PbOHGlobalIndicies, 'Surface')
            else:
                # for atom_index in PbOHGlobalIndicies:
                setter_surface(PbOHGlobalIndicies, 'Bulk')
        else:
            continue

def get_others(atom_index):
    """Takes in atom index; Returns list of atom indices belonging to the same molecule as that atom"""
    molecule_list = [atom_index]
    molecule_size = 0
    if df.loc[atom_index, 'Molecule'] == 'OH':
        molecule_size = 2
    elif df.loc[atom_index, 'Molecule'] == 'H2O':
        molecule_size = 3
    elif df.loc[atom_index, 'Molecule'] == 'H3O':
        molecule_size = 4
    elif df.loc[atom_index, 'Molecule'] == 'PbTiO3':
        molecule_size = 5
    elif df.loc[atom_index, 'Molecule'] == 'TiHO2':
        molecule_size = 4
    elif df.loc[atom_index, 'Molecule'] == 'PbOH':
        molecule_size = 3
    for i in range(nStart, nStop):
        if df.loc[i, 'Molecule Index'] == df.loc[atom_index, 'Molecule Index']:
            molecule_list.append(i)
            if len(molecule_list) == molecule_size:
                return molecule_list
            else:
                continue