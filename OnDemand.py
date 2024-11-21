import math
import os

import numpy as np
import pandas as pd

import time

start_time = time.time()


def find(name, path):
    """Takes in file name ans path and finds first match
    of file name and path and returns joined root and name"""
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


file = find('pbtio3x10.xyz', 'C:\PHY487')
print(file)

# read file into a DataFrame and select val at [0][0]
natoms = pd.read_table(file, header=None, delim_whitespace=True, nrows=1)[0][0]
# df = pd.DataFrame()
drop_list = [0, 1]
df = pd.read_table(file, header=None, skiprows=drop_list, delim_whitespace=True)
df = df.dropna()
df.columns = ['ATOM', 'X', 'Y', 'Z']
df['STEP'] = df.reset_index().index // natoms + 1
# print(df.index)
print("Number of Atoms: {}".format(natoms))
# b=df.index
s = set(df['STEP'])
print("Number of Time Steps: {}".format(len(s)))
nan_value = float("NaN")
df['Molecule'] = nan_value
df['Molecule Index'] = nan_value
df['TiHO2/PbOH'] = nan_value
df['TiHO2/PbOH Index'] = nan_value
df['Surface'] = nan_value

HO_L = 1.20
TiO_L = 2.20
TiPb_L = 3.70
OHTiO_tio = 2.20  # at least 2 Ti z<0.5
PbOH_pbO = 3.30  # 3 Pb z<0.5
SURFACE_RADIUS = 3.00
LATTICE_SPACING = 4.00
TiHO2GlobalIndicies = []
PbOHGlobalIndicies = []
count_O = 0
count_H = 0
count_Pb = 0
count_Ti = 0
count_OH = 0
count_H2O = 0
count_H3O = 0
count_PbTiO3 = 0
count_TiHO2 = 0
count_PbOH = 0


def search_multiple_strings_in_file(file_name, list_of_strings):
    """Get line from the file along with line numbers, which contains any string from the list"""
    line_number = 0
    list_of_results = []
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains any string from the list of strings
            for string_to_search in list_of_strings:
                if string_to_search in line:
                    # If any string is found in line, then append that line along with line number in list
                    list_of_results.append((string_to_search, line_number, line.rstrip()))
            line_number += 1

    # Return list of tuples containing matched string, line numbers and lines where string is found
    return list_of_results


def extract_lines_between_two_lines(file_name, start_of_block, end_of_block):
    """Search file for block and extract the species info. from the rows inside the block headers"""
    # f = open(fileB)
    # allLines = f.readlines()
    # f.close()
    lines = (range(start_of_block + 1, end_of_block))
    block_lines = []
    line_number = 0
    # Open the file in read only mode
    with open(file_name, 'r') as fileHandler:
        # Read all lines in the file one by one
        for line in fileHandler:
            # For each line, check if line matches index of line between header and footer
            if line_number in lines:
                # Copy string; split string on whitespaces; append to list
                block_lines.append((line.strip().split()))
            line_number += 1

    # Return list of tuples containing strings from all lines in between block headers
    return block_lines


# search file for Chemical Species label block and extract the species info. from the rows inside the block headers
fileB = find('65h2o_pto.fdf', 'C:\PHY487')
print(fileB)
species_block_headers = search_multiple_strings_in_file(fileB,
                                                        ["%block ChemicalSpeciesLabel",
                                                         "%endblock ChemicalSpeciesLabel"])
print('Total Matched lines : ', len(species_block_headers))
for elem in species_block_headers:
    print('Word = ', elem[0], ' :: Line Number = ', elem[1], ' :: Line = ', elem[2])
print("")

species_block_lines = extract_lines_between_two_lines(fileB, species_block_headers[0][1], species_block_headers[1][1])
species_df = pd.DataFrame(species_block_lines, columns=['Index', 'Atomic #', 'Species'])
print('# of Species = ', len(species_block_lines))
# print('# of Species = ', species_df.size/len(species_df.columns))
for elem in range(len(species_block_lines)):
    print('Species = ', species_df.loc[elem, 'Species'], ' :: Atomic # = ', species_df.loc[elem, 'Atomic #'])
print("")

bc_block_headers = search_multiple_strings_in_file(fileB, ['%block LatticeVectors', '%endblock LatticeVectors'])
bc_block_lines = extract_lines_between_two_lines(fileB, bc_block_headers[0][1], bc_block_headers[1][1])

BC_X = float(bc_block_lines[0][0])
BC_Y = float(bc_block_lines[1][1])
BC_Z = float(bc_block_lines[2][2])

def boundary_conditions(vector_list):
    """Takes in list of x, y, and z euclidean distances and returns list with boundary conditions applied"""
    for i in range(len(vector_list)):
        if i == 0:
            if vector_list[i] > BC_X / 2:
                vector_list[i] = (vector_list[i] - BC_X)
            elif vector_list[i] < -(BC_X / 2):
                vector_list[i] = (vector_list[i] + BC_X)
        elif i == 1:
            if vector_list[i] > BC_Y / 2:
                vector_list[i] = (vector_list[i] - BC_Y)
            elif vector_list[i] < -(BC_Y / 2):
                vector_list[i] = (vector_list[i] + BC_Y)
        elif i == 2:
            if vector_list[i] > BC_Z / 2:
                vector_list[i] = (vector_list[i] - BC_Z)
            elif vector_list[i] < -(BC_Z / 2):
                vector_list[i] = (vector_list[i] + BC_Z)

    return vector_list


def dist(atom_a, atom_b):
    """Takes in two atom indexes, gets their coordinates from the dataframe,
    and returns the distance between them with boundary conditions applied"""
    lst_1 = []
    lst_2 = []
    pwr_lst_p = []
    pwr_lst_n = []
    vec_lst_p = []
    vec_lst_n = []
    lst_1.append(df.loc[atom_a, 'X'])
    lst_1.append(df.loc[atom_a, 'Y'])
    lst_1.append(df.loc[atom_a, 'Z'])
    lst_2.append(df.loc[atom_b, 'X'])
    lst_2.append(df.loc[atom_b, 'Y'])
    lst_2.append(df.loc[atom_b, 'Z'])
    # zip coordinates
    zipped_list = list(zip(lst_1, lst_2))
    for (a, b) in zipped_list:  # (x1,x2),(y1,y2),(z1,z2)
        vec_lst_p.append(a - b)
        vec_lst_n.append(b - a)
    vec_lst_pb = boundary_conditions(vec_lst_p)
    vec_lst_nb = boundary_conditions(vec_lst_n)
    for i in vec_lst_pb:
        pwr_lst_p.append(math.pow(i, 2))
    mag_p = math.sqrt(sum(pwr_lst_p))
    for i in vec_lst_nb:
        pwr_lst_n.append(math.pow(i, 2))
    mag_n = math.sqrt(sum(pwr_lst_n))

    # Return magnitude of distance and z-component
    return mag_n, abs(vec_lst_nb[2])


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


def setter_PbOH(atom_lst):
    """Takes in list of atom indices belonging to molecule and sets their 'Molecule' in the dataframe for PbOH"""
    global count_PbOH
    count_PbOH += 1
    for atom_index in atom_lst:
        df.loc[atom_index, 'TiHO2/PbOH'] = "PbOH"
        df.loc[atom_index, 'TiHO2/PbOH Index'] = count_PbOH


def setter_OHTiO(atom_lst):
    """Takes in list of atom indices belonging to molecule and sets their 'Molecule' in the dataframe for TiHO2"""
    global count_TiHO2
    count_TiHO2 += 1
    for atom_index in atom_lst:
        df.loc[atom_index, 'TiHO2/PbOH'] = "TiHO2"
        df.loc[atom_index, 'TiHO2/PbOH Index'] = count_TiHO2


def setter(atom_lst):
    """Takes in list of atom indices belonging to molecule and sets their 'Molecule' in the dataframe"""
    global count_O
    global count_H
    global count_Pb
    global count_Ti
    global count_OH
    global count_H2O
    global count_H2O
    global count_H3O
    global count_PbTiO3

    if len(atom_lst) == 1:
        if df.loc[atom_lst[0], 'ATOM'] == 'O':
            count_O += 1
            df.loc[atom_lst[0], 'Molecule'] = 'O'
            df.loc[atom_lst[0], 'Molecule Index'] = count_O
        elif df.loc[atom_lst[0], 'ATOM'] == 'H':
            count_H += 1
            df.loc[atom_lst[0], 'Molecule'] = 'H'
            df.loc[atom_lst[0], 'Molecule Index'] = count_H
        elif df.loc[atom_lst[0], 'ATOM'] == 'Pb':
            count_Pb += 1
            df.loc[atom_lst[0], 'Molecule'] = 'Pb'
            df.loc[atom_lst[0], 'Molecule Index'] = count_Pb
        elif df.loc[atom_lst[0], 'ATOM'] == 'Ti':
            count_Ti += 1
            df.loc[atom_lst[0], 'Molecule'] = 'Ti'
            df.loc[atom_lst[0], 'Molecule Index'] = count_Ti
    elif len(atom_lst) == 2:
        count_OH += 1
        df.loc[atom_lst[0], 'Molecule'] = 'OH'
        df.loc[atom_lst[1], 'Molecule'] = 'OH'
        df.loc[atom_lst[0], 'Molecule Index'] = count_OH
        df.loc[atom_lst[1], 'Molecule Index'] = count_OH
    elif len(atom_lst) == 3:
        count_H2O += 1
        df.loc[atom_lst[0], 'Molecule'] = 'H2O'
        df.loc[atom_lst[1], 'Molecule'] = 'H2O'
        df.loc[atom_lst[2], 'Molecule'] = 'H2O'
        df.loc[atom_lst[0], 'Molecule Index'] = count_H2O
        df.loc[atom_lst[1], 'Molecule Index'] = count_H2O
        df.loc[atom_lst[2], 'Molecule Index'] = count_H2O
    elif len(atom_lst) == 4:
        count_H3O += 1
        df.loc[atom_lst[0], 'Molecule'] = 'H3O'
        df.loc[atom_lst[1], 'Molecule'] = 'H3O'
        df.loc[atom_lst[2], 'Molecule'] = 'H3O'
        df.loc[atom_lst[3], 'Molecule'] = 'H3O'
        df.loc[atom_lst[0], 'Molecule Index'] = count_H3O
        df.loc[atom_lst[1], 'Molecule Index'] = count_H3O
        df.loc[atom_lst[2], 'Molecule Index'] = count_H3O
        df.loc[atom_lst[3], 'Molecule Index'] = count_H3O
    elif len(atom_lst) == 5:
        count_PbTiO3 += 1
        df.loc[atom_lst[0], 'Molecule'] = 'PbTiO3'
        df.loc[atom_lst[1], 'Molecule'] = 'PbTiO3'
        df.loc[atom_lst[2], 'Molecule'] = 'PbTiO3'
        df.loc[atom_lst[3], 'Molecule'] = 'PbTiO3'
        df.loc[atom_lst[4], 'Molecule'] = 'PbTiO3'
        df.loc[atom_lst[0], 'Molecule Index'] = count_PbTiO3
        df.loc[atom_lst[1], 'Molecule Index'] = count_PbTiO3
        df.loc[atom_lst[2], 'Molecule Index'] = count_PbTiO3
        df.loc[atom_lst[3], 'Molecule Index'] = count_PbTiO3
        df.loc[atom_lst[4], 'Molecule Index'] = count_PbTiO3


def same_type(same_type_lst, i):
    """Compares the indices of two atoms of the same type
    to ensure they are different and not the same atom counted twice"""
    count = 0
    for j in range(len(same_type_lst)):
        # check every element of same_type_lst
        if same_type_lst[j] == i:
            count += 1
        else:
            count = count
    return count


def get_atom_xyz(ts_start, ts_stop, x, y, z):
    """takes in user specified time step start and stop, and x,y,z coordinates;
    Searches for coordinates in specified time step and returns index and type"""
    for index in range(ts_start, ts_stop):
        if df.loc[index, 'X'] == x and df.loc[index, 'Y'] == y and df.loc[index, 'Z'] == z:
            return index, df.loc[index, 'ATOM']
    else:
        return ['not found', 'not found']


def getatom_type(atom_index):
    """Receives atom index and returns type from dataframe"""
    if pd.isnull(df.loc[atom_index, 'Molecule']):
        return df.loc[atom_index, 'ATOM']
    else:
        return "not found"


# def getatom_index(atom_type):
#     for i in range(nStart, nStop): #(0 to 274)
#         if (df.loc[i, 'ATOM'] == atom_type) and
#         (pd.isnull(df.loc[i, 'Molecule']) == True): #can prob make more efficient
#             return i

def compare_atoms(atom1index, atom2type, d):
    """Compares two atoms to find if they belong to the same molecule;
    takes in first atom type, second atom index, and distance parameter
    for particular molecule; returns index of second atom"""
    for i in range(nStart, nStop):
        if df.loc[i, 'ATOM'] == atom2type:
            if dist(atom1index, i)[0] <= d:
                return i
    else:
        return "not found"


def compare_atoms_same_type(atom1index, atom2type, d, same_type_lst):
    """Compares two atoms of the same type to find if they belong to the same molecule;
    takes in first atom type, second atom index, distance parameter, and list of atom
    indices of the same atom type for particular molecule; returns index of second atom"""
    for i in range(nStart, nStop):
        if df.loc[i, 'ATOM'] == atom2type:
            count = same_type(same_type_lst, i)
            if (count == 0) and (dist(atom1index, i)[0] <= d):
                return i
    else:
        return "not found"


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


def find_molecule():
    """For each atom in the time step this function checks for other atoms which may be in it's proximity to
    determine to which molecule it may belong and checks molecules proximity to surface """
    global PbOHGlobalIndicies
    global TiHO2GlobalIndicies
    # test el by el of time step for type/Molecule
    for e in range(nStart, nStop):
        # get atom type at current index
        atomtype = getatom_type(e)
        if atomtype == 'not found':
            continue
        # search cases of which given atom type may be a part
        if e == 234334:
            print('')

        if atomtype == 'O':
            atom_lst = [e]  # This list is use to keep track of all atoms in a molecule
            same_type_lst = []  # This list is used to keep track of atoms of the same type in a molecule (ie. all
            # Oxygen in PbTiO3 or Hydrogen in H2O) This is used as the final arg. in the func. compare_atoms_same_type()

            # Check for first Hydrogen
            Hindex = compare_atoms(e, 'H', HO_L)
            if Hindex == "not found":
                # Lead Titanate; 1 Ti surrounded by 3 O's and 1 Pb
                # Check for Titanium
                Tiindex = compare_atoms(e, 'Ti', TiO_L)
                if Tiindex == 'not found':
                    # No Hydrogen or Titanium found; set as single Oxygen
                    setter(atom_lst)
                    # Check & set proximity to surface
                    surface_atom_index = compare_atoms(e, 'Ti', SURFACE_RADIUS)
                    if surface_atom_index == "not found":
                        setter_surface(atom_lst, False)
                    else:
                        setter_surface(atom_lst, True)
                    continue
                else:
                    # Add Titanium to atom_lst
                    atom_lst.append(Tiindex)
                # Check for Lead
                Pbindex = compare_atoms(Tiindex, 'Pb', TiPb_L)
                if Pbindex == 'not found':
                    # Only found an Oxygen and a Titanium at required distance from one another
                    continue
                else:
                    # Add Lead to atom_lst
                    atom_lst.append(Pbindex)
                    # Add first Oxygen to same_type_lst because we now need to search for additional Oxygen
                    same_type_lst.append(e)
                # Check for second Oxygen
                O2index = compare_atoms_same_type(Tiindex, 'O', TiO_L, same_type_lst)
                if O2index == 'not found':
                    # THIS CASE SHOULD NOT OCCUR
                    # Only found O, Ti, and Pb
                    continue
                else:
                    # Add second Oxygen to atom_lst
                    atom_lst.append(O2index)
                    # Add second Oxygen to same_type_lst
                    same_type_lst.append(O2index)
                # Check for third Oxygen
                O3index = compare_atoms_same_type(Tiindex, 'O', TiO_L, same_type_lst)
                if O3index == 'not found':
                    # Only found O, Ti, Pb, and second O
                    continue
                else:
                    # Add third Oxygen to atom_lst
                    atom_lst.append(O3index)
                    # Set as PbTiO3
                    setter(atom_lst)
                    # Check & set proximity to surface
                    ti_lst = [Tiindex]
                    surface_atom_index = compare_atoms_same_type(Tiindex, 'Ti', LATTICE_SPACING, ti_lst)
                    if surface_atom_index == "not found":
                        setter_surface(atom_lst, False)
                    else:
                        setter_surface(atom_lst, True)
                    continue
            else:
                # Add first Hydrogen to atom_lst
                atom_lst.append(Hindex)
                # Add first Hydrogen to same_type_lst
                same_type_lst.append(Hindex)

            # Check for second Hydrogen
            H2index = compare_atoms_same_type(e, 'H', HO_L, same_type_lst)
            if H2index == "not found":
                # Check for TiHO2
                Tiindex = compare_atoms(e, 'Ti', TiO_L)
                if Tiindex == "not found":
                    # Check for PbOH
                    Pbindex = compare_atoms(e, 'Pb', PbOH_pbO)
                    if Pbindex == 'not found':
                        # Set as OH
                        setter(atom_lst)
                        # Check & set proximity to surface
                        surface_atom_index = compare_atoms(e, 'Ti', SURFACE_RADIUS)
                        if surface_atom_index == "not found":
                            setter_surface(atom_lst, False)
                        else:
                            setter_surface(atom_lst, True)
                        continue
                    else:
                        # First set O and H as OH in 'Molecule' column
                        setter(atom_lst)
                        # Add Pb to atom_lst
                        atom_lst.append(Pbindex)
                        # Set as PbOH in TiHO2/PbOH column
                        setter_PbOH(atom_lst)
                        # PbOHGlobalIndicies = atom_lst  # [O,H,Pb]
                        # Check & set proximity to surface
                        pb_lst = [Pbindex]
                        Pb2index = compare_atoms_same_type(e, 'Pb', PbOH_pbO, pb_lst)
                        pb_lst.append(Pb2index)
                        Pb3index = compare_atoms_same_type(e, 'Pb', PbOH_pbO, pb_lst)
                        if Pb2index == "not found" or Pb3index == "not found":
                            # Should Not Occur (If PbOH exists it should be bonded to the surface, as well)
                            # Atoms not found
                            pass
                        elif (dist(e, Pbindex)[1] < 0.5) and (dist(e, Pb2index)[1] < 0.5) and (
                                dist(e, Pb3index)[1] < 0.5):
                            # Both conditions met
                            setter_surface(atom_lst, True)
                        else:
                            # Should Not Occur (If PbOH exists it should be bonded to the surface, as well)
                            # Atoms found but not meeting z-component condition
                            setter_surface(atom_lst, False)
                        continue
                else:
                    # Set O and H
                    setter(atom_lst)
                    # Add Titanium to atom_lst
                    atom_lst.append(Tiindex)
                    # Create second same_type_lst for Oxygen b/c first in use for Hydrogen
                    same_type_lst_OHTiO = [e]
                    # Check for second Oxygen
                    Oindex = compare_atoms_same_type(Tiindex, 'O', TiO_L, same_type_lst_OHTiO)
                    if Oindex == "not found":
                        # THIS CASE SHOULD NOT OCCUR
                        # Only found O, H, and Ti
                        # Remove Ti from atom_lst
                        # atom_lst.pop()
                        # # Set as OH
                        # setter(atom_lst)
                        continue
                    else:
                        # Add Oxygen to atom_lst
                        atom_lst.append(Oindex)
                        # Set as TiHO2
                        setter_OHTiO(atom_lst)
                        # TiHO2GlobalIndicies = atom_lst  # [O,H,Ti,O]
                        ti_lst = [Tiindex]
                        Ti2index = compare_atoms_same_type(e, 'Ti', TiO_L, ti_lst)
                        if Ti2index == 'not found':
                            # Should Not Occur (If TiHO2 exists it should be bonded to the surface, as well)
                            # If 2nd Ti not found at distance TiO_L from O in TiHO2 then it is TiHO2 from Bulk
                            pass
                        # Check condition to determine TiHO2 Surface or Bulk
                        elif (dist(e, Tiindex)[1] < 0.5) and (dist(e, Ti2index)[1] < 0.5):
                            setter_surface(atom_lst, True)
                        else:
                            # Should Not Occur (If TiHO2 exists it should be bonded to the surface, as well)
                            setter_surface(atom_lst, False)
                        continue
            else:
                # Add second Hydrogen to atom_lst
                atom_lst.append(H2index)
                # Add second Hydrogen to same_type_lst
                same_type_lst.append(H2index)

            # Check for third Hydrogen
            H3index = compare_atoms_same_type(e, 'H', HO_L, same_type_lst)
            if H3index == "not found":
                # Set as H2O
                setter(atom_lst)
                # Check & set proximity to surface
                surface_atom_index = compare_atoms(e, 'Ti', SURFACE_RADIUS)
                if surface_atom_index == "not found":
                    setter_surface(atom_lst, False)
                else:
                    setter_surface(atom_lst, True)
                continue
            else:
                # Add third Hydrogen to atom_lst
                atom_lst.append(H3index)
                # Set as H3O
                setter(atom_lst)
                # Check & set proximity to surface
                surface_atom_index = compare_atoms(e, 'Ti', SURFACE_RADIUS)
                if surface_atom_index == "not found":
                    setter_surface(atom_lst, False)
                else:
                    setter_surface(atom_lst, True)
                continue

        if atomtype == 'H':
            same_type_lst = [e]
            atom_lst = [e]

            Oindex = compare_atoms(e, 'O', HO_L)
            if Oindex == "not found":
                # Set as single Hydrogen
                setter(atom_lst)
                # Check & set proximity to surface
                surface_atom_index = compare_atoms(e, 'Ti', SURFACE_RADIUS)
                if surface_atom_index == "not found":
                    setter_surface(atom_lst, False)
                else:
                    setter_surface(atom_lst, True)
                continue
            else:
                # Add Oxygen to atom_lst
                atom_lst.append(Oindex)

            # Check for second Hydrogen
            H2index = compare_atoms_same_type(Oindex, 'H', HO_L, same_type_lst)
            if H2index == "not found":
                # Check for TiHO2
                Tiindex = compare_atoms(Oindex, 'Ti', TiO_L)
                if Tiindex == "not found":
                    # Check for PbOH
                    Pbindex = compare_atoms(Oindex, 'Pb', PbOH_pbO)
                    if Pbindex == "not found":
                        # Set as OH
                        setter(atom_lst)
                        # Check & set proximity to surface
                        surface_atom_index = compare_atoms(Oindex, 'Ti', SURFACE_RADIUS)
                        if surface_atom_index == "not found":
                            setter_surface(atom_lst, False)
                        else:
                            setter_surface(atom_lst, True)
                        continue
                    else:
                        # First set O and H as OH
                        setter(atom_lst)
                        # Set as PbOH
                        atom_lst.append(Pbindex)
                        setter_PbOH(atom_lst)
                        # PbOHGlobalIndicies = atom_lst  # [H,O,Pb]
                        # Check & set proximity to surface
                        pb_lst = [Pbindex]
                        Pb2index = compare_atoms_same_type(Oindex, 'Pb', PbOH_pbO, pb_lst)
                        pb_lst.append(Pb2index)
                        Pb3index = compare_atoms_same_type(Oindex, 'Pb', PbOH_pbO, pb_lst)
                        if Pb2index == "not found" or Pb3index == "not found":
                            # Should Not Occur (If PbOH exists it should be bonded to the surface, as well)
                            pass
                        elif (dist(Oindex, Pbindex)[1] < 0.5) and (dist(Oindex, Pb2index)[1] < 0.5) and (
                                dist(Oindex, Pb3index)[1] < 0.5):
                            setter_surface(atom_lst, True)
                        else:
                            # Should Not Occur (If PbOH exists it should be bonded to the surface, as well)
                            setter_surface(atom_lst, False)
                        continue
                else:
                    # First set O and H as OH
                    setter(atom_lst)
                    # Add Titanium to atom_lst
                    atom_lst.append(Tiindex)
                    # Create second same_type_lst for Oxygen b/c first in use for Hydrogen
                    same_type_lst_OHTiO = [Oindex]
                    # Check for second Oxygen
                    O2index = compare_atoms_same_type(Tiindex, 'O', TiO_L, same_type_lst_OHTiO)
                    if O2index == "not found":
                        # THIS CASE SHOULD NOT OCCUR
                        # Only found H, O, and Ti
                        # Remove Ti from atom_lst
                        # atom_lst.pop()
                        # # Set as OH
                        # setter(atom_lst)
                        continue
                    else:
                        # Add second Oxygen to atom_lst
                        atom_lst.append(O2index)
                        # Set as TiHO2
                        setter_OHTiO(atom_lst)
                        # TiHO2GlobalIndicies = atom_lst  # [H,O,Ti,O]
                        ti_lst = [Tiindex]
                        Ti2index = compare_atoms_same_type(Oindex, 'Ti', TiO_L, ti_lst)
                        if Ti2index == 'not found':
                            # Should Not Occur (If TiHO2 exists it should be bonded to the surface, as well)
                            # If 2nd Ti not found at distance TiO_L from O in TiHO2 then it is TiHO2 from Bulk
                            pass
                        # Check condition to determine TiHO2 Surface or Bulk
                        elif (dist(Oindex, Tiindex)[1] < 0.5) and (dist(Oindex, Ti2index)[1] < 0.5):
                            setter_surface(atom_lst, True)
                        else:
                            # Should Not Occur (If TiHO2 exists it should be bonded to the surface, as well)
                            setter_surface(atom_lst, False)
                        continue
            else:
                # Add second Hydrogen to atom_lst
                atom_lst.append(H2index)
                # Add second Hydrogen to same_type_lst
                same_type_lst.append(H2index)

            # Check for third Hydrogen
            H3index = compare_atoms_same_type(Oindex, 'H', HO_L, same_type_lst)
            if H3index == "not found":
                # Set as H2O
                setter(atom_lst)
                # Check & set proximity to surface
                surface_atom_index = compare_atoms(Oindex, 'Ti', SURFACE_RADIUS)
                if surface_atom_index == "not found":
                    setter_surface(atom_lst, False)
                else:
                    setter_surface(atom_lst, True)
                continue
            else:
                # Add third Hydrogen to atom_lst
                atom_lst.append(H3index)
                # Set as H3O
                setter(atom_lst)
                # Check & set proximity to surface
                surface_atom_index = compare_atoms(Oindex, 'Ti', SURFACE_RADIUS)
                if surface_atom_index == "not found":
                    setter_surface(atom_lst, False)
                else:
                    setter_surface(atom_lst, True)
                continue

        if atomtype == 'Pb':
            same_type_lst = []
            atom_lst = [e]

            # Check for Titanium
            Tiindex = compare_atoms(e, 'Ti', TiPb_L)
            if Tiindex == "not found":
                # Check for PbOH
                Oindex = compare_atoms(e, 'O', PbOH_pbO)
                if Oindex == "not found":
                    # Set as single Lead
                    setter(atom_lst)
                    # Check & set proximity to surface
                    surface_atom_index = compare_atoms(e, 'Ti', SURFACE_RADIUS)
                    if surface_atom_index == "not found":
                        setter_surface(atom_lst, False)
                    else:
                        setter_surface(atom_lst, True)
                    continue
                else:
                    # Add Oxygen to atom_lst
                    atom_lst.append(Oindex)
                    # Check for Hydrogen
                    Hindex = compare_atoms(Oindex, 'H', HO_L)
                    if Hindex == "not found":
                        # THIS CASE SHOULD NOT OCCUR
                        # Only found Pb, and O
                        continue
                    else:
                        # First set O and H as OH
                        setter(atom_lst[1:])
                        # Add Hydrogen to atom_lst
                        atom_lst.append(Hindex)
                        # Set as PbOH
                        setter_PbOH(atom_lst)
                        # Check & set proximity to surface
                        # PbOHGlobalIndicies = atom_lst  # [Pb,O,H]
                        pb_lst = [e]
                        Pbindex = compare_atoms_same_type(Oindex, 'Pb', PbOH_pbO, pb_lst)
                        pb_lst.append(Pbindex)
                        Pb2index = compare_atoms_same_type(Oindex, 'Pb', PbOH_pbO, pb_lst)
                        if Pbindex == "not found" or Pb2index == "not found":
                            # Should Not Occur (If PbOH exists it should be bonded to the surface, as well)
                            pass
                        elif (dist(Oindex, e)[1] < 0.5) and (dist(Oindex, Pbindex)[1] < 0.5) and (
                                dist(Oindex, Pb2index)[1] < 0.5):
                            setter_surface(atom_lst, True)
                        else:
                            # Should Not Occur (If PbOH exists it should be bonded to the surface, as well)
                            setter_surface(atom_lst, False)
                        continue
            else:
                # Add Titanium to atom_lst
                atom_lst.append(Tiindex)

            # Check for first Oxygen
            Oindex = compare_atoms(Tiindex, 'O', TiO_L)
            if Oindex == "not found":
                # Only Pb, and Ti
                continue
            else:
                # Add first Oxygen to atom_lst
                atom_lst.append(Oindex)
                # Add first Oxygen to same_type_lst
                same_type_lst.append(Oindex)

            # Check for second Oxygen
            O2index = compare_atoms_same_type(Tiindex, 'O', TiO_L, same_type_lst)
            if O2index == "not found":
                # Only found Pb, Ti, and O
                continue
            else:
                # Add second Oxygen to atom_lst
                atom_lst.append(O2index)
                # Add second Oxygen to same_type_lst
                same_type_lst.append(O2index)

            # Check for third Oxygen
            O3index = compare_atoms_same_type(Tiindex, 'O', TiO_L, same_type_lst)
            if O3index == "not found":
                # Only found Pb, Ti, O, and O
                continue
            else:
                # Add third Oxygen to atom_lst
                atom_lst.append(O3index)
                # Set as PbTiO3
                setter(atom_lst)
                # Check & set proximity to surface
                ti_lst = [Tiindex]
                surface_atom_index = compare_atoms_same_type(Tiindex, 'Ti', LATTICE_SPACING, ti_lst)
                if surface_atom_index == "not found":
                    setter_surface(atom_lst, False)
                else:
                    setter_surface(atom_lst, True)
                continue

        if atomtype == 'Ti':
            same_type_lst = []
            atom_lst = [e]

            # Check for Pb
            Pbindex = compare_atoms(e, 'Pb', TiPb_L)
            if Pbindex == "not found":
                # Check for TiHO2
                Oindex = compare_atoms(e, 'O', TiO_L)
                if Oindex == "not found":
                    # Set as single Ti
                    setter(atom_lst)
                    # Check & set proximity to surface
                    surface_atom_index = compare_atoms_same_type(e, 'Ti', LATTICE_SPACING, atom_lst)
                    if surface_atom_index == "not found":
                        setter_surface(atom_lst, False)
                    else:
                        setter_surface(atom_lst, True)
                    continue
                else:
                    # Add first Oxygen to atom_lst
                    atom_lst.append(Oindex)
                    # Create sametypLst_TiHO2 for subprocess and save
                    # same_type_lst for parent process to keep with convention
                    sametypLst_TiHO2 = [Oindex]

                # Check for second Oxygen
                O2index = compare_atoms_same_type(e, 'O', TiO_L, sametypLst_TiHO2)
                if O2index == "not found":
                    # Only found Ti, and O
                    # should not occur
                    continue
                else:
                    # Add second Oxygen to atom_lst
                    atom_lst.append(O2index)

                # Check for Hydrogen
                Hindex = compare_atoms(Oindex, 'H', HO_L)
                if Hindex == "not found":
                    # Only have Ti, O, and O
                    # Should not occur
                    continue
                else:
                    # Add Hydrogen to atom_lst
                    atom_lst.append(Hindex)
                    # First set O and H as OH
                    setter(atom_lst[2:])  # [Ti,O,O,H]
                    # Set as TiHO2
                    setter_OHTiO(atom_lst)
                    # TiHO2GlobalIndicies = atom_lst
                    ti_lst = [e]
                    Ti2index = compare_atoms_same_type(Oindex, 'Ti', TiO_L, ti_lst)
                    if Ti2index == 'not found':
                        # Should Not Occur (If TiHO2 exists it should be bonded to the surface, as well)
                        # If 2nd Ti not found at distance TiO_L from O in TiHO2 then it is TiHO2 from Bulk
                        pass
                    # Check condition to determine TiHO2 Surface or Bulk
                    elif (dist(Oindex, e)[1] < 0.5) and (dist(Oindex, Ti2index)[1] < 0.5):
                        setter_surface(atom_lst, True)
                    else:
                        # Should Not Occur (If TiHO2 exists it should be bonded to the surface, as well)
                        setter_surface(atom_lst, False)
                    continue
            else:
                # Add Lead to atom_lst
                atom_lst.append(Pbindex)

            # Check for first Oxygen
            Oindex = compare_atoms(e, 'O', TiO_L)
            if Oindex == "not found":
                continue
            else:
                # Add first Oxygen to atom_lst
                atom_lst.append(Oindex)
                # Add first Oxygen to same_type_lst
                same_type_lst.append(Oindex)

            # Check for second Oxygen
            O2index = compare_atoms_same_type(e, 'O', TiO_L, same_type_lst)
            if O2index == "not found":
                continue
            else:
                # Add second Oxygen to atom_lst
                atom_lst.append(O2index)
                # Add second Oxygen to same_type_lst
                same_type_lst.append(O2index)

            # Check for third Oxygen
            O3index = compare_atoms_same_type(e, 'O', TiO_L, same_type_lst)
            if O3index == "not found":
                continue
            else:
                # Add third Oxygen to atom_lst
                atom_lst.append(O3index)
                # Set as PbTiO3
                setter(atom_lst)
                # Check & set proximity to surface
                ti_lst = [e]
                surface_atom_index = compare_atoms_same_type(e, 'Ti', LATTICE_SPACING, ti_lst)
                if surface_atom_index == "not found":
                    setter_surface(atom_lst, False)
                else:
                    setter_surface(atom_lst, True)
        if e == 274:
            print('')
            print("time taken to reach end of first time step: {:.2f}s".format(time.time() - start_time))
            print('')


def setter_surface(mol_list, category):
    """Takes in list of atoms in molecule and category and sets 'Surface' column of dataframe to either True
    or False """
    if category:
        for index in mol_list:
            df.loc[index, 'Surface'] = category
    elif not category:
        for index in mol_list:
            df.loc[index, 'Surface'] = category


def compare_mol(atom_index):
    """Takes in atom index and measures distance to Surface to set Surface category"""
    for i in range(nStart, nStop):
        if i != atom_index:
            if (df.loc[i, 'ATOM'] == 'Ti') and (dist(atom_index, i)[0] <= SURFACE_RADIUS):
                setter_surface(get_others(atom_index), True)
                break
            elif i == nStop - 1:
                setter_surface(get_others(atom_index), False)
                break
        else:
            continue


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


# O       2.5089390000      5.7752510000     15.6478280000


Y_N = input("Search by coordinates? Y/N: ")
while Y_N is not ('Y' and 'N'):
    print("INVALID INPUT: Try again: Enter Y or N")
    Y_N = input("Search by coordinates? Y/N: ")
if Y_N == 'Y':
    ts = int(input("Enter time step: "))
    n = natoms * ts
    nStart = ((n - natoms) + (ts - 1))
    nStop = (n + (ts - 1))
    X_Coord = np.float64(input("Enter X coordinate: "))
    Y_Coord = np.float64(input("Enter Y coordinate: "))
    Z_Coord = np.float64(input("Enter Z coordinate: "))
    find_molecule_user_choice(nStart, nStop, X_Coord, Y_Coord, Z_Coord)
else:
    # for ts in s:
    # for every time step in the df
    PbOHGlobalIndicies = []
    TiHO2GlobalIndicies = []
    ts = 850
    # calculate beginning and end of time step
    n = natoms * ts
    nStart = ((n - natoms) + (ts - 1))
    nStop = (n + (ts - 1))
    find_molecule()
    print('')


