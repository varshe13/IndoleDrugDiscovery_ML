from inspect import BoundArguments
from tracemalloc import start
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm as tq
import openpyxl
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdPartialCharges
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMolAlign
from rdkit.Chem import PeriodicTable
from rdkit.Chem import rdEHTTools
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
import os
import scipy as sp
from scipy import spatial
from pathlib import Path
from functools import cmp_to_key
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import py3Dmol
import superpose3d as S3D
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle
import numpy_indexed as npi
from more_itertools import chunked
import h5py

p = AllChem.ETKDGv2()
p.verbose = False
#import superpose3d as S3D
import re
#import open3d as o3d
#import skimage
#from skimage.draw import line


#Defining Indole Class============================================================

#============================================================
class Indole:
    def __init__(self, Mol_inp, Attbs_inp, RDKit_Mol_inp, **kwargs):     #Takes in the Molecules and Attributes from the Parse DB function
        
        #//Defining mol block as self attribute
        self.Mol_block = Mol_inp
        
        if type(Mol_inp) == str:#if input is string just parse inputs
            file_content = Mol_inp 
            file_content = file_content.split('\n')#Converting converting string into list of string
            file_content = [i for i in file_content if i != '']
            file_content = [i.split(' ') for i in file_content]
            file_content = [[j for j in i if j != ''] for i in file_content]
            file_content = [i for i in file_content if i != []]#Removing empty lists from file content

        else:
            #//Reading and defining .mol file information
            self.Mol_Dir = Mol_inp #Defingin .mol file directory
            with open(Mol_inp) as f:
                file_content = [line for line in f.readlines()]
                file_content = [i.split() for i in file_content]#Converting content into a list of lists
                file_content = [i for i in file_content if i != []]#Removing empty lists from file content
        

        #//Defining Molecular Structure Attributes
        self.Num_atoms = int(file_content[1][0])#Unpacking Number of Atoms
        self.Num_bonds = int(file_content[1][1])#Unpacking Number of Bonds
        self.Mol_cord = [np.array([float(j) for j in i[0:3]]) for i in file_content[2:2+self.Num_atoms]]#Generating Molecular Cordinates list of lists from number of atoms
        self.Atom_ident = [i[3] for i in file_content[2:2+self.Num_atoms]]#Generating Atom Identities list from number of atoms
        self.Atom_info_mat = [i[4:] for i in file_content[2:2+self.Num_atoms]]#Generating Atom Identities list from number of atoms
        self.Bond_mat = []
        for i in file_content[2+self.Num_atoms:len(file_content) - 1]:
            temp = []
            validLine = True
            for j in i:
                if j == "M":
                    validLine = False
                    break
                
                temp.append(int(j))
            
            if validLine:
                self.Bond_mat.append(temp)


        # https://stackoverflow.com/a/25046328
        self.Bond_mat.sort(key = lambda x: (x[0], x[1])) # Sorting bond matrix by first atom (column) then second atom (column)
        self.Mol_inp = Mol_inp

        #/The rest of them
        if isinstance(Attbs_inp, pd.DataFrame): #If attributes is inputed as a dataframe
            self.Mol_weight = float(Attbs_inp['Molecular Weight'])
            self.Hydrogen_bond_acceptors = int(Attbs_inp['Hydrogen Bond Acceptors'])
            self.Hydrogen_bond_donors = int(Attbs_inp['Hydrogen Bond Donors'])
            self.Molar_refractivity = float(Attbs_inp['Molar Refractivity'])
            self.ClogP = float(Attbs_inp['ClogP'])
            try:
                self.GP_activity = float(Attbs_inp['Gram Positive Activity'])
            except:
                self.GP_activity = np.nan
            try:
                self.GN_activity = float(Attbs_inp['Gram Negative Activity'])
            except:
                self.GN_activity = np.nan
        elif isinstance(Attbs_inp, Indole): #If attributes is entered as an class indole
            self.Mol_weight = float(Attbs_inp.Mol_weight)
            self.Hydrogen_bond_acceptors = int(Attbs_inp.Hydrogen_bond_acceptors)
            self.Hydrogen_bond_donors = int(Attbs_inp.Hydrogen_bond_donors)
            self.Molar_refractivity = float(Attbs_inp.Molar_refractivity)
            self.ClogP = float(Attbs_inp.ClogP)
            try:
                self.GP_activity = float(Attbs_inp.GP_activity)
            except:
                self.GP_activity = np.nan
                
            try:
                self.GN_activity = float(Attbs_inp.GN_activity)
            except:
                self.GN_activity = np.nan
        elif isinstance(Attbs_inp, list): #If attributes is entered as a list
            self.Mol_weight = Attbs_inp[0]
            self.Hydrogen_bond_acceptors = Attbs_inp[1]
            self.Hydrogen_bond_donors = Attbs_inp[2]
            self.Molar_refractivity = Attbs_inp[3]
            self.ClogP = Attbs_inp[4]
            self.GP_activity = Attbs_inp[5]
            self.GN_activity = Attbs_inp[6]
        else:
            print('Unable to parse attributes inputs')

        self.rdkit_mol = RDKit_Mol_inp #RD_kit obj
        
        try:
            self.partialcharge = kwargs.get('partial_charges')
        except:
            self.partialcharge = None
        
        self.DatabaseSheet = None #Initalizing database sheet as none

    
    def vis(self, **kwargs): #Visualization function to view the molecules in 3D space, with optional highlighting parameters
        print('visualizing . . .')
        try:
            highlight_atms = kwargs.get('highlight_atms')
            vis_mols = kwargs.get('vis_mols')
        except:
            highlight_atms = 0
            vis_mols = 0
                
        if isinstance(vis_mols, list):
            print('Visualizing multiple molecules')
            view = py3Dmol.view(width=500, height=500)
            view.addModel(self.Mol_block, 'mol')#adding self molecule
            for i in vis_mols:#adding other molecules
                view.addModel(i.Mol_block, 'mol')
            view.setStyle('stick')
            view.zoomTo()
            view.show()


        if isinstance(highlight_atms, list):
            highlight_atms = [i-1 for i in highlight_atms]
            print(highlight_atms)
            view = py3Dmol.view(width=500, height=500)
            view.addModel(self.Mol_block, 'mol')
            view.setStyle('stick')
            view.setStyle({'serial':highlight_atms},{'stick':{'color': 'purple'}});
            view.zoomTo()
            view.show()

        if not isinstance(vis_mols, list) and not isinstance(highlight_atms, list):
            print('Visualizing individual without highlighting')
            view = py3Dmol.view(width=200, height=200)
            view.addModel(self.Mol_block, 'mol')
            view.setStyle('stick')
            view.zoomTo()
            view.show()



#Parse Database Functions============================================================

#============================================================
#!!!May need to change if new antibiotic activity measurements change
def Determine_Activity(activity_value:float, aa_cutoff_dict:dict, activity_hml_values:list): #Bin the antibiotic activity values to high/mid/no activity
    if not isinstance(activity_value, float): #Checking if the antibiotic activity is a float
        return(np.nan)
    else:
        if activity_value < aa_cutoff_dict['MIC (uM)'][0]: #If below first cutoff
            return(activity_hml_values[0]) #Return high integer
        elif activity_value > aa_cutoff_dict['MIC (uM)'][0] and activity_value < aa_cutoff_dict['MIC (uM)'][1]: #If above first cutoff but below second
            return(activity_hml_values[1]) #Return mid integer
        elif activity_value > aa_cutoff_dict['MIC (uM)'][1]:  #If above second cutoff
            return(activity_hml_values[2]) #Return low integer
        

def ParseDB_excel(DB, aa_cutoffs, activity_hml_value): #Parses database compound values and activites, generates other values through RDkit
    with pd.ExcelFile(DB) as f: #Opening database
        sheets = f.sheet_names
        SMILES = [] #Initiating SMILES list
        GP_activity = [] #Initializing Gram Positive activity list
        GN_activity = [] #Initializing Gram Negative activity list
        sheet_l = [] #Initializing list of sheets for each molecule
        print('Parsing database sheets:')
        for sht in sheets: #Looping through sheets of database
            print(sht)
            df = f.parse(sht) #Parsing sheet data
            if sht == 'Experimental Values': #If traditional parsing for experiemntal database==================================================================
                for count, i in enumerate(df['SMILES']): #Looping thorugh dataframe
                    sheet_l.append(sht) #Appending sheet to list
                    SMILES.append(i.strip()) #Appending smiles
                    GP_activity.append(df['Gram Positive Activity'][count]) #Appending GP activity
                    GN_activity.append(df['Gram Negative Activity'][count]) #Appending GN activity

            else: #For all litrature data======================================================================================================================
                for count, i in enumerate(df['SMILES']): #Looping thorugh dataframe
                    sheet_l.append(sht) #Appending sheet to list
                    SMILES.append(i.strip()) #Appending litrature SMILES to list
                    activity = [df['GP Activity'][count], df['GN Activity'][count]] #Making temp list of activities
                    try:
                        Descriptors.ExactMolWt(Chem.MolFromSmiles(i)) #Testing if RD.kit can parse SMILE with no errors
                    except:
                        print('Unable to parse following SMILES:')
                        print(df['SMILES'][count], df['Source'][count], df['Litrature Designation'][count])
                    
                    for count2, j in enumerate(activity): #Looping through the two activities (Gram positive and Gram negative)
                        if count2 == 0: #If GP activity
                            if df['Activity Type'][count] == 'MIC (uM)': #If activity is MIC in uM------------------------------------------------
                                GP_activity.append(Determine_Activity(j, aa_cutoffs, activity_hml_value)) #Appending binned activity value to list based on defined cutoffs
                            #!!!Something going wrong here        
                            elif df['Activity Type'][count] == 'MIC (ug/mL)': #If activity is MIC in ug/mL------------------------------------------------
                                try:
                                    uM_activity = j*(1000/Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(df['SMILES'][count]))) #Converting ug/mL activity to uM activity
                                    GP_activity.append(Determine_Activity(uM_activity, aa_cutoffs, activity_hml_value)) #Appending binned activity value to list based on defined cutoffs
                                except:
                                    GP_activity.append(np.nan)
                                    
                        if count2 == 1: #If GN activity
                            if df['Activity Type'][count] == 'MIC (uM)': #If activity is MIC in uM------------------------------------------------
                                GN_activity.append(Determine_Activity(j, aa_cutoffs, activity_hml_value)) #Appending binned activity value to list based on defined cutoffs
     
                            elif df['Activity Type'][count] == 'MIC (ug/mL)': #If activity is MIC in ug/mL------------------------------------------------
                                try:
                                    uM_activity = j*(1000/Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(df['SMILES'][count]))) #Converting ug/mL activity to uM activity
                                    GN_activity.append(Determine_Activity(uM_activity, antibiotic_activity_cutoffs, activity_hml_value)) #Appending binned activity value to list based on defined cutoffs
                                except: #If activity value is Nan append Nan
                                    GN_activity.append(np.nan)

    #Performing initial molecule embedding===========================================================================================================================================================================
    Molecules = [Chem.MolFromSmiles(i) for i in SMILES] #Generating RD.Kit molecule list generation
    Molecule_MolBlocks = [] #Initializing molecule mol block list
    print('Performing initial molecule embedding')
    for i in tq(Molecules): #Performing initial molecule 3D embedding !!!!May not be necessary as a result of molecule registration and embedding!!!
        i = Chem.AddHs(i)#Explicity adding Hydrogens
        AllChem.EmbedMolecule(i, randomSeed=0xf00d)#Embedding Molecules, with consistent random seed for reproducibility
        Molecule_MolBlocks.append(Chem.MolToMolBlock(i))#Converting to Mol_Block and appending to list
        
    #Calculating other relevant values and structuring output list==============================================================================================================================
    output_l = []
    for count, i in tq(enumerate(Molecules)):
        attributes_list = [] #Initializing attributes list
        #Calculating other values
        attributes_list.append(Chem.Descriptors.ExactMolWt(i)) #0Calculating list of molecular weights and appending to list
        attributes_list.append(Chem.Lipinski.NumHAcceptors(i)) #1Calculating the number of Hydrogen bond acceptors of molecule and appending to list
        attributes_list.append(Chem.Lipinski.NumHDonors(i)) #2Calculating the number of Hydrogen bond acceptors of molecule and appending to list
        attributes_list.append(Chem.Crippen.MolMR(i)) #3Calculating the molar refractivity and appending to list
        attributes_list.append(Chem.Crippen.MolLogP(i)) #4Calculating the molecular logP value and appending to list
        attributes_list.append(GP_activity[count]) #5Adding Gram Positive activities to list
        attributes_list.append(GN_activity[count]) #6Adding Gram Negative activities to list
        
        #Building Indole List
        Indole_out = Indole(Molecule_MolBlocks[count], attributes_list, i)
        Indole_out.DatabaseSheet = sheet_l[count]
        output_l.append(Indole_out) #Appending output indole to list

    #Returning output values=============
    return(output_l) #Returning list of indoles



#Registering Conformers
def mol_registration(molecule_list, standard_molecule_idx, confs, **kwargs):
    #Initial embedding========================================================================================
    #Embedding multiple comformers----------------------------------------------------------------------------
    standard = molecule_list[standard_molecule_idx].rdkit_mol #Adding RD.kit mol to object
    standard = Chem.AddHs(standard) #Explicitly adding Hydrogens
    AllChem.EmbedMultipleConfs(standard, 1, p) #Embedding multiple conformers
    contrib_stanard = rdMolDescriptors._CalcCrippenContribs(standard) #Calculating the crippen contributions for the standard molecule
    
    mol_id_l = [] #Initializing list of molecule ids
    sample_l = molecule_list #Removing standard molecule from list
    sample_l_transformed = []

    print('Attempting to register and generate conformers')
    iters = len(sample_l)*confs
    tq._instances.clear()
    with tq(total=iters, position=0, leave=True) as pbar:
        for mol_count, i in tq(enumerate(sample_l), position=0, leave=True):
            initial_sample = i
            sample = Chem.AddHs(i.rdkit_mol) #Explicitly adding Hydrogens
            AllChem.EmbedMultipleConfs(sample, confs, p) #Embedding multiple conformers

            for cid in range(confs): #Looping through and aligning each conformation to the standard
                try: #Attempting to perform direct registration on substructure
                    Chem.rdMolAlign.AlignMol(sample, standard, cid, 0) #Aligning each molecule to substructure
                except: #Passing if failed as the molecules are already in a common vector space
                    pass
                aligned_molblock = Chem.MolToMolBlock(sample, confId=cid) #Generating molblock for each conformer


                #Calculating alinged molblock partial charges
                _, res = rdEHTTools.RunMol(sample, confId=cid) #Running molecule through EHT
                p_charges = res.GetAtomicCharges()[:sample.GetNumAtoms()] #Extracting static charges

                #initial_sample.rdkit_mol = aligned_molblock
                #print(type(aligned_molblock), type(initial_sample), type(sample))
                sample_l_transformed.append(Indole(aligned_molblock, initial_sample, sample, partial_charges = p_charges)) #Creating new list of indoles 
                mol_id_l.append(mol_count) #Appending mol_count as molecule ID
                pbar.update(1)
                
    return(sample_l_transformed, mol_id_l)

#Utility functions=========================================================

#===========================================================

# translates each point stored in matrix by (x, y, z)
def translateMolecule(matrix, x, y, z):
    for i in range(len(matrix)):
        matrix[i] += np.array([x, y, z])

    return matrix

# rotates each point stored in matrix by z degrees around the z  - axis,
# y degrees about the y - axis, and x degrees about the x - axis, in that order
def rotateMolecule(matrix, x, y, z):
    rotation = Rotation.from_euler("zyx", [z, y, x], degrees = True)
    for i in range(len(matrix)):
        matrix[i] = rotation.apply(matrix[i])

    return matrix

# assumes both matrices have the same dimensions
# assumes both matrices are 3D
# assumes both matrices are perfect rectangular prisms
def combineMatrices(matrix1, matrix2):
    shape = matrix1.shape
    matrix3 = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                #Accounting for positive negative charged values
                if abs(matrix1[i][j][k]) > abs(matrix2[i][j][k]):
                    matrix3[i][j][k] = matrix1[i][j][k] #Appending value if greatest absolute value
                else:
                    matrix3[i][j][k] = matrix2[i][j][k] #Else appending matrix 2 value

    return matrix3

def maxDistances(matrix_l, buffer):
    data = matrix_l # Sample 2D array data
    #print(data.shape)
    #print(type(data))
    #print(data)
    # Extract x, y, z values-----
    x_values = data[:, 0]
    y_values = data[:, 1]
    z_values = data[:, 2]
    # Determining the maximum and minimum values along each of the axis accounting for set buffer
    max_x = np.max(x_values) + buffer
    min_x = np.min(x_values) - buffer
    max_y = np.max(y_values) + buffer
    min_y = np.min(y_values) - buffer
    max_z = np.max(z_values) + buffer
    min_z = np.min(z_values) - buffer
    # Returning max and min values
    max_l = [max_x, max_y, max_z]
    min_l  = [min_x, min_y, min_z]
    distances = [ max_x - min_x, max_y - min_y, max_z - min_z]
    # Maximum distance among x, y, z distances
    #max_distance = np.max(distances)
    return (max_l, min_l, distances)

def combineMatrices_l(matrixList):
    shape = matrixList[0].shape
    matrix3 = np.zeros(shape)
    
    #print('Counting matricies. . .')
    for matrix in matrixList:
            matrix3 = combineMatrices(matrix3, matrix)
            
    return matrix3

def dict_array_remap(mapping, inp_array):
    inp_shape = inp_array.shape
    inp_array = inp_array.flatten()
    out = [mapping[i] for i in tq(inp_array)]
    out = np.array(out)
    out = out.reshape(inp_shape)
    return out


#Embedding functions


covalent_atomic_radi = {'C':0.76, 'N':0.71, 'H':0.31, 'F':0.71, 'O':0.66,
                        'S':1.05, 'Cl':1.02, 'Br':1.2, 'I':1.39, 'P':1.07}
largest_radi = max(covalent_atomic_radi, key=covalent_atomic_radi.get)
largest_radi = covalent_atomic_radi[largest_radi] #Setting largest radi for buffer region in embedding and dot size

#Function for embedding sphere in np orgrid------------------------------
def create_bin_sphere(coords, center, r):
    distance = np.sqrt((coords[0] - center[0])**2 + (coords[1]-center[1])**2 + (coords[2]-center[2])**2) 
    return 1*(distance <= r)

#Visualization functions ----------------------------------
mass_color_dict = {
    0.0:'#00FFFFFF',
    1.008:'#d6d6d6',
    12.011:'#575757',
    14.007:'#5b8ef5',
    15.999:'#ff454b'
}
def vis_channel(channel, channel_ident):
    if channel_ident == 'mass':
        norm = plt.Normalize()
        values = list(np.unique(channel))

        mass_colors = dict_array_remap(mass_color_dict, channel)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection='3d')
        ax.voxels(channel, facecolors = mass_colors, edgecolor='none')
        # Hide grid lines
        ax.grid(False)

        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax._axis3don = False


    elif channel_ident == 'charge':
        charge_colors = cm.seismic(channel)
        
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection='3d')
        ax.voxels(channel, facecolors = charge_colors, edgecolor='none')
        # Hide grid lines
        ax.grid(False)

        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax._axis3don = False

        cbar = plt.colorbar(
            plt.cm.ScalarMappable(cmap=cm.seismic),
            ax=plt.gca()
        )

        cbar.ax.set_ylabel('Partial Charge', rotation=270, labelpad = 15, fontname = 'Arial')

    plt.show() 



def molecule_3d_embedding(molecule_list, volume_size_px, **kwargs): #, atom_radius_px, bond_radius_px
    #Defining nomalization parameters-----------------------------------------------------------------------------
    Unscaled_atom_positions = [] #Initializing unscaled atom positions list
    Partial_charges = [] #Initializing partial charges list
    total_atoms = None #initializing empty np array
    
    if kwargs.get('min_max_norm'): #If min_max_normalizatin==========================================
        for molecule_count, i in enumerate(molecule_list): #Looping through the list of molecules to format and calculate normalization parameters
            mol_2Darray = np.stack(i.Mol_cord) #Stacking to 2D array
            Unscaled_atom_positions.append(mol_2Darray) #Appending Unscaled mol cord to list as 2D array
            Partial_charges.append(i.partialcharge) #Appending partial charges similarly
            if molecule_count == 0: #If first molecule
                total_atoms = mol_2Darray #Concatenating to total atom array
            else:
                total_atoms = np.concatenate((total_atoms, mol_2Darray), axis = 0) #Stacking to total atom array
        
        _, min_l, distances = maxDistances(total_atoms, 1.39) #Calculating max, min, and distances for min/max normalization
        scalingFactor = max(distances) #Setting scaling factor as max value of total atom distances
        
        
    #Normalizinig molecules=========================================================================
    Normalized_molecules = []
    Embedded_molecules = []
    for i in Unscaled_atom_positions: #Looping thorugh molecules
        reshaped = np.transpose(i) #Transposing so that the dimensions are accesible
        Normalized = [] #Initializing normalized molecule list
        for count, j in enumerate(reshaped): #Minmax scaling by dimension
            normalized_dimension = (j-min_l[count])/(scalingFactor) #Min max normalization
            Normalized.append(normalized_dimension) #Appending dimension
        Normalized = np.array(Normalized) #Converting back to array
        Normalized = np.transpose(Normalized) #Retransposing array
        
        #Centrizing array (based on convex hull volume)-----------------------------------------------------------------------------------
        hull = ConvexHull(Normalized) #Create convex hull object
        cx = np.mean(hull.points[hull.vertices,0]) #Calculating centroid x coordinates from convex hull volume
        cy = np.mean(hull.points[hull.vertices,1]) #Calculating centroid y coordinates from convex hull volume
        cz = np.mean(hull.points[hull.vertices,2]) #Calculating centroid z coordinates from convex hull volume
        centroid_translation_matrix = [0.5-cx, 0.5-cy, 0.5-cz] #Creating list of centroid points
        Normalized = Normalized + centroid_translation_matrix #Centering molecule
        
        
        Normalized_molecules.append(Normalized) #Appending Normalized molecule to list
        Embedded_molecules.append(np.round(Normalized*volume_size_px)) #Embedding atom-center positions in volume by np array index
        
        
    

    #Scaling and rounding atomic radi--------------------------------------------------------------------------
    Scaled_radi = {k: int(round((v / scalingFactor)* volume_size_px)) for k, v in covalent_atomic_radi.items()} #Dividing all values by the scaling factor

    #Initializing 4d np array---------------------------------------------------------------------------
    molecule_w_channels = [] #Initializing list of molecules in volxel display
    tq._instances.clear()
    for molecule_counter, i in tq(enumerate(Embedded_molecules)): #Looping through molecules of the list
        temp_masschannel_atom_l = [] #Initializing atom list
        temp_chargechannel_atom_l = [] #Initializing atom list

        #/Initializng volume channel------------------------------------------------------------------------
        vol_channel = np.ogrid[:volume_size_px, :volume_size_px, :volume_size_px] #Intializing zero square array of size volume_size_pz
        #/Embedding atoms in volume------------------------------------------------------------------------------------------
        for atom_counter, j in enumerate(i): #Looping through atoms of molecule
            r = Scaled_radi[molecule_list[molecule_counter].Atom_ident[atom_counter]] #Pulling atomic radi from dictionary based on atom identity
            atom_matrix = create_bin_sphere(vol_channel, j, r) #Appending matricies to temp atom list
            
            atom_mass_matrix = atom_matrix*atomic_mass[molecule_list[molecule_counter].Atom_ident[atom_counter]] #Multiplying atom matrix by atomic mass from dictionary
            atom_charge_matrix = atom_matrix*Partial_charges[molecule_counter][atom_counter] #Multiplying atom matrix by atomic partial charge
            
            temp_masschannel_atom_l.append(atom_mass_matrix) #Appedning atom voxel mass to list
            temp_chargechannel_atom_l.append(atom_charge_matrix) #Appending atom voxel charge to list

        mass_channel = combineMatrices_l(temp_masschannel_atom_l) #Combining atom mass together for 1 np array
        charge_channel = combineMatrices_l(temp_chargechannel_atom_l) #Combining atom charge together for 1 np array
        
        molecule_w_channels.append(np.array([mass_channel, charge_channel])) #Appending molecule array
    
    return(molecule_w_channels) #Returning list of molecules



def molecule_3d_embedding_v2(molecule_list, atomic_mass, volume_size_px, batch_size, out_dir, **kwargs): #, atom_radius_px, bond_radius_px

    def chunk_h5_writing(data, batch_size, data_set_size, iter, out_dir):
        if iter == 0:
            with h5py.File(out_dir, 'w') as f:
                # create empty data set
                dset = f.create_dataset('embeds', shape=(data_set_size, data.shape[1], data.shape[2], data.shape[3], data.shape[4]),
                                        maxshape=(None, data.shape[1], data.shape[2], data.shape[3], data.shape[4]), 
                                        chunks=(batch_size, data.shape[1], data.shape[2], data.shape[3], data.shape[4]),
                                        dtype=np.float32, 
                                        compression = 'gzip')
                
                # add first batch of data
                dset[:batch_size, :, :, :, :] = data[:, :, :, :, :]
                
                # Create attribute with last_index value
                dset.attrs['last_index']=batch_size
                    
        else:
            # add more data
            with h5py.File(out_dir, 'a') as f: # USE APPEND MODE
                dset = f['embeds']
    
                start = dset.attrs['last_index']
                # add chunk of rows
                dset[start:start+batch_size, :, :, :, :] = data[:, :, :, :, :]
            
                # Resize the dataset to accommodate the next chunk of rows
                #dset.resize(26, axis=0)
                
                # Create attribute with last_index value
                dset.attrs['last_index']=start+batch_size
    
    def combineMatrices(matrix1, matrix2):
        shape = matrix1.shape
        matrix3 = np.zeros(shape)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    #Accounting for positive negative charged values
                    if abs(matrix1[i][j][k]) > abs(matrix2[i][j][k]):
                        matrix3[i][j][k] = matrix1[i][j][k] #Appending value if greatest absolute value
                    else:
                        matrix3[i][j][k] = matrix2[i][j][k] #Else appending matrix 2 value
    
        return matrix3
    
    def create_bin_sphere(coords, center, r):
        distance = np.sqrt((coords[0] - center[0])**2 + (coords[1]-center[1])**2 + (coords[2]-center[2])**2) 
        return 1*(distance <= r)
        
    def maxDistances(matrix_l, buffer):
        data = matrix_l # Sample 2D array data
        #print(data.shape)
        #print(type(data))
        #print(data)
        # Extract x, y, z values-----
        x_values = data[:, 0]
        y_values = data[:, 1]
        z_values = data[:, 2]
        # Determining the maximum and minimum values along each of the axis accounting for set buffer
        max_x = np.max(x_values) + buffer
        min_x = np.min(x_values) - buffer
        max_y = np.max(y_values) + buffer
        min_y = np.min(y_values) - buffer
        max_z = np.max(z_values) + buffer
        min_z = np.min(z_values) - buffer
        # Returning max and min values
        max_l = [max_x, max_y, max_z]
        min_l  = [min_x, min_y, min_z]
        distances = [ max_x - min_x, max_y - min_y, max_z - min_z]
        # Maximum distance among x, y, z distances
        #max_distance = np.max(distances)
        return (max_l, min_l, distances)
    
    def combineMatrices_l(matrixList):
        shape = matrixList[0].shape
        matrix3 = np.zeros(shape)
        
        #print('Counting matricies. . .')
        for matrix in matrixList:
                matrix3 = combineMatrices(matrix3, matrix)
                
        return matrix3
    
    def dict_array_remap(mapping, inp_array):
        inp_shape = inp_array.shape
        inp_array = inp_array.flatten()
        out = [mapping[i] for i in tq(inp_array)]
        out = np.array(out)
        out = out.reshape(inp_shape)
        return out

    
    #Defining nomalization parameters-----------------------------------------------------------------------------
    batch_iter = 0
    number_molecules = len(molecule_list)
    for molecule_list_batch in chunked(molecule_list, batch_size):
        Unscaled_atom_positions = [] #Initializing unscaled atom positions list
        Partial_charges = [] #Initializing partial charges list
        total_atoms = None #initializing empty np array
        
        if kwargs.get('min_max_norm'): #If min_max_normalizatin==========================================
            for molecule_count, i in enumerate(molecule_list_batch): #Looping through the list of molecules to format and calculate normalization parameters
                mol_2Darray = np.stack(i.Mol_cord) #Stacking to 2D array
                Unscaled_atom_positions.append(mol_2Darray) #Appending Unscaled mol cord to list as 2D array
                Partial_charges.append(i.partialcharge) #Appending partial charges similarly
                if molecule_count == 0: #If first molecule
                    total_atoms = mol_2Darray #Concatenating to total atom array
                else:
                    total_atoms = np.concatenate((total_atoms, mol_2Darray), axis = 0) #Stacking to total atom array
            
            _, min_l, distances = maxDistances(total_atoms, 1.39) #Calculating max, min, and distances for min/max normalization
            scalingFactor = max(distances) #Setting scaling factor as max value of total atom distances
            
            
        #Normalizinig molecules=========================================================================
        Normalized_molecules = []
        Embedded_molecules = []
        for i in Unscaled_atom_positions: #Looping thorugh molecules
            reshaped = np.transpose(i) #Transposing so that the dimensions are accesible
            Normalized = [] #Initializing normalized molecule list
            for count, j in enumerate(reshaped): #Minmax scaling by dimension
                normalized_dimension = (j-min_l[count])/(scalingFactor) #Min max normalization
                Normalized.append(normalized_dimension) #Appending dimension
            Normalized = np.array(Normalized) #Converting back to array
            Normalized = np.transpose(Normalized) #Retransposing array
            
            #Centrizing array (based on convex hull volume)-----------------------------------------------------------------------------------
            hull = ConvexHull(Normalized) #Create convex hull object
            cx = np.mean(hull.points[hull.vertices,0]) #Calculating centroid x coordinates from convex hull volume
            cy = np.mean(hull.points[hull.vertices,1]) #Calculating centroid y coordinates from convex hull volume
            cz = np.mean(hull.points[hull.vertices,2]) #Calculating centroid z coordinates from convex hull volume
            centroid_translation_matrix = [0.5-cx, 0.5-cy, 0.5-cz] #Creating list of centroid points
            Normalized = Normalized + centroid_translation_matrix #Centering molecule
            
            
            Normalized_molecules.append(Normalized) #Appending Normalized molecule to list
            Embedded_molecules.append(np.round(Normalized*volume_size_px)) #Embedding atom-center positions in volume by np array index
            
            
        
    
        #Scaling and rounding atomic radi--------------------------------------------------------------------------
        Scaled_radi = {k: int(round((v / scalingFactor)* volume_size_px)) for k, v in covalent_atomic_radi.items()} #Dividing all values by the scaling factor
    
        #Initializing 4d np array---------------------------------------------------------------------------
        molecule_w_channels = [] #Initializing list of molecules in volxel display
        tq._instances.clear()
        for molecule_counter, i in tq(enumerate(Embedded_molecules)): #Looping through molecules of the list
            temp_masschannel_atom_l = [] #Initializing atom list
            temp_chargechannel_atom_l = [] #Initializing atom list
    
            #/Initializng volume channel------------------------------------------------------------------------
            vol_channel = np.ogrid[:volume_size_px, :volume_size_px, :volume_size_px] #Intializing zero square array of size volume_size_pz
            #/Embedding atoms in volume------------------------------------------------------------------------------------------
            for atom_counter, j in enumerate(i): #Looping through atoms of molecule
                r = Scaled_radi[molecule_list_batch[molecule_counter].Atom_ident[atom_counter]] #Pulling atomic radi from dictionary based on atom identity
                atom_matrix = create_bin_sphere(vol_channel, j, r) #Appending matricies to temp atom list
                
                atom_mass_matrix = atom_matrix*atomic_mass[molecule_list_batch[molecule_counter].Atom_ident[atom_counter]] #Multiplying atom matrix by atomic mass from dictionary
                atom_charge_matrix = atom_matrix*Partial_charges[molecule_counter][atom_counter] #Multiplying atom matrix by atomic partial charge
                
                temp_masschannel_atom_l.append(atom_mass_matrix) #Appedning atom voxel mass to list
                temp_chargechannel_atom_l.append(atom_charge_matrix) #Appending atom voxel charge to list
    
            mass_channel = combineMatrices_l(temp_masschannel_atom_l) #Combining atom mass together for 1 np array
            charge_channel = combineMatrices_l(temp_chargechannel_atom_l) #Combining atom charge together for 1 np array
            
            molecule_w_channels.append(np.array([mass_channel, charge_channel])) #Appending molecule array
        
        molecule_w_channels = np.asarray(molecule_w_channels) #Converting to np array for storage
        chunk_h5_writing(molecule_w_channels, len(molecule_list_batch), number_molecules, batch_iter, out_dir) #Writing batch data
        batch_iter += 1

    #Check data
    if kwargs.get('check_shape'):
        with h5py.File(out_dir, 'r') as f:
            print(f['embeds'].attrs['last_index'])
            print(f['embeds'].shape)


            
