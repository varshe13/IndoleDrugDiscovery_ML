'''
03/31/25- Generated file, created new dataset class, combining functions

'''

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
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle
import numpy_indexed as npi
from more_itertools import chunked
import h5py
import re
import math


p = AllChem.ETKDGv2()
p.verbose = False

###################
#Utility Functions#
###################

# Write h5 file in chuncks
def chunk_h5_writing(data, batch_size, data_set_size, iteration, out_dir):
    if iteration == 0:
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


# Resize h5 to meet written amount of data
def resize_h5(out_dir):
    """Trim an HDF5 dataset ('embeds') so its first dimension matches last_index."""
    with h5py.File(out_dir, 'a') as f:
        dset = f['embeds']

        # Number of rows actually written
        last_index = dset.attrs['last_index']

        # Current allocated size
        current_size = dset.shape[0]

        # Only resize if dataset is larger than needed
        if current_size != last_index:
            print(f"Resizing dataset from {current_size} → {last_index}")
            dset.resize((last_index,) + dset.shape[1:])
        else:
            print("Dataset size already matches last_index.")

        # Optional: store the final correct size for reference
        dset.attrs['final_size'] = last_index
        
# Parse antibiotic activity based on threshold
def parse_antibiotic_activity(smile, antibiotic_activity, activity_type, activity_threshold):
    if np.isnan(antibiotic_activity):
        return np.nan
    else:
        if activity_type == 'MIC (uM)':
            uM_activity = antibiotic_activity
    
        if activity_type == 'MIC (ug/mL)':
            try:
                uM_activity = antibiotic_activity*(1000/Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(smile)))
            except:
                return np.nan

        if uM_activity <= activity_threshold:
            return 1
        else:
            return 0

# Parse dataframe based on input source and actiivity threshold
def parse_df(inp_df, activity_threshold, source):
    molecule_l = []
    print(source)
    if source == 'Experimental Values':
        for _, row in tq(inp_df.iterrows()):
            molecule_l.append(molecule(Chem.AddHs(Chem.MolFromSmiles(row['SMILES'])),
                                            row['GP Activity'],
                                            row['GN Activity'],
                                            mol_source = source))

    elif isinstance(source, str) and source != 'Experimental Values':
        for _, row in tq(inp_df.iterrows()):
            molecule_l.append(molecule(Chem.AddHs(Chem.MolFromSmiles(row['SMILES'])),
                                parse_antibiotic_activity(row['SMILES'], row['GP Activity'], row['Activity Type'], activity_threshold),
                                parse_antibiotic_activity(row['SMILES'], row['GN Activity'], row['Activity Type'], activity_threshold),
                                mol_source = source))

    else:
        for _, row in tq(inp_df.iterrows()):
            molecule_l.append(molecule(Chem.AddHs(Chem.MolFromSmiles(row['SMILES'])),
                                parse_antibiotic_activity(row['SMILES'], row['GP Activity'], 'MIC (uM)', activity_threshold),
                                parse_antibiotic_activity(row['SMILES'], row['GN Activity'], 'MIC (uM)', activity_threshold),
                                mol_source = source))

    return molecule_l
                            
    



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





#Visualization functions ----------------------------------
mass_color_dict = {
    0.0:'#00FFFFFF',
    1.008:'#d6d6d6',
    12.011:'#575757',
    14.007:'#5b8ef5',
    15.999:'#ff454b',
    32.067:'#d9d168',
    35.453:'#78cc7f'
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
        #norm = plt.Normalize(vmin = np.min(channel), vmax = np.max(channel))
        #charge_colors = cm.seismic(norm(channel))
        #print(cmapcolor)
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

        #cmap = mpl.cm.seismic
        norm = mpl.colors.Normalize(vmin=np.min(channel), vmax=np.max(channel))

        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cm.seismic),
             cax=ax, orientation='vertical', label='Some Units')

        '''
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(cmap=cm.seismic),
            ax=plt.gca()
        )

        cbar.ax.set_ylabel('Partial Charge', rotation=270, labelpad = 15, fontname = 'Arial')
        '''

    plt.show()



################################
#3D Embedding Related Functions#
################################
# Calculating Scaling Factor and Dataset Dimension Size (1, 2)
def calculate_scaling_factor(rdkit_mol_list, H_atom_diameter):
    periodicTable = Chem.GetPeriodicTable()
    largest_rad = 0
    raw_scaling_factor = 0
    count = 0
    embeddings_len = 0
    print('Calculating Scaling Factor. . .')
    for mol in tq(rdkit_mol_list):
        #Determining Largest Radius
        mol_largest_rad = np.max([periodicTable.GetRcovalent(atom.GetSymbol()) for atom in mol.rdkit_mol.GetAtoms()])

        #Determining Raw Scaling Factor
        for conformer in mol.rdkit_mol.GetConformers():
            temp_scaling_factor = maxDistances(conformer.GetPositions())
            if temp_scaling_factor > raw_scaling_factor:
                raw_scaling_factor = temp_scaling_factor
                largest_mol_idx = count
            embeddings_len += 1
        
        if mol_largest_rad > largest_rad:
            largest_rad = mol_largest_rad
        count += 1

    #Calculating factors
    adj_scaling_factor = raw_scaling_factor+(2*largest_rad)
    dimension_size = int(math.ceil(H_atom_diameter*(adj_scaling_factor/(2*periodicTable.GetRcovalent(1)))))
    
    return adj_scaling_factor, dimension_size, largest_mol_idx, embeddings_len
    

# Scaling Molecule Positions (3)
def scale_molecule_positions(mol_array, scaling_factor, volume_size_px):
    Normalized = []
    reshaped = np.transpose(mol_array) #Transposing so that the dimensions are accesible
    for dimension in reshaped: #Minmax scaling by dimension
        normalized_dimension = (dimension-np.min(dimension))/scaling_factor #Min max normalization
        Normalized.append(normalized_dimension) #Appending dimension
    Normalized = np.array(Normalized) #Converting back to array
    Normalized = np.transpose(Normalized) #Retransposing array
    
    #Centrizing array (based on convex hull volume)-----
    hull = ConvexHull(Normalized) #Create convex hull object
    cx = np.mean(hull.points[hull.vertices,0]) #Calculating centroid x coordinates from convex hull volume
    cy = np.mean(hull.points[hull.vertices,1]) #Calculating centroid y coordinates from convex hull volume
    cz = np.mean(hull.points[hull.vertices,2]) #Calculating centroid z coordinates from convex hull volume
    centroid_translation_matrix = [0.5-cx, 0.5-cy, 0.5-cz] #Creating list of centroid points
    Normalized = Normalized + centroid_translation_matrix #Centering molecule
    Normalized = np.round(Normalized*volume_size_px)
    
    #Embedded_molecules.append(np.round(Normalized*volume_size_px)) #Embedding atom-center positions in volume by np array index

    return Normalized
    

# Embedd Atoms position wise (4)
def embedd_molecule(rdkit_mol, scaled_pos, dimension_size, scaling_factor):
    periodicTable = Chem.GetPeriodicTable()
    vol_channel = np.ogrid[:dimension_size, :dimension_size, :dimension_size] #Intializing zero square array of size
    for count, atom in enumerate(rdkit_mol.GetAtoms()):
        scaled_atom_radi = dimension_size*(periodicTable.GetRcovalent(atom.GetSymbol())/scaling_factor)
        atom_matrix = create_bin_sphere(vol_channel, scaled_pos[atom.GetIdx()], scaled_atom_radi)
        
        atom_mass_matrix = atom_matrix*atom.GetMass()
        atom_charge_matrix = atom_matrix*float(atom.GetProp('_GasteigerCharge'))

        if count == 0:
            mass_channel = atom_mass_matrix
            charge_channel = atom_charge_matrix
        else:
            mass_channel = combineMatrices_v2(mass_channel, atom_mass_matrix)
            charge_channel = combineMatrices_v2(charge_channel, atom_charge_matrix)
            
        
    molecule_embedding = np.array([mass_channel, charge_channel])
    return molecule_embedding


#Function for embedding sphere in np orgrid
def create_bin_sphere(coords, center, r):
    distance = np.sqrt((coords[0] - center[0])**2 + (coords[1]-center[1])**2 + (coords[2]-center[2])**2) 
    return 1*(distance <= r)


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
def combineMatrices_v2(matrix1, matrix2):
    matrix3 = np.where(matrix1 != 0, matrix1, matrix2)
    return matrix3


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

# Calculates maxium distanced in matrix list
def maxDistances(molecule_matrix):
    # Extract x, y, z values-----
    x_values = molecule_matrix[:, 0]
    y_values = molecule_matrix[:, 1]
    z_values = molecule_matrix[:, 2]
    # Determining the maximum and minimum values along each of the axis accounting for set buffer
    max_x = np.max(x_values)
    min_x = np.min(x_values)
    max_y = np.max(y_values)
    min_y = np.min(y_values)
    max_z = np.max(z_values)
    min_z = np.min(z_values)
    # Returning max and min values
    max_l = [max_x, max_y, max_z]
    min_l  = [min_x, min_y, min_z]
    distances = [ max_x - min_x, max_y - min_y, max_z - min_z]
    # Maximum distance among x, y, z distances
    max_distance = np.max(distances)
    return (max_distance)



################
#Molecule Class#
################

class molecule:
    def __init__(self, rdkit_mol, gp_activity, gn_activity, mol_source = np.nan):     #Takes in the Molecules and Attributes from the Parse DB function
        self.rdkit_mol = rdkit_mol
        self.gp_activity = gp_activity
        self.gn_activity = gn_activity
        self.mol_source = mol_source
        self.channel_embedding = np.nan
        self.dataset_source = None
    
    def add_conformers(self, confs, rmsd_threshold, optimization_method = 'MMFF', maxIters = 5000):
        AllChem.EmbedMultipleConfs(self.rdkit_mol, confs, pruneRmsThresh=rmsd_threshold) #Note will prune to less than desired number of conformers

        if optimization_method == 'MMFF':
            self.conf_Es = AllChem.MMFFOptimizeMoleculeConfs(self.rdkit_mol, maxIters = maxIters)

    def save_embeddings(self, embeddings_array):
        self.channel_embedding = embeddings_array

    def vis_embedding(self, channel_idx, conf_id, style):
        vis_channel(self.channel_embedding[conf_id][channel_idx], style) 



         
    

###############
#Dataset Class#
###############
class dataset:
    def __init__(self, inp_table, activity_threshold = 100):
        self.compounds = []

        if inp_table is None:
            pass

        elif os.path.exists(inp_table):
            if inp_table.endswith('.xlsx'):
                with pd.ExcelFile(inp_table) as f: #Opening database
                    sheets = f.sheet_names
                    for sht in sheets:
                        temp_df = f.parse(sht)
                        self.compounds.extend(parse_df(temp_df, activity_threshold, sht))

            elif inp_table.endswith('.csv'):
                self.compounds.extend(parse_df(pd.read_csv(inp_table), activity_threshold, None))
                
            else:
                raise ValueError("Path not .csv or .xlsx")


        elif isinstance(inp_table, pd.DataFrame):
            self.compounds.extend(parse_df(inp_table, activity_threshold, 'Input DataFrame'))

        else:
            raise ValueError("Input must be pandas dataframe, .csv, or .xlsx")
        

    
    def generate_embedded_conformers(self, confs = 10, rmsd_threshold = 0, optimization_method = 'MMFF', remove_unembedded = True):
        mols_wo_conformers = []
        for count, mol in tq(enumerate(self.compounds)):
            try:
                mol.add_conformers(confs = confs, rmsd_threshold = rmsd_threshold, optimization_method = optimization_method)
            except:
                mols_wo_conformers.append(count)

        if remove_unembedded:
            mols_wo_conformers.sort(reverse=True)
            for index in mols_wo_conformers:
                del self.compounds[index]
                
                


    def generate_voxel_embedding(self, out_dir, H_atom_diameter = 3, batch_size = 10, min_max_norm = True, check_shape = True, return_channels = False, verbose = False):
        self.adj_scaling_factor, self.dimension_size, self.largest_mol_idx, self.num_embeddings = calculate_scaling_factor(self.compounds, H_atom_diameter)
        if verbose:
            print(f'Scaling factor: {self.adj_scaling_factor}')
            print(f'Dimension size: {self.dimension_size}')
            print(f'Largest mol index: {self.largest_mol_idx}')

        self.non_embeded_mols = []
        batch_iter = 0
        total_molecules_count = 0
        for batch in tq(chunked(self.compounds, batch_size)):
            batch_molecules_count = 0
            skipped_molecules_count = 0
            batch_embeddings = []
            for molecule in batch:
                molecule.rdkit_mol.ComputeGasteigerCharges() #NOTE, may need to calculate partial charges for hydrogens
    
                if return_channels:
                    molecules_embeddings = []
                
                for conformer in molecule.rdkit_mol.GetConformers():
                    '''
                    #Can adapt EHT methods for calculating static/partial charges
                    #Note that GasteigerCharges are conformer agnostic
                    _, res = rdEHTTools.RunMol(molecule.rdkit_mol, confId=conformer.GetId()) #Running molecule through EHT
                    p_charges = res.GetAtomicCharges() #Extracting static charges
                    '''
                    try:
                        scaled_position = scale_molecule_positions(conformer.GetPositions(), self.adj_scaling_factor, self.dimension_size)
                        embedding = embedd_molecule(molecule.rdkit_mol, scaled_position, self.dimension_size, self.adj_scaling_factor)
                        batch_embeddings.append(np.array(embedding))
                        
                    except:
                        if verbose:
                            print(f'Unable to embed conformer molecule number: {total_molecules_count}. May be due to error in convex hull formation coordinates. Skipping Embedding.')
                        self.non_embeded_mols.append(total_molecules_count)
                        skipped_molecules_count += 1
                        
                    total_molecules_count += 1
                    batch_molecules_count += 1
                    if return_channels:
                        molecules_embeddings.append(embedding)
    
                if return_channels:
                    molecule.save_embeddings(molecules_embeddings)
                
            batch_embeddings = np.array(batch_embeddings)
            chunk_h5_writing(batch_embeddings, batch_embeddings.shape[0], self.num_embeddings, batch_iter, out_dir) #Writing batch data
            batch_iter += 1

        resize_h5(out_dir) #Resizing h5 to match adjusted size after molecules were removed

        if check_shape:
            with h5py.File(out_dir, 'r') as f:
                print(f['embeds'].attrs['last_index'])
                print(f['embeds'].shape)

    def generate_out_df(self):
        molecule_id_l = []
        conf_id_l = []
        GP_activity_l = []
        GN_activity_l = []
        dataset_source_l = []
        

        molecule_id = 0
        for molecule in tq(self.compounds):
            for conformer in molecule.rdkit_mol.GetConformers():
                molecule_id_l.append(molecule_id)
                conf_id_l.append(conformer.GetId())
                GP_activity_l.append(molecule.gp_activity)
                GN_activity_l.append(molecule.gn_activity)
                dataset_source_l.append(molecule.dataset_source)

            molecule_id+=1

        out_df = pd.DataFrame()
        out_df['Molecule ID'] = molecule_id_l
        out_df['Conformer ID'] = conf_id_l
        out_df['GP Activity'] = GP_activity_l
        out_df['GN Activity'] = GN_activity_l
        out_df['Dataset Source'] = dataset_source_l

        #Removing non embedded molecules
        out_df = out_df.drop(self.non_embeded_mols)
        out_df = out_df.reset_index(drop=True)
        
        return out_df
            

# Merge two compound datasets
def merge_dataset(dataset_1, dataset_2, dataset_id_list = ['dataset_1', 'dataset_2']):
    outdataset = dataset(None)
    for compound_d1 in tq(dataset_1.compounds):
        compound_d1.dataset_source = dataset_id_list[0]
        outdataset.compounds.append(compound_d1)

    for compound_d2 in tq(dataset_2.compounds):
        compound_d2.dataset_source = dataset_id_list[1]
        outdataset.compounds.append(compound_d2)


    return outdataset

                

        
    
        

    
        