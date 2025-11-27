import numpy as np
import pandas as pd
import os
import pickle
import rdkit
import SmilesPE
import torch
import torch.nn as nn
from SmilesPE.pretokenizer import atomwise_tokenizer
from rdkit import Chem
import torch
import torch.nn
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Draw import IPythonConsole


def switch_keys_values(my_dict):
    return {value: key for key, value in my_dict.items()}

def padd_vecs(vec):
    max_len = max(len(seq) for seq in vec)  
    padded_vec = [seq + [0] * (max_len - len(seq)) for seq in vec] 

def molecule_SMILES_embedding(raw_smiles, dims):

    #SMILE standardization and vocab generation
    standardized_smile = []
    vocab_tokens = []
    for x in raw_smiles:
        mol = Chem.MolFromSmiles(x)
        if mol == None:
            print(x)
        else:
            normalized_mol = rdMolStandardize.Normalize(mol)
            #Converts a SMILE into a list of tokens and merges into dataset
            tokenized_SMILE = spe.tokenize(Chem.MolToSmiles(normalized_mol)).split() #.split() may be nescarry on the tokenizer for output to be list
            for token in tokenized_SMILE:
                vocab_tokens.append(token)


    vocab_tokens = sorted(set(vocab_tokens)) #Sorted set of all tokens in dataset
    vocab_dict = switch_keys_values(dict(enumerate(vocab))) #Makes into dictionary with tokens

    #Generating smiles vector scalar list
    vector_scalar_list = []
    for list1 in token_strings:
        for i in range(len(list1)):
         key = list1[i]
         if key in index:
            list1[i] = index[key]
        vector_scalar_list.append(list1)
    vector_df = pd.DataFrame(vector_scalar_list)



    


    

    return embedded_smiles