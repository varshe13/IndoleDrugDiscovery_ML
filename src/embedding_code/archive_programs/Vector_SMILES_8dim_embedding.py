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

def save_data(data_set, filename): #Saves to CSV with location
    data_set.to_csv("filename")
    print("Saving to:", os.path.abspath(filename))
    print(f'CSV saved as {filename}')


#Dataframe of SMILES and biological activity
raw_file_pickle = open('C:\\Users\\lizst\\OneDrive\\Research\\Indole_Computations\\030825_SMILES_testdata(1).pickle', mode='rb')
raw_data_pickle = pickle.load(raw_file_pickle)
raw_df_pickle = pd.DataFrame(raw_data_pickle)
raw_file_pickle.close()
#Alternative updated CSV input instead of pickles input
raw_data_updated = pd.read_csv("C:\\Users\\lizst\\OneDrive\\Research\\Indole_Computations\\updated_Raw_df.csv")

#raw_df is the universal input
raw_df = raw_data_updated



import codecs
from SmilesPE.tokenizer import *
#Dataframe of tokens
RAW_index = codecs.open("C:\\Users\\lizst\\OneDrive\\Research\\Indole_Computations\\Corrected_SPE_ChEMBL(1).txt")
spe = SPE_Tokenizer(RAW_index)
spe_vocab = codecs.open("C:\\Users\\lizst\\OneDrive\\Research\\Indole_Computations\\Corrected_SPE_ChEMBL(1).txt").read().split("\n")


#pulls list of SMILES and list of activity
raw_smiles = raw_df["SMILES"].to_list()
raw_smiles_df = raw_df["SMILES"]
raw_activity = raw_df["Gram-Positive Activity"]


#standardizes SMILES
empty_df2 = []
for x in raw_smiles:
    mol = Chem.MolFromSmiles(x)
    if mol == None:
        print(x)
    else:
        normalized_mol = rdMolStandardize.Normalize(mol)
        normalized_smile = Chem.MolToSmiles(normalized_mol)
        normalized_df = pd.DataFrame({"Normalized SMILES" : [normalized_smile]})
        empty_df2.append(normalized_df)

standardized_df = pd.concat(empty_df2, ignore_index=True)
standardized_smiles = standardized_df["Normalized SMILES"]



#Converts a SMILE into a list of tokens and merges into dataset
#.split() may be nescarry on the tokenizer for output to be list
empty_df = []
for x in standardized_smiles: 
    tokenizer = spe.tokenize(x).split()
    tokenization = pd.DataFrame({'Tokenized SMILES': [tokenizer]})  
    empty_df.append(tokenization)  
tokenized_df = pd.concat(empty_df, ignore_index=True)
output3 = pd.concat([raw_smiles_df, tokenized_df], axis=1, ignore_index=False, join='outer')
print(output3)




#defines smaller vocabulary using the tokens found in the dataset and places them into an index
token_strings = output3["Tokenized SMILES"]
token_series = output3["Tokenized SMILES"]
vocab = sorted(set(token for tokens in token_strings for token in tokens)) 
vocab_size = len(vocab)


index3 = dict(enumerate(vocab)) #assigns every value to single scalar
def switch_keys_values(my_dict):
    return {value: key for key, value in my_dict.items()}

index = switch_keys_values(index3)



#Modifies the tokens into ID and outputs every single ID in it's own column, molecules seperated by row
vector_scalar_list = []
for list1 in token_strings:
    for i in range(len(list1)):
     key = list1[i]
     if key in index:
        list1[i] = index[key]
    vector_scalar_list.append(list1)
vector_df = pd.DataFrame(vector_scalar_list)


#combines all data into a CSV file for storage and visualization
final_df = pd.concat([standardized_smiles, token_series, raw_activity, tokenized_df, vector_df],axis=1)
final_df.to_csv("finalfinal1.csv")


#converts all vector represenations to the same amount by adding 0(s)
max_len = max(len(seq) for seq in vector_scalar_list)  
padded_scalar_vectors = [seq + [0] * (max_len - len(seq)) for seq in vector_scalar_list] 



#embeds molecules into 8 dimensional vectors of type:tensor

#embedding_dim = 8  
#embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
#indexed_tensor = torch.tensor(padded_scalar_vectors, dtype=torch.long)
#embedded_tokens = embedding_layer(indexed_tensor)

# Gives (batch_size, sequence_length, embedding_dim) as well as the first molecule's vectors
#print("Token Embeddings Shape:", embedded_tokens.shape)
#print("Embeddings for first molecule:\n", embedded_tokens[0])


#Save tensor file numpy_array (.npy)
#embedded_tokens_array = embedded_tokens.detach().numpy()
#np.save("Embedded_tokens_array.npy", embedded_tokens_array)

#Save tensor file pytorch (.pt)
#torch.save(embedded_tokens, "Embedded_tokens.pt")


#embed as 16 dimension vector
embedding_dim = 16
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
indexed_tensor = torch.tensor(padded_scalar_vectors, dtype=torch.long)
embedded_tokens = embedding_layer(indexed_tensor)

print("Embeddings for first molecule:\n", embedded_tokens[0])
print("Token Embeddings Shape:", embedded_tokens.shape)

torch.save(embedded_tokens, "Embedded_tokens.pt")

embedded_tokens_array = embedded_tokens.detach().numpy()
np.save("Embedded_tokens_array.npy", embedded_tokens_array)

