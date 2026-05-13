from __future__ import annotations
import json, time, os, sys, glob , tqdm 
import shutil
import jax.numpy as jnp 
import optax
from flax import nnx 


from jax import Array 
from jaxtyping import Array, Int, Float
# torch data load 
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

#extra utlities 
import copy 
import random 
import itertools 

# utility functions 
#TODO can it be improved? 
def parse_fasta(filename, limit=-1, omit=[]):
    header = []
    sequence = []
    lines = open(filename, "r")
    for line in lines:
        if line[0] == ">":
            if line[0] == limit:
                break
            header.append(line[1:])
            sequence.append([])
        else:
            if omit:
                line = [item for item in line if item not in omit]
                line = "".join(line)
            line = " ".join(line)
            sequence[-1].append(line)
    lines.close()
    sequence = [''.join(seq) for seq in sequence]
    return jnp.array(header), jnp.array(sequence)

def _scores(S: Int[Array, "B L"], 
            log_probs: Float[Array, "B L V"],
            mask: Float[Array, "B L"]) -> Float[Array "B"]:
    """Negative log probabilities"""
    # here we have to use a gather function instead of nnl 
    loss = - jnp.take_along_axis(
        log_probs, #[B, L, V]
        S[..., None], #[B, L, 1]
        axis=-1
    ).squeeze(-1) #[B, L]
    scores = jnp.sum(loss * mask, axis=-1) / jnp.sum(mask, axis=-1)
    return scores 

def _S_to_seq(S, mask):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    seq = ''.join([alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) if m>0])
    return seq

def parse_PDB_biounits(x, atoms=['N', 'CA', 'C'], chain=None):
    '''
        Input: x = PDB filename 
        atoms = atoms to extract (optional)
    output: (length, atoms, coords=(x, y, z)) sequence
    '''
    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    states = len(alpha_1)
    alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
             'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']
    aa_1_N = {a:n for n,a in enumerate(alpha_1)}
    aa_3_N = {a:n for n,a in enumerate(alpha_3)}
    aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
    aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
    aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}
    
    def AA_to_N(x):
        #["ANRD"] -> [[0, 1, 2, 3]]
        x = jnp.array(x)
        if x.ndim == 0: x = x[None]
        return [[aa_1_N.get(a, states-1)for  a in y] for y in x]
    
    def N_to_AA(x):
        # [[0, 1, 2, 3]] -> ["ARND"]
        x = jnp.array(x)
        if x.ndim == 1: x = x[None]
        return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]

    xyz, seq, min_resn, max_resn = {}, {}, 1e6, 1e-6
    for line in open(x, "rb"):
        line = line.decode("utf-8","ignore").rstrip()
        if line[:6] == "HETATM" and line[17:17+3] == "MSE":
            line = line.replace("HETATM","ATOM  ")
            line = line.replace("MSE","MET")
        if line[:4] == "ATOM":
            ch = line[21:22]
            if ch == chain or chain is None:
                atom = line[12:12+4].strip()
                resi = line[17:17+3]
                resn = line[22:22+5].strip()
                xc, yc, zc = [float(line[i:(i+8)]) for i in [30,38,46]]

                if resn[-1].isalpha(): 
                    resa,resn = resn[-1],int(resn[:-1])-1
                else: 
                    resa,resn = "",int(resn)-1
#         resn = int(resn)
                if resn < min_resn: 
                    min_resn = resn
                if resn > max_resn: 
                    max_resn = resn
                if resn not in xyz: 
                    xyz[resn] = {}
                if resa not in xyz[resn]: 
                    xyz[resn][resa] = {}
                if resn not in seq: 
                    seq[resn] = {}
                if resa not in seq[resn]: 
                    seq[resn][resa] = resi

                if atom not in xyz[resn][resa]:
                    xyz[resn][resa][atom] = jnp.array([xc, yc, zc])
    seq_, xyz_ = [], []
    try:
        for resn in range(min_resn, max_resn + 1):
            if resn in seq:
                for k in sorted(seq[resn]):
                    seq_.append(aa_3_N.get(seq[resn][k], 20))
            else:
                seq_.append(20)
            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_.append(xyz[resn][k][atom])
                        else:
                            xyz_.append(jnp.full(3, jnp.nan))
            else:
                for atom in atoms:
                    xyz_.append(jnp.full(3, jnp.nan))
        return jnp.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(jnp.array(seq_))
    except TypeError:
        return 'no_chain', 'no_chain'

def parse_PDB(path_to_pdb, input_chain_list=None, ca_only=False):
    c=0
    pdb_dict_list = []
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(jnp.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    if input_chain_list:
        chain_alphabet = input_chain_list

    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ''
        concat_N = []
        concat_CA = []
        concat_C = []
        concat_O = []
        concat_mask = []
        coords_dict = {}
        for letter in chain_alphabet:
            if ca_only:
                sidechain_atoms = ['CA']
            else:
                sidechain_atoms = ['N', 'CA', 'C', 'O']
            xyz, seq = parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter)
            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_'+letter]=seq[0]
                coords_dict_chain = {}
                if ca_only:
                    coords_dict_chain['CA_chain_'+letter]=xyz.tolist()
                else:
                    coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                    coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                    coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                    coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_'+letter]= coords_dict_chain
                s +=1

        fi = biounit.rfind("/")
        my_dict["name"]=biounit[(fi+1):-4]
        my_dict['num_of_chains']=s
        my_dict['seq'] = concat_seq
        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c+=1
        return pdb_dict_list

#TODO: this function can be splitted into multiples
def tied_featurize():
    '''
        Pack and pad batch into tensors
    '''
