from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import jax.numpy as jnp

from jax import Array
from jaxtyping import Array, Int, Float

import itertools

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
N_AA = len(ALPHABET)  # 21


@dataclass
class TiedFeaturizeOutput:
    X: jnp.ndarray
    S: jnp.ndarray
    mask: jnp.ndarray
    lengths: jnp.ndarray
    chain_M: jnp.ndarray
    chain_encoding_all: jnp.ndarray
    letter_list_list: list[list[str]]
    visible_list_list: list[list[str]]
    masked_list_list: list[list[str]]
    masked_chain_length_list_list: list[list[int]]
    chain_M_pos: jnp.ndarray
    omit_AA_mask: jnp.ndarray
    residue_idx: jnp.ndarray
    dihedral_mask: jnp.ndarray
    tied_pos_list_of_lists_list: list[list[list[int]]]
    pssm_coef_all: jnp.ndarray
    pssm_bias_all: jnp.ndarray
    pssm_log_odds_all: jnp.ndarray
    bias_by_res_all: jnp.ndarray
    tied_beta_all: jnp.ndarray

    def as_tuple(self) -> tuple[Any, ...]:
        return (
            self.X,
            self.S,
            self.mask,
            self.lengths,
            self.chain_M,
            self.chain_encoding_all,
            self.letter_list_list,
            self.visible_list_list,
            self.masked_list_list,
            self.masked_chain_length_list_list,
            self.chain_M_pos,
            self.omit_AA_mask,
            self.residue_idx,
            self.dihedral_mask,
            self.tied_pos_list_of_lists_list,
            self.pssm_coef_all,
            self.pssm_bias_all,
            self.pssm_log_odds_all,
            self.bias_by_res_all,
            self.tied_beta_all,
        )

    def __iter__(self):
        return iter(self.as_tuple())

    def __getitem__(self, idx):
        return self.as_tuple()[idx]

    def __len__(self) -> int:
        return 20

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
            mask: Float[Array, "B L"]) -> Float[Array, "B"]:
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
    alphabet = 'ACDEFGHIKLMjnpQRSTVWYX'
    seq = ''.join([alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) if m>0])
    return seq

def parse_PDB_biounits(x, atoms=['N', 'CA', 'C'], chain=None):
    '''
        Ijnput: x = PDB filename 
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

def parse_PDB(path_to_pdb, ijnput_chain_list=None, ca_only=False):
    c=0
    pdb_dict_list = []
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(jnp.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    if ijnput_chain_list:
        chain_alphabet = ijnput_chain_list

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


# ---------------------------------------------------------------------------
# tied_featurize and helpers
#
# Design: all per-sample bookkeeping happens in NumPy (mutable buffers, easy
# slice assignment). We only cast to jnp at the very end, for the tensors that
# actually feed the model. JAX places arrays on accelerators implicitly, so
# there is no `device=` argument anywhere — it's a no-op in JAX-land.
# ---------------------------------------------------------------------------


def _chain_lists(b, chain_dict):
    """Return (masked_chains, visible_chains, all_chains) sorted alphabetically.

    `chain_dict[name] = (masked, visible)` when provided; otherwise every chain
    present in `b` is treated as masked.
    """
    if chain_dict is not None:
        masked, visible = chain_dict[b["name"]]
    else:
        masked = [k[-1:] for k in b if k.startswith("seq_chain_")]
        visible = []
    masked = sorted(masked)
    visible = sorted(visible)
    return masked, visible, masked + visible


def _chain_coords(coords, letter, ca_only):
    """Stack backbone atoms for a single chain into [L_chain, n_atoms, 3]."""
    if ca_only:
        x = np.asarray(coords[f"CA_chain_{letter}"], dtype=np.float32)
        if x.ndim == 2:
            x = x[:, None, :]
        return x  # [L_chain, 1, 3]
    return np.stack(
        [np.asarray(coords[f"{a}_chain_{letter}"], dtype=np.float32)
         for a in ("N", "CA", "C", "O")],
        axis=1,
    )  # [L_chain, 4, 3]


def _fixed_pos_mask(L_c, b, letter, fixed_position_dict):
    """1.0 = position is free to be redesigned; 0.0 = fixed to wild-type AA."""
    m = np.ones(L_c, dtype=np.float32)
    if fixed_position_dict is not None:
        fixed = fixed_position_dict[b["name"]][letter]
        if fixed:
            m[np.asarray(fixed) - 1] = 0.0  # 1-indexed → 0-indexed
    return m


def _omit_aa_mask(L_c, b, letter, omit_AA_dict):
    """Per-position {0,1} matrix; a 1 forbids that AA at that position."""
    m = np.zeros((L_c, N_AA), dtype=np.int32)
    if omit_AA_dict is not None:
        for pos_list, aa_letters in omit_AA_dict[b["name"]][letter]:
            pos_idx = np.asarray(pos_list, dtype=np.int32) - 1
            aa_idx = np.asarray([ALPHABET.index(a) for a in aa_letters],
                                dtype=np.int32)
            m[np.ix_(pos_idx, aa_idx)] = 1
    return m


def _pssm(L_c, b, letter, pssm_dict):
    """PSSM tensors. Defaults: coef=0, bias=0, log_odds=+1e4 (no constraint)."""
    coef = np.zeros(L_c, dtype=np.float32)
    bias = np.zeros((L_c, 21), dtype=np.float32)
    log_odds = 1e4 * np.ones((L_c, 21), dtype=np.float32)
    if pssm_dict and pssm_dict[b["name"]][letter]:
        entry = pssm_dict[b["name"]][letter]
        coef = np.asarray(entry["pssm_coef"], dtype=np.float32)
        bias = np.asarray(entry["pssm_bias"], dtype=np.float32)
        log_odds = np.asarray(entry["pssm_log_odds"], dtype=np.float32)
    return coef, bias, log_odds


def _tied_positions(b, tied_positions_dict, letter_list, global_starts, L_max):
    """Resolve symmetry constraints into flat global indices + per-position weights.

    Returns:
        tied_lists: list of lists; each inner list groups global indices that
            must be decoded to the same AA (e.g. for homomers).
        tied_beta: [L_max] float — relative weighting of each tied position
            within its group (1.0 by default).
    """
    tied_beta = np.ones(L_max, dtype=np.float32)
    tied_lists = []
    if tied_positions_dict is None:
        return tied_lists, tied_beta
    tied_pos_list = tied_positions_dict[b["name"]]
    if not tied_pos_list:
        return tied_lists, tied_beta
    letter_arr = np.asarray(letter_list)
    for tied_item in tied_pos_list:
        flat = []
        for k, v in tied_item.items():
            start = global_starts[int(np.argwhere(letter_arr == k)[0][0])]
            if isinstance(v[0], list):  # (positions, weights)
                positions, weights = v
                for p, w in zip(positions, weights):
                    flat.append(start + p - 1)
                    tied_beta[start + p - 1] = w
            else:
                for p in v:
                    flat.append(start + p - 1)
        tied_lists.append(flat)
    return tied_lists, tied_beta


def _dihedral_mask(residue_idx):
    """φ/ψ/ω validity from residue-index gaps. A dihedral is only well-defined
    when the neighbouring residues are sequentially adjacent in the PDB."""
    jumps = ((residue_idx[:, 1:] - residue_idx[:, :-1]) == 1).astype(np.float32)
    phi = np.pad(jumps, [(0, 0), (1, 0)])    # needs (i-1, i)
    psi = np.pad(jumps, [(0, 0), (0, 1)])    # needs (i, i+1)
    omega = np.pad(jumps, [(0, 0), (0, 1)])  # needs (i, i+1)
    return np.stack([phi, psi, omega], axis=-1)  # [B, L, 3]


def tied_featurize(
    batch,
    chain_dict,
    fixed_position_dict=None,
    omit_AA_dict=None,
    tied_positions_dict=None,
    pssm_dict=None,
    bias_by_res_dict=None,
    ca_only=False,
):
    """Pack a list of parsed-PDB dicts into padded batch arrays for ProteinMPNN.

    Returns a `TiedFeaturizeOutput` dataclass; it preserves legacy tuple-style
    unpacking/indexing via `__iter__` and `__getitem__`. Shapes
    use B = batch size, L = L_max = longest concatenated chain in the batch.
    """
    B = len(batch)
    lengths = np.array([len(b["seq"]) for b in batch], dtype=np.int32)
    L_max = int(lengths.max())
    n_atoms = 1 if ca_only else 4

    # Per-batch padded buffers (numpy; mutable).
    X = np.zeros((B, L_max, n_atoms, 3), dtype=np.float32)
    S = np.zeros((B, L_max), dtype=np.int32)
    chain_M = np.zeros((B, L_max), dtype=np.float32)
    chain_M_pos = np.zeros((B, L_max), dtype=np.float32)
    chain_encoding_all = np.zeros((B, L_max), dtype=np.int32)
    residue_idx = -100 * np.ones((B, L_max), dtype=np.int32)
    omit_AA_mask = np.zeros((B, L_max, N_AA), dtype=np.int32)
    pssm_coef_all = np.zeros((B, L_max), dtype=np.float32)
    pssm_bias_all = np.zeros((B, L_max, 21), dtype=np.float32)
    pssm_log_odds_all = 1e4 * np.ones((B, L_max, 21), dtype=np.float32)
    bias_by_res_all = np.zeros((B, L_max, 21), dtype=np.float32)
    tied_beta_all = np.ones((B, L_max), dtype=np.float32)

    letter_list_list, visible_list_list, masked_list_list = [], [], []
    masked_chain_length_list_list = []
    tied_pos_list_of_lists_list = []

    for i, b in enumerate(batch):
        masked, visible, all_chains = _chain_lists(b, chain_dict)

        # Per-chain buffers, concatenated below.
        xs, ms, seqs, encs = [], [], [], []
        fixed, omit, bias_res = [], [], []
        pcoef, pbias, plo = [], [], []
        letter_list, visible_list, masked_list, masked_lengths = [], [], [], []
        global_starts = [0]

        c, l0 = 1, 0
        for letter in all_chains:
            is_masked = letter in masked
            chain_seq = b[f"seq_chain_{letter}"].replace("-", "X")
            L_c = len(chain_seq)
            coords = b[f"coords_chain_{letter}"]

            xs.append(_chain_coords(coords, letter, ca_only))
            ms.append(np.ones(L_c, dtype=np.float32) if is_masked
                      else np.zeros(L_c, dtype=np.float32))
            seqs.append(chain_seq)
            encs.append(c * np.ones(L_c, dtype=np.int32))

            # residue_idx: +100 jump between chains so positional encoding sees a "break".
            residue_idx[i, l0:l0 + L_c] = 100 * (c - 1) + np.arange(L_c)
            global_starts.append(global_starts[-1] + L_c)

            # Constraints only apply to masked (designable) chains.
            fixed.append(
                _fixed_pos_mask(L_c, b, letter,
                                fixed_position_dict if is_masked else None))
            omit.append(
                _omit_aa_mask(L_c, b, letter,
                              omit_AA_dict if is_masked else None))
            cc, bb, lo = _pssm(L_c, b, letter,
                               pssm_dict if is_masked else None)
            pcoef.append(cc); pbias.append(bb); plo.append(lo)

            if is_masked and bias_by_res_dict:
                bias_res.append(np.asarray(
                    bias_by_res_dict[b["name"]][letter], dtype=np.float32))
            else:
                bias_res.append(np.zeros((L_c, 21), dtype=np.float32))

            letter_list.append(letter)
            (masked_list if is_masked else visible_list).append(letter)
            if is_masked:
                masked_lengths.append(L_c)
            c += 1
            l0 += L_c

        # Concatenate chains → one row of the padded batch.
        x = np.concatenate(xs, 0)
        all_seq = "".join(seqs)
        L = len(all_seq)
        X[i, :L] = x
        chain_M[i, :L] = np.concatenate(ms, 0)
        chain_M_pos[i, :L] = np.concatenate(fixed, 0)
        chain_encoding_all[i, :L] = np.concatenate(encs, 0)
        omit_AA_mask[i, :L] = np.concatenate(omit, 0)
        pssm_coef_all[i, :L] = np.concatenate(pcoef, 0)
        pssm_bias_all[i, :L] = np.concatenate(pbias, 0)
        pssm_log_odds_all[i, :L] = np.concatenate(plo, 0)
        bias_by_res_all[i, :L] = np.concatenate(bias_res, 0)
        S[i, :L] = np.asarray([ALPHABET.index(a) for a in all_seq],
                              dtype=np.int32)

        tied_lists, tied_beta = _tied_positions(
            b, tied_positions_dict, letter_list, global_starts, L_max)
        tied_pos_list_of_lists_list.append(tied_lists)
        tied_beta_all[i] = tied_beta

        letter_list_list.append(letter_list)
        visible_list_list.append(visible_list)
        masked_list_list.append(masked_list)
        masked_chain_length_list_list.append(masked_lengths)

    # mask = 1 where every atom in the residue is finite (no missing coords).
    mask = np.isfinite(X.sum(axis=(2, 3))).astype(np.float32)
    X = np.where(np.isnan(X), 0.0, X)

    dihedral_mask = _dihedral_mask(residue_idx)

    X_out = X[:, :, 0] if ca_only else X

    # Final cast: numpy → jnp. JAX handles device placement implicitly.
    return TiedFeaturizeOutput(
        X=jnp.asarray(X_out, dtype=jnp.float32),
        S=jnp.asarray(S, dtype=jnp.int32),
        mask=jnp.asarray(mask, dtype=jnp.float32),
        lengths=jnp.asarray(lengths, dtype=jnp.int32),
        chain_M=jnp.asarray(chain_M, dtype=jnp.float32),
        chain_encoding_all=jnp.asarray(chain_encoding_all, dtype=jnp.int32),
        letter_list_list=letter_list_list,
        visible_list_list=visible_list_list,
        masked_list_list=masked_list_list,
        masked_chain_length_list_list=masked_chain_length_list_list,
        chain_M_pos=jnp.asarray(chain_M_pos, dtype=jnp.float32),
        omit_AA_mask=jnp.asarray(omit_AA_mask, dtype=jnp.int32),
        residue_idx=jnp.asarray(residue_idx, dtype=jnp.int32),
        dihedral_mask=jnp.asarray(dihedral_mask, dtype=jnp.float32),
        tied_pos_list_of_lists_list=tied_pos_list_of_lists_list,
        pssm_coef_all=jnp.asarray(pssm_coef_all, dtype=jnp.float32),
        pssm_bias_all=jnp.asarray(pssm_bias_all, dtype=jnp.float32),
        pssm_log_odds_all=jnp.asarray(pssm_log_odds_all, dtype=jnp.float32),
        bias_by_res_all=jnp.asarray(bias_by_res_all, dtype=jnp.float32),
        tied_beta_all=jnp.asarray(tied_beta_all, dtype=jnp.float32),
    )


