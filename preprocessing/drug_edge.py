import numpy as np
from rdkit import Chem


def one_hot_encoding(x, permitted_list):
    enc = [int(x == p) for p in permitted_list]
    if sum(enc) == 0:
        enc[-1] = 1  # Unknown
    return enc


def get_atom_features(atom,
                      use_chirality=True,
                      hydrogens_implicit=True):
    # Basis-Features wie zuvor…
    permitted_atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'Si', 'Unknown']
    atom_type = one_hot_encoding(atom.GetSymbol(), permitted_atoms)
    degree = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, "MoreThanFour"])
    charge = one_hot_encoding(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybrid = one_hot_encoding(str(atom.GetHybridization()),
                              ["SP", "SP2", "SP3", "OTHER"])
    in_ring = [int(atom.IsInRing())]
    aromatic = [int(atom.GetIsAromatic())]
    mass = [(atom.GetMass() - 10.812) / 116.092]
    # H-Bindungsspender/-akzeptor
    is_donor = int(atom.GetSymbol() in ("N", "O") and atom.GetTotalNumHs() > 0)
    is_acceptor = int(atom.GetSymbol() in ("N", "O") and atom.GetTotalNumHs() < 3)
    hb_donor = [is_donor]
    hb_acceptor = [is_acceptor]
    vec = (atom_type + degree + charge + hybrid +
           in_ring + aromatic + mass +
           hb_donor + hb_acceptor)
    if use_chirality:
        chir = one_hot_encoding(str(atom.GetChiralTag()),
                                ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW",
                                 "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        vec += chir
    if hydrogens_implicit:
        hcount = one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, "MoreThanFour"])
        vec += hcount
    return np.array(vec, dtype=float)
