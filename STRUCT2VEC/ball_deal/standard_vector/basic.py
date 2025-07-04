# standard elements (sorted by aboundance) (32)
std_elements = np.array([
    'C', 'O', 'N', 'S', 'P', 'Se', 'Mg', 'Cl', 'Zn', 'Fe', 'Ca', 'Na',
    'F', 'Mn', 'I', 'K', 'Br', 'Cu', 'Cd', 'Ni', 'Co', 'Sr', 'Hg', 'W',
    'As', 'B', 'Mo', 'Ba', 'Pt'
])

# standard residue names: AA/RNA/DNA (sorted by aboundance) (29)
std_resnames = np.array([
    'LEU', 'GLU', 'ARG', 'LYS', 'VAL', 'ILE', 'PHE', 'ASP', 'TYR',
    'ALA', 'THR', 'SER', 'GLN', 'ASN', 'PRO', 'GLY', 'HIS', 'TRP',
    'MET', 'CYS', 'G', 'A', 'C', 'U', 'DG', 'DA', 'DT', 'DC'
])

# standard atom names contained in standard residues (sorted by aboundance) (63)
std_names = np.array([
    'CA', 'N', 'C', 'O', 'CB', 'CG', 'CD2', 'CD1', 'CG1', 'CG2', 'CD',
    'OE1', 'OE2', 'OG', 'OG1', 'OD1', 'OD2', 'CE', 'NZ', 'NE', 'CZ',
    'NH2', 'NH1', 'ND2', 'CE2', 'CE1', 'NE2', 'OH', 'ND1', 'SD', 'SG',
    'NE1', 'CE3', 'CZ3', 'CZ2', 'CH2', 'P', "C3'", "C4'", "O3'", "C5'",
    "O5'", "O4'", "C1'", "C2'", "O2'", 'OP1', 'OP2', 'N9', 'N2', 'O6',
    'N7', 'C8', 'N1', 'N3', 'C2', 'C4', 'C6', 'C5', 'N6', 'N4', 'O2',
    'O4'
])

# backbone
std_backbone = np.array([
    'CA', 'N', 'C', 'O'
    # "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'",
    # "C3'", "O3'", "C2'", "O2'", "C1'",
])

# amino-acids
std_aminoacids = np.array([
    'LEU', 'GLU', 'ARG', 'LYS', 'VAL', 'ILE', 'PHE', 'ASP', 'TYR',
    'ALA', 'THR', 'SER', 'GLN', 'ASN', 'PRO', 'GLY', 'HIS', 'TRP',
    'MET', 'CYS',
])

# resname categories
categ_to_resnames = {
    "protein": ['GLU', 'LEU', 'ALA', 'ASP', 'SER', 'VAL', 'GLY', 'THR', 'ARG',
                'PHE', 'TYR', 'ILE', 'PRO', 'ASN', 'LYS', 'GLN', 'HIS', 'TRP',
                'MET', 'CYS'],
    "rna": ['A', 'U', 'G', 'C'],
    "dna": ['DA', 'DT', 'DG', 'DC'],
    "ion": ['MG', 'ZN', 'CL', 'CA', 'NA', 'MN', 'K', 'IOD', 'CD', 'CU', 'FE', 'NI',
            'SR', 'BR', 'CO', 'HG'],
    "ligand": ['SO4', 'NAG', 'PO4', 'EDO', 'ACT', 'MAN', 'HEM', 'FMT', 'BMA',
               'ADP', 'FAD', 'NAD', 'NO3', 'GLC', 'ATP', 'NAP', 'BGC', 'GDP',
               'FUC', 'FES', 'FMN', 'GAL', 'GTP', 'PLP', 'MLI', 'ANP', 'H4B',
               'AMP', 'NDP', 'SAH', 'OXY'],
    "lipid": ['PLM', 'CLR', 'CDL', 'RET'],
}
resname_to_categ = {rn:c for c in categ_to_resnames for rn in categ_to_resnames[c]}