import numpy as np
from Bio.PDB import PDBParser

def read_and_clean_coord(filename, pdb_id, chn_id, insert_IDs, model_id=0):
    """
    filename: (str) pdb structure file path
    pdb_id: (str) four letter pdb code
    chn_id: (str) chain letter to work on
    model_id: (int) default to 0
    """
    # First read in the Ca coords
    p = PDBParser(PERMISSIVE=1)                    # PDB parser obj
    structure = p.get_structure(pdb_id, filename)  # read in PDB structure
    model = structure[model_id]                    # get model from structure
    chain = model[chn_id]
    # Clean PDB of all heteroatoms
    residue_to_remove = []
    res_IDs = []  # does this really have to be here???
    for residue in chain:
        res_IDs.append(residue.get_id())
        if residue.id[0] != ' ':
            residue_to_remove.append((chain.id, residue.id))
    for residue in residue_to_remove:
        model[residue[0]].detach_child(residue[1])
    # Get starting res index number to correct for PDB not starting from 1
    PDB_starting_point_str = str(res_IDs[0])
    # this is a VERY messy way to get the starting number and correct for indexing
    PDB_starting_point = int(''.join(i for i in PDB_starting_point_str if i.isdigit()))
    # Optional: remove loop regions
    residue_ids_to_remove = insert_IDs
    for id in residue_ids_to_remove:
        if chain.__contains__(id) == True: #make sure that the residue that needs to be removed actually exists in the PDB structure 
            chain.detach_child((' ', id, ' '))
    return chain, PDB_starting_point

def get_ca_coord(chn_id, start, end, PDB_starting_point):
    """
    chn_id: (str) chain letter to work on
    start: (int) user defined starting residue
    end: (int) user defined ending residue
    PDB_starting_point: (int) 
    """
    Ca_num = len(chn_id)
    Cas = []
    atm_ca_list = []
    for i in range(Ca_num):
        res = chn_id.get_unpacked_list()[i]  # returns list of all atoms
        res_list = [str(atom) for atom in res.get_unpacked_list()] # convert all of the atom names from class type objects to strings 
        if res_list.count('<Atom CA>') > 0: # make sure that the residues being processed do have Ca atoms (some PDB resideus only have the N atom present at the end)
           atm_ca = res['CA'].get_vector()
           atm_ca_list = list(atm_ca)
           Cas.append(atm_ca_list)
    # corrects for PDB indexing not starting from 1
    Cas = np.array(Cas)
    solenoid_region = Cas[(start - PDB_starting_point):(end - PDB_starting_point + 1)]
    return solenoid_region

def find_repeat_centroids(coords, coil_len):
    """
    coords: (array) chain letter to work on
    coil_len: (int) user defined length of repeating unit to form a coil (i.e LRR 22-30 AA, PPR 5*4=20AA) 
    """
    num_of_coils = int(coords.shape[0] / coil_len)
    centroids_ls = []
    for i in range(1, num_of_coils + 1):
        new_centroid = np.mean(coords[(coil_len*i - coil_len): coil_len*i], axis=0)
        centroids_ls.append(list(new_centroid))
    centroids = np.array(centroids_ls)
    centroid_no = centroids.shape[0]
    print("--- Number of repeats: ", centroids.shape[0])
    return centroids, centroid_no

def find_Ca_for_second_helix(coords, coil_len, Ca_IDs, insert_IDs):
    # first make sure that the inserts do not disrupts Ca_ID values and correct for any movements
    new_cas = []
    for i in Ca_IDs:
        new_ca = i
        for j in insert_IDs:
            if i > j:
                new_ca = new_ca - 1
        new_cas.append(new_ca)
    residues = sorted(new_cas)
    Ca_coords = []
    for i in residues:
        Ca_coord = coords.T[i-residues[0]]
        Ca_coords.append(list(Ca_coord))
    Ca_coords_arr = np.array(Ca_coords)
    return Ca_coords_arr