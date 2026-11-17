from typing import Counter

from Bio.PDB import PDBParser, PDBList
import os
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.internal_coords import *
import numpy as np
from scipy.spatial import Delaunay



class ProteinAtoms():
    """This class takes a PDD file and parse the atoms of the protein.
    It keeps the informations associated with the atoms.
    
    Attributes:
        pdb_file (str) : The pdb file name containing N atoms
        atoms_data (list of dict) : Contain each atom information (ccordinate, residue, element ... See ~l180 for details)
        points (np.array) [n_atoms,3]: Contains the coordinates of all the atoms of the protein, shape (N,3)
        structure (chain) : Biopython object containing the first model of the pdb file.
        triangulation (Delaunay) : The Delaunay triangulation of the protein atoms. Delaunay object from scipy.spatial
        exclude_solvent (bool) : Whether to exclude solvent molecules from the analysis ro not
        residues (list) [n-res]: List of all residues in the protein, in the form 'ALA 123' with a space
        # gabriel (GabrielGraphOptimized) : The Gabriel graph of the protein atoms, computed from the DT if specified.
        # gabriel_edges (list) [n_edges,2] : List of the edges of the Gabriel graph, each edge contains the indices of its two atoms in the points structure.
    """

    _cache = {}  # key: (pdb_file, exclude_solvent, chain_id) -> dict of data

    def __init__(self, pdb_file, chain_restriction = None, exclude_solvent=True, force_reload=False, dih_angles = False):
            # If pdb_file is just a PDB ID (e.g., '1be9'), construct the full path
        if not os.path.exists(pdb_file):
            # Check if it looks like a PDB ID (no path separators, no extension or .pdb extension)
            basename = os.path.basename(pdb_file)
            if '.' not in basename or pdb_file.endswith('.pdb'):
                module_dir = os.path.dirname(os.path.abspath(__file__))
                pdb_dir = os.path.join(module_dir, 'pdb_files')
                pdb_file = os.path.join(pdb_dir, f"{basename.split('.')[0]}.pdb")
        self.pdb_file = pdb_file
        self.exclude_solvent = exclude_solvent
        self._key = (self.pdb_file, self.exclude_solvent)

        if force_reload or self._key not in ProteinAtoms._cache:
            # print(f'Calculating {self.pdb_file} data...')
            self._build_cache_entry(chain_restriction)

        self.ss_map = self.build_ss_map_from_pdb() # Secondary structure map   

        data = ProteinAtoms._cache[self._key]

        # Instance views on cached data
        self.atoms_data = data['atoms_data']
        self.points = data['points']
        self.residues = data['residues']
        self.chains = data['chains']
        self.structure = data['structure']
        self.triangulation = data['triangulation']
        self.terminal_atoms_indicies = data['terminal_atoms_indicies']
        self.res_atoms = data['res_atoms']

        self._build_residue_index()  # Build the residue index for quick lookup (attribute self.residue_atom_index)

        if dih_angles:
            import json
            import math
            dih_data = {}
            ppb = PPBuilder()
            chain = self.structure
            # Load the structure database to get the number of chi angles per residue
            with open('structure_database.json', 'r') as f:
                aa_db = json.load(f)

            # Convert to internal coordinates for the chain
            chain.atom_to_internal_coordinates(verbose=False)

            for poly in ppb.build_peptides(chain):
                phi_psi = poly.get_phi_psi_list()
                for res, (phi, psi) in zip(poly, phi_psi):
                    res_name = res.get_resname()
                    if res_name in aa_db:
                        res_key = f"{res.get_resname()} {res.get_id()[1]}"
                        if phi:
                            phi = round(math.degrees(phi),3)
                        if psi:
                            psi = round(math.degrees(psi),3)
                        dih_data[res_key] = {'φ': phi, 'ψ': psi}

                        # Get the number of chi angles for this residue type from the database
                        n_chi = aa_db[res_name]['dof']
                        ric = res.internal_coord

                        # Calculate all available chi angles
                        for chi_num in range(1, n_chi + 1):
                            chi_name = f"chi{chi_num}"
                            try:
                                chi_angle = ric.get_angle(chi_name)
                                dih_data[res_key][f"χ{chi_num}"] = chi_angle
                            except Exception:
                                # Chi angle not available for this residue
                                pass
                            
            # Store dih_data in the cache for later access
            self.dih_data = dih_data


    def _build_cache_entry(self, chain_restriction, print_info=False):
        """Builds and stores the parsed data and triangulation into the cache for this key."""
        import json

        pdb_file = self.pdb_file

        # -- Download pdb file if it is not in pdb_files --
        if not os.path.exists(pdb_file):
            # Extract PDB ID from filename (e.g., "4hsc.pdb" -> "4hsc")
            pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]

            # Get the module directory to save PDB files there
            module_dir = os.path.dirname(os.path.abspath(__file__))
            self.pdb_save_dir = os.path.join(module_dir, 'pdb_files')
            os.makedirs(self.pdb_save_dir, exist_ok=True)
            
            # Download from PDB
            pdbl = PDBList(verbose=False)
            downloaded_file = pdbl.retrieve_pdb_file(pdb_id, pdir=self.pdb_save_dir, file_format='pdb')

            # Extract just the filename from the path
            filename = os.path.basename(downloaded_file)

            # Remove 'pdb' prefix from filename if present
            if filename.startswith('pdb'):
                new_filename = filename[3:]  # Remove 'pdb' prefix
                new_pdb_file = os.path.join(os.path.dirname(downloaded_file), new_filename)
                os.rename(downloaded_file, new_pdb_file)
                downloaded_file = new_pdb_file

            # Rename .ent file to .pdb if necessary
            if downloaded_file.endswith('.ent'):
                pdb_file = downloaded_file.replace('.ent', '.pdb')
                os.rename(downloaded_file, pdb_file)
            else:
                pdb_file = downloaded_file

            print(f"PDB file downloaded: {pdb_file}")
        # ----

        # -- Extracting all the informations from the pdb file needed --
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)

        coords_list = []
        residue_list = []
        atoms_data = []
        terminal_atoms_indicies = []
        res_atoms ={}
        seq = ''

        first_model_id = None
        chain_ids = []
        with open('structure_database.json', 'r') as f:
            aa_db = json.load(f)

        # Iterate over all models and chains
        idx = 0
        for model in structure:
            if first_model_id is None:
                first_model_id = model.id
            for chain in model:
                if chain_restriction and chain.id not in chain_restriction:
                    continue
                chain_ids.append(chain.id)
                for residue in chain:
                    if self.exclude_solvent and self._is_solvent(residue):
                        continue
                    current_res = f"{residue.get_resname()} {residue.get_id()[1]}"
                    residue_list.append(current_res)
                    terminal_atoms = aa_db[current_res.split()[0]].get('terminal_atom', [])
                    res_atoms[current_res] = []
                    seq += aa_db[current_res.split()[0]].get('1_letter_code', [])
                    for atom in residue:
                        atoms_data.append({
                            'coord': atom.get_coord(),
                            'residue_id': current_res,
                            'atom_name': atom.get_name(),
                            'chain_id': chain.id,
                            'model_id': model.id,
                            'element': atom.element
                        })
                        if atom.get_name() in terminal_atoms:
                            terminal_atoms_indicies.append(idx)
                        res_atoms[current_res].append(idx)
                        idx += 1
                        coords_list.append(atom.get_coord())

        points = np.array(coords_list)
        triangulation = Delaunay(points)
        if print_info:
            print('PDB file loaded successfully.')
            print('Delaunay triangulation completed.')

        #printing aa distrivution
        self.sequence = seq
        #----

        # -- Filling cache entry --
        # Here I am selecting ONLY the first model. This may not be the case for all application cases
        ProteinAtoms._cache[self._key] = {
            'atoms_data': atoms_data,
            'points': points,
            'residues': residue_list,
            'chains': chain_ids,
            'structure': structure[first_model_id],
            'triangulation': triangulation,
            'terminal_atoms_indicies': terminal_atoms_indicies,
            'res_atoms': res_atoms
        }
        # ----

    def build_ss_map_from_pdb(self):
        """Parse HELIX/SHEET records directly from a PDB file into a dict:
           key=(chain_id, resseq, icode) -> 'H' or 'E'."""
        ss = {}
        with open(self.pdb_file, 'r') as f:
            for line in f:
                rec = line[:6]
                if rec.startswith('HELIX '):
                    # PDB format columns (1-based):
                    # initChainID(20), initSeqNum(22-25), initICode(26)
                    # endChainID(32),  endSeqNum(34-37),  endICode(38)
                    c1 = _icode_norm(line[19])
                    sseq = int(line[21:25].strip())
                    sic = _icode_norm(line[25])
                    c2 = _icode_norm(line[31])
                    eseq = int(line[33:37].strip())
                    eic = _icode_norm(line[37])
                    if c1 != c2:
                        continue  # safety
                    step = 1 if sseq <= eseq else -1
                    for r in range(sseq, eseq + step, step):
                        # interior residues use ' ' icode; endpoints carry ICodes
                        icode = ' '
                        if r == sseq:
                            icode = sic
                        if r == eseq:
                            icode = eic
                        ss.setdefault((c1, r, icode), 'H')
                elif rec.startswith('SHEET '):
                    # start: chain(22), seq(23-26), iCode(27)
                    # end  : chain(33), seq(34-37), iCode(38)
                    c1 = _icode_norm(line[21])
                    sseq = int(line[22:26].strip())
                    sic = _icode_norm(line[26])
                    c2 = _icode_norm(line[32])
                    eseq = int(line[33:37].strip())
                    eic = _icode_norm(line[37])
                    if c1 != c2:
                        continue
                    step = 1 if sseq <= eseq else -1
                    for r in range(sseq, eseq + step, step):
                        icode = ' '
                        if r == sseq:
                            icode = sic
                        if r == eseq:
                            icode = eic
                        ss.setdefault((c1, r, icode), 'E')
        return ss

    def _is_solvent(self, residue):
        """ Check if a ressidue is a solvent molecule."""
        RESIDUE = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

        return residue.get_resname() not in RESIDUE


# ----------------------------
    def delaunay_triangulation(self):
        """Performs Delaunay triangulation using protein atoms coordinates."""
        ProteinAtoms._triangulation =Delaunay(self.points)


    def _build_residue_index(self):
        """Build a lookup index: (chain_id, residue_id) -> [atom_indices]"""
        self.residue_atom_index = {}
        for i, atom in enumerate(self.atoms_data):
            key = (atom['chain_id'], atom['residue_id'])
            if key not in self.residue_atom_index:
                self.residue_atom_index[key] = []
            self.residue_atom_index[key].append(i)


    def delaunay_to_vtu(self, filtered = False, filtered_simplices = None, filename="output.vtu"):
        """
        Export a tetrahedral mesh (from scipy.spatial.Delaunay)
        to a VTU file readable by ParaView or obj file readable by ChimeraX.

        Parameters
        ----------
        points : (N, 3) array
            Coordinates of points (atoms).
        simplices : (M, 4) array
            Indices of tetrahedra (each row = 4 vertex indices).
        filename : str
            Output file name (should end with .vtu).
        """
        import pyvista as pv

        n_tets = len(self.triangulation.simplices)
        statement = f"Saved {filename} with {len(self.points)} points and "
        if filtered:
            simplices = filtered_simplices
            n_tets_actual = len(filtered_simplices)
            statement = statement + f"{n_tets_actual} tatrahedra ({n_tets} originally)."
        else:
            simplices = self.triangulation.simplices
            statement = statement + f"{n_tets} tatrahedra."
            n_tets_actual = n_tets

        # Build PyVista UnstructuredGrid
        # Each tetra cell: [n_points_in_cell, p0, p1, p2, p3]
        cells = np.hstack(
            np.c_[np.full(n_tets_actual, 4), simplices]
        ).astype(np.int64).ravel()

        # VTK cell type for tetrahedra is 10
        celltypes = np.full(n_tets_actual, 10, dtype=np.uint8)

        grid = pv.UnstructuredGrid(cells, celltypes, self.points)

        # Si fichier .obj, extraire la surface du maillage
        if filename.endswith('.obj'):
            grid = grid.extract_surface(algorithm = 'dataset_surface')

        grid.save(filename)



def _icode_norm(x):
    # mmCIF may use '?' or '.'; PDB uses ' ' when no insertion code
    return ' ' if x in (None, '', '?', '.') else x



