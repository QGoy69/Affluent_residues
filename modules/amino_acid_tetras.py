# File amino_acid_tetras.py contains amino_acid_tetras class
from modules import protein_atoms # Import the MODULE containing the class
PA = protein_atoms.ProteinAtoms # Import the CLASS from the MODULE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from modules import remove_long_edges as rle
import importlib
importlib.reload(rle)
import json

_RADIUS_DB = None

def _get_radius_db(path: str = 'radius_database.json'):
    """Charges a database and saves it in cache, at the module level"""
    global _RADIUS_DB
    if _RADIUS_DB is None:
        with open(path, 'r') as f:
            _RADIUS_DB = json.load(f)
    return _RADIUS_DB

def set_radius(res_name, atom_name, radius_data=None):
    """Return the radius(Å). res_name ex: 'ALA' ou 'ALA 123'. atom_name ex: 'CA','CB','O'."""
    radius_data = radius_data or _get_radius_db()

    # 1) Checking Backbone first
    bb = radius_data.get("__backbone__", {})
    if atom_name in bb:
        return bb[atom_name]["radius"]

    # 2) Sidechain by residue
    res_code = str(res_name).split()[0] if res_name else None  # ex: 'ALA' depuis 'ALA 123'
    if res_code and res_code in radius_data:
        atom_map = radius_data[res_code]
        if atom_name in atom_map:
            return atom_map[atom_name]["radius"]

    # 3) Default value if not found (ligands, missing cases)
    return 1.76

class amino_acid_tetras(PA):
    """Class representing an amino acid and its associated tetraedra in a Delaunay triangulation. Inherit from protein class.
    Attributes:
        protein (ProteinAtoms) : Protein to which the amino acid belongs. 
        residue_id (str) : Residue identifier of the amino acid, in the form 'ALA 123' with a space.
        chain_id (str) : Chain identifier of the amino acid.
        filter (dict) : {'low':3, 'high': 17} by default. Filter for lengths distrib to calculate breakpoint and exclude cavities only.
        tetraedra (dict) : Dictionnay containing a List of tetraedra (list of 4 atom indices) associated with the amino acid keys: ["tetras"], ["edges"],["lengths"],["filtered_lengths"]. Each keys are {"tetras": [], "edges": [], "lengths": [], "filtered_lengths": []}.
        atom_indices (int) : Index in self.points corresponding to the atoms of the amino acid.
        filtered_length_distrib (list) : a D list containing all the filtered lenths of the amino acid (filtered by filter)
        surface (bool) : a boolean saying if the aa is considered at the surface or burried
        threshold (float) : Threshold distance to cut tetraedra lengths to remove cavities of the protein.
        filtered_tetras (list) : List of the tetraedras that pass the cavity criterion (whcich only have edges below threshold)
        nearest_neighbors (set) : Nearest neighbors of the amino acid
        tetras_volume (list) [n*4] : List of tetraedra forming the empy volume around an amino acid
        empty_volume (float) : Empty volume around the amion acid (in Å^3)

    Functions:
        find_tetraedra : Assign self.tetraedra as an extensive dict.
        edges_data : Calculates edges lengths and filter them for a tetraedron.
        calc_length : Calculates the length of an edge given its 2 vertices.
        plot_hist : Plots an histogram of filtered lengths distribution.
        is_surface : Determine if a residue is at the surface or not. In this case a surface residue is a residue which has cavities that we nt to remove.
        cut_long_edges : Remove the tetraedra having edges longer than the threshold i.e. reomove cavities
        n_in_common : Check wether there is n atoms in common btwn a reference list and a tetraedron.
        select_nearest_neighbors : Select the neares neighbors in the amino acid.
        extract_tetras_volume : Exctract the tetraedra forming the empty volume around the amino acid.
        calc_empty_volume : Calculates the total volume of empty space around the amino acid.
        """
    # Cache for shared ProteinAtoms instances
    _protein_cache = {}

    def __init__(self, protein_pdb, residue_id, chain_id, filter = {'low':3, 'high': 15}):

        # Reuse or create a shared ProteinAtoms instance for this PDB file
        if protein_pdb not in amino_acid_tetras._protein_cache:
            amino_acid_tetras._protein_cache[protein_pdb] = PA(protein_pdb)
        
        # Copy attributes from the shared instance instead of creating a PA class for each residue of the protein
        shared_pa = amino_acid_tetras._protein_cache[protein_pdb]
        self.atoms_data = shared_pa.atoms_data
        self.points = shared_pa.points
        self.residues = shared_pa.residues
        self.chains = shared_pa.chains
        self.structure = shared_pa.structure
        self.triangulation = shared_pa.triangulation
        self.terminal_atoms_indicies = shared_pa.terminal_atoms_indicies
        self.res_atoms = shared_pa.res_atoms
        self.residue_atom_index = shared_pa.residue_atom_index  # Already built!
        self.pdb_file = shared_pa.pdb_file
        self.exclude_solvent = shared_pa.exclude_solvent
        self.ss_map = shared_pa.ss_map
        self.residue_id = residue_id
        self.chain_id = chain_id

        if self.residue_id in self.residues:
            pass
        else:
            raise Exception("The residue asked is not in the sequence.")
        
        self.target_residue = self.select_biopython_res_robust(residue_id, chain_id)   # Biopython residue object
        self.filter = filter
        self.tetraedra = None
        self.atom_indices = None
        self.find_tetraedra()
        self.filtered_length_distrib =np.concatenate(self.tetraedra["filtered_lengths"])
        self.surface = self.is_surface()
        self.threshold = None
        if self.surface:
            self.filtered_tetras = self.cut_long_edges()
        else:
            self.filtered_tetras = self.tetraedra["tetras"] # If the residue is not at the surface, we do not filter the tetraedra.


    def find_tetraedra(self, print_data = False):
        """ Exctract target_residue indices in atoms_data and then extract all tetrahedra with an atom in target_residue.
        Args:
            target_residue (str) : The residue of interest, in the form 'ALA 123' with a space
        Returns:
            Nothing. Assign the self.tetraedra attribute."""
        
        with open('structure_database.json', 'r') as f:
            struct_db = json.load(f)

        # -- Get indices of atoms belonging to target aa --
        key = (self.chain_id, self.residue_id)
        target_indices = self.residue_atom_index.get(key, [])
        # ----

        res_name = self.residue_id.split()[0]
        terminal_atoms = struct_db[res_name]['terminal_atom']

        target_indicies_terminal = [
            i for i in target_indices 
            if self.atoms_data[i]['atom_name'] in terminal_atoms
        ]

        self.atom_indices = target_indices
        self.terminal_atoms_indicies = target_indicies_terminal

        # -- Find edges of the DT belonging to these atoms --
        tetraedra_struct = {"tetras": [], "edges": [], "lengths": [], "filtered_lengths": []}
        #                    points idx(N,4), points idx(N,6*(1,2)), int(N,6), int(N,<6)
        for simplex in self.triangulation.simplices:
            # Check if any vertex of the simplex is in the target aa
            has_target = any(vertex in target_indices for vertex in simplex)
            if has_target: # So if one of the vertex of the tetrahedron is in the target_res
                # A simplex is the an array (N,4) with the indices of atoms forming a tetrahedron.
                # Here, we extract the tetraedra and its edges lengths
                tetraedra_struct["tetras"].append(simplex)
                edges_index, lengths, filtered_lengths = self.edges_data(simplex)
                tetraedra_struct["edges"].append(edges_index)
                tetraedra_struct["lengths"].append(lengths)
                tetraedra_struct["filtered_lengths"].append(filtered_lengths)

        self.tetraedra = tetraedra_struct
        # ----
        if print_data:
            print(f'{self.residue_id} has {len(self.tetraedra["tetras"])} tetraedra in the original DT')
            print(f'{self.residue_id} has {len(np.concatenate(self.tetraedra["lengths"]))} edges')

    def edges_data(self, simplex):
        """ Calculates the length of the edge between the two given points.
        Args:
            simplex (np.array) : The simplex descibing a tetrahedron
        Return:
            edges_index (list) [6*(2)]: List of tuples containing the index (in points) of the edges of all simplex edges.
            lengths (list) [1*6]: List of the lengths of the tetraedra.
            filtered_lengths (list) [1*n] : Remove lengths outside self.filter."""

        lengths =[]
        edges_index =[]
        filtered_lengths = []
        for i in range(4):
            for j in range(i+1,4):
                edges_index.append((simplex[i], simplex[j]))
                length = self.calc_length(simplex[i], simplex[j])
                lengths.append(length) # So this returns the indices of the vertices of the edge in the point structure !
            if length > self.filter['low'] and length < self.filter['high']: # Remove all edges not into the window.
                filtered_lengths.append(length)
        
        return edges_index, lengths, filtered_lengths
        
    def calc_length(self, idx1, idx2):
        """ Calculates the length of the edge between the two given points.
        Args:
            idx1 and idx2 (float): Index in self.points of vertices of the segment whose length we want to calculate
        Return:
            length (float): Edge length."""

        #Get the coordinates of the two points forming the edge
        p1 = self.points[idx1]
        p2 = self.points[idx2]

        #Calculate the Euclidean distance
        length = np.linalg.norm(p1-p2) # Calculates the norm of the vector p1-p2

        return length
    
    def plot_hist(self, bins = 50, print_data = False):
        """Plot the distribution of the filtered lengths.
        Args:
            bins (int) : Number of bins of the histogram
            print_data (bool) : If we want to print information about the number of lengths of the distrib.
        Return:
            Nothing. Plots the histogram"""
        if print_data:
            print(f'{self.residue_id} has {len(self.filtered_length_distrib)} edges after filtering with a filter of {self.filter} Å')

        plt.figure(figsize=(10, 6))
        plt.hist(self.filtered_length_distrib, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
        plt.xlabel('Edge Length (Å)', fontsize=12)
        plt.title(f'Distribution of filtered Edge lengths for {self.residue_id}')
        plt.show()

    def is_surface(self):
        """ Determine if a residue is at the surface by looking at the kurtosis of the distribution of its edges lengths.
        A residue is considered at the surface if the kurtosis is > 0.
        Args:
        Return :
            (bool) : if the residue is at the surface."""
        # Calculate kurtosis, quantifies if a distribution is heavly yailed. kurt<0 means a light tailed distribution
        lengths = np.concatenate(self.tetraedra['lengths'])
        kurt = stats.kurtosis(lengths)

        if kurt < 0:
            return False
        else:
            return True

        
    def cut_long_edges(self):
        """ Remove edges of the residues having cavities. The threshold is determined by the best rupture point.
        Args:
        Return :
            (list) : All the tetraedra having lengths below the best breakpoint to remove cavities."""
        if self.threshold == None:
            self.threshold = rle.find_best_breakpoint(self.filtered_length_distrib, verbose = False) # Searching for the threshold value of this aa

        pd_tetras = pd.DataFrame(self.tetraedra) # Conversion into pd dataframe
        mask = pd_tetras["lengths"].apply(lambda lst: all(x < self.threshold for x in lst)) # Keep only the tetraedra with all edges below the filter

        # Filtering
        filtered_tetras_complete = pd_tetras[mask]
        flitered_tatras = filtered_tetras_complete["tetras"].tolist()
        flitered_tatras = [arr.tolist() for arr in flitered_tatras]
        return flitered_tatras
    
    def _display_filtered_tetras_info(self):
    
        """Affiche rapidement la longueur max et le threshold des tetras filtrés."""
        if isinstance(self.filtered_tetras, list) and len(self.filtered_tetras) > 0:
            if self.surface:
                # ✅ SOLUTION : Convertir en set de tuples pour les comparaisons rapides
                filtered_tetras_set = set(tuple(tetra) for tetra in self.filtered_tetras)

                max_length = max(
                    max(length for length in self.tetraedra["lengths"][i])
                    for i in range(len(self.tetraedra["tetras"]))
                    if tuple(self.tetraedra["tetras"][i]) in filtered_tetras_set
                )
                print(f"{self.residue_id} | Threshold: {self.threshold:.2f} Å | Max edge: {max_length:.2f} Å")
            else:
                # Pas de filtrage (buried residue)
                all_lengths = np.concatenate(self.tetraedra["lengths"])
                max_length = np.max(all_lengths)
                print(f"{self.residue_id} | Buried (no threshold) | Max edge: {max_length:.2f} Å")
    
    
    def extract_tetras_specific_aa(self, residue_list: list):
        """Find the tetraedra between self and residue_list.
        Arg:
            residue_list (list) : List of residue_id of amino acids to consider for the tetre extraction.
        Return :
            tetras_volume (list) [n*6] : A list of tetras forming the empty volume of the amino acid."""
        indicies_interest_res = {} # Indicies of all amino acids of interest
        tetra_of_interest = [] # Tetras involving amino acids of interest
        for interest_res in residue_list:
            interest_indices = [] # Indicies of 1 amino acid
            interest_tetra = [] # Tetras involving this amino acid
            for i, atom in enumerate(self.atoms_data): # atoms_data inherited from ProteinAtoms
                if atom['residue_id'] == interest_res:
                    interest_indices.append(i)
            indicies_interest_res[interest_res] = interest_indices
            print(f"Indicies of interest = {interest_indices}")

            for tetra in self.filtered_tetras:
                atoms_of_interest = set(interest_indices)
                for x in tetra:
                    if x in atoms_of_interest:
                        interest_tetra.append(tetra)
                        continue

            tetra_of_interest.extend(interest_tetra)

        return tetra_of_interest
    
    
    def _is_het_resname(self, resname: str) -> bool:
        standard = {
            'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS',
            'ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP',
            'TYR','VAL','SEC','PYL'
        }
        return resname not in standard
    
    def select_biopython_res_robust(self, residue_id, chain_id):
        resname, seq_icode = residue_id.split()
        seqnum = int(''.join(ch for ch in seq_icode if ch.isdigit()))
        if self._is_het_resname(resname):
            target_residue = self._find_het_residue(resname, seqnum)
        else:
            target_residue = self.structure[chain_id][seqnum]

        return target_residue
    
    def extract_terminal_tetras(self, type_of_tetra: str = "at_least_1_terminal_atom"):
        """Extract the tetraedra having at least 1 terminal atom from the current amino acid in it.
        Args:
            type_of_tetra (str) : "at_least_1_terminal_atom" or "dof_sum_sup_x" (x being a number)."""
        from collections import Counter

        terminal_tetras = {}
        with open('dof_database.json', 'r') as f:
            dof_db = json.load(f)

        # Step 1: find tetras belonging to the current aa
        tetras_with_terminal_atoms = set()
        res_of_atom_in_tetra = []
        tetras_dof = []
        for tetra in self.filtered_tetras:
            dof_sum = 0
            for atom in tetra:
                atom_name = self.atoms_data[atom]['atom_name'] # "CA"
                res_of_atom = self.atoms_data[atom]['residue_id'] # residue id of the atom ("ALA 10 ")
                res_of_atom_in_tetra.append(res_of_atom)
                dof_atom = dof_db.get(res_of_atom.split()[0], {}).get(atom_name, 0) # get the dof for this atom, default to 0 if not found
                dof_sum += dof_atom
                if atom in self.terminal_atoms_indicies:
                    tetras_with_terminal_atoms.add(tuple(tetra))

            counts = Counter(res_of_atom_in_tetra)
            atoms_belonging_to_same_res = max(list(counts.values())) if counts else 0
            tetras_dof.append(dof_sum)



        if type_of_tetra == "at_least_1_terminal_atom":
            #Step 2: Know the dof and number of terminal atoms in the tetra
            for tetra in tetras_with_terminal_atoms:
                count_term_atoms = 0
                for atom in tetra:
                    if atom in self.terminal_atoms_indicies:
                        count_term_atoms += 1

                # Initialiser la clé si elle n'existe pas
                if count_term_atoms not in terminal_tetras:
                    terminal_tetras[count_term_atoms] = []

                # Translate each tetra to the origin (using N atom as reference) and add it to the list of tetras with count_term_atoms terminal atoms
                tetra_coord = []
                for atom in tetra:
                    atom_coord = self.atoms_data[atom]['coord']
                    tetra_coord.append(atom_coord)

                # Ajouter la liste de coordonnées à la liste
                terminal_tetras[count_term_atoms].append(tetra_coord)



            return terminal_tetras
        
        elif type_of_tetra.startswith("dof_sum_sup_"):
            treshold = int(type_of_tetra.split('_')[-1])
            tet_to_select = [tetra for tetra, dof in zip(self.filtered_tetras, tetras_dof) if dof > treshold]
            all_tetra_coord = []
            nei_res = set()
            for tetra in tet_to_select:
                tet_coord =[]
                for atom in tetra:
                    atom_coord = self.atoms_data[atom]['coord']
                    atom_chain = self.atoms_data[atom]['chain_id'] # "A"
                    res_of_atom = self.atoms_data[atom]['residue_id']
                    nei_res.add((res_of_atom, atom_chain))
                    tet_coord.append(atom_coord)
                all_tetra_coord.append(tet_coord)

            return all_tetra_coord, nei_res
            
