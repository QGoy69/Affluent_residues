# This python file analyses the Delaunay tetrahedra in terms of number of terminal atoms/ number of degrees of freedom.

from copyreg import pickle
import os

from modules import amino_acid_tetras # Import the MODULE containing the class
import importlib
importlib.reload(amino_acid_tetras) # Reload the MODULE to ensure the latest version is used
AAT = amino_acid_tetras.amino_acid_tetras # Import the CLASS from the MODULE
from modules import protein_atoms 
PA = protein_atoms.ProteinAtoms
import math
import matplotlib.pyplot as plt
import numpy as np
import json
import networkx as nx
import pandas as pd
from collections import Counter, defaultdict


class TetraAnalyser():
    """Class to analyse the tetrahedra of a list of proteins in terms of number of terminal atoms and degrees of freedom.
    Attributes:
        protein_list (list): The list of pdb structures that will be analysed.
        PA_proteins (list): A list of PA objects corresponding to the pdb structures.
        tetras_by_prot (dict): A dictionary where keys are protein names and values are sets of unique filtered tetrahedra.
        fit_params (dict): A dictionary to store the parameters of the exponential fit on tetras dof distribution for each PDB.
    Methods:
        extract_filtered_tetras(): Extracts the filtered tetrahedra for each protein in the list.
        counter(): Counts the number of tetrahedra with 0, 1, 2, 3, or 4 terminal atoms and organizes the data for plotting,
        also calculates the number of dof for each tetra"""
    
    standard_res = {
            'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS',
            'ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP',
            'TYR','VAL','SEC','PYL'
        }
    
    tet_in_diff_res = [4,3] # Do not take into account internal tetrahedra or tetrahedra btwn 2 amino acids.

    def __init__(self, prot_info: list, filter = False):
        """Initializes the tetra_analyser with a list of protein PDB files.
        Args:
            prot_info (list): The list of pdb structures that will be analysed. If contains 2 informations: the first one is the pdb file name, the second one is the chain id(s) to consider for the analysis. If only 1 information, it is considered as the pdb file name and all chains are considered for the analysis.

        Attributes:
            protein_list (list): The list of pdb structures that will be analysed.
            chain_ids (list): The list of chain ids to consider for each protein in the list.
            tetras_by_prot (dict): Keys are protein names and values are sets of unique filtered tetrahedra.
            tetras_dof (dict): Keys are protein names and values are lists of degrees of freedom for each tetrahedron.
            number_of_diff_res (dict): Number of different residues in each tetrahedron.
            residues_in_tetra (dict): List of residues in each tetrahedron.
            dof_threshold (dict): sdm threshold defined by a comparison between random and protein sdm distributions.
        """
        
        self.protein_list = [prot[0] for prot in prot_info]
        self.chain_ids = [prot[1] for prot in prot_info]
        self.tetras_by_prot = self.extract_filtered_tetras()
        self.tetras_dof, self.number_of_diff_res, self.residues_in_tetras = self.counter() # ntait = number_of_terminal_atoms_in_tetras
        # self.ntait , self.tetras_dof, self.number_of_diff_res, self.residues_in_tetras = self.counter() # ntait = number_of_terminal_atoms_in_tetras
        self.dof_threshold = self.define_threshold(export_data=False)

        # Selection of residues with sdm >= dof_threshold
        self.res_involved, self.tetras_list, self.corresp_res_in_tetra = self.calc_raw_data()

        # Some analysis to get the size and weight of tetrahedra clusters
        self.graphs = self.get_high_sdof_clusters(print_info=False)
        self.prot_sdof_carac = self.get_connected_components_info(print_info=False)
        if filter:
            self.filter_components_by_size()
            self.graphs = self.get_high_sdof_clusters(print_info=False)
            self.prot_sdof_carac = self.get_connected_components_info(print_info=False)

        # Print information in the log:
        self.final_df = self.display_components_table()
        # For plots, calculates the number of necessary columns (plot on 2 lines)
        self.num_pdbs = len(self.protein_list)
        self.num_cols = math.ceil(self.num_pdbs / 2)

        #Look at the resides distribution in the tetrahedra (uncomment to execute)
        # self.plot_aa_distrib_involved(export_as_txt=True)

    def extract_filtered_tetras(self, print_info = False):
        """Extracts the filtered tetrahedra for each protein in the list.
        Returns:
            dict: A dictionary where keys are protein names and values are sets of unique filtered tetrahedra. The tetrahedra are simplices from scipy.spatial.Delaunay.
        """
        
        PA_proteins = {}
        tetras_by_prot = {}
        dof_detail = {}
        for idx, (pdb, chains) in enumerate(zip(self.protein_list, self.chain_ids)):
            chain_restriction = chains if chains != 'all' else None
            current_prot = PA(pdb, chain_restriction=chain_restriction)
            PA_proteins[pdb] = current_prot
            dof_detail[pdb] = {}
            filtered_tetras_prot = set() # Will contain all unique tetrahedra of the structure
            if chains == 'all':
                self.chain_ids[idx] = current_prot.chains # if "all" chains, we consider all the chains of the structure for the analysis

           # Iterate through non-water residues
            for chain in getattr(current_prot, "structure"):
                atom_in_chain = 0
                dof_detail[pdb][chain] = {}
                if chain.id in self.chain_ids[idx]:
                    print(f"Processing chain {chain.id} of {pdb}...")
                    
                    with open('dom_database.json', 'r') as f:
                        dom_database = json.load(f)

                    for res in chain:
                         # Getting the chain id of the residue
                         chain_id = res.get_parent().id
                         # == Getting a unique set of tetrahedra from each aa ==
                         if res.get_resname() not in TetraAnalyser.standard_res:
                             continue
                         name = f'{res.get_resname()} {res.get_id()[1]}'

                         for atom in res:
                             dof_atom = dom_database.get(res.get_resname(), {}).get(atom.get_name(), 0)
                             atom_in_chain += 1
                             # dof_detail[dof_atom] = dof_detail.get(dof_atom, 0) + 1
                             if dof_atom not in dof_detail[pdb][chain]:
                                 dof_detail[pdb][chain][dof_atom] = 1
                             else:
                                 dof_detail[pdb][chain][dof_atom] += 1
                             
                         res = AAT(pdb, name, chain_id)
                         res_tetras = res.filtered_tetras # Extract filtered tetras from the residue
                         filtered_tetras_prot.update(tuple(tetra) for tetra in res_tetras)
                if print_info:
                    print(f"Total atoms for chain {chain.id} of {pdb}: {atom_in_chain}; DOF detail: {dof_detail[pdb][chain]}")

            tetras_by_prot[pdb] = filtered_tetras_prot
            if print_info:
                print(f"Number of unique filtered tetrahedra for {pdb}: {len(filtered_tetras_prot)}")
        self.dof_detail = dof_detail
        self.PA_proteins = PA_proteins
        return tetras_by_prot
    
    def counter(self ,exclude_internal_tetras = False):

        tetras_dof = {}
        number_of_diff_res = {}
        residues_in_tetra = {}

        with open('dom_database.json', 'r') as f:
            dom_database = json.load(f)
        i=0

        for pdb, tetra_ensemble in self.tetras_by_prot.items():

            tetras_dof[pdb] = []
            number_of_diff_res[pdb] = []
            residues_in_tetra[pdb] = []

            for tetra in tetra_ensemble:
                res_of_atom_in_term_tetra = []
                res_of_atom_in_tetra = []
                count_terminal_atoms = 0
                dof_sum = 0
                for atom in tetra:
                    # == Terminal atoms ==
                    current_protein = self.PA_proteins[pdb]
                    at = current_protein.atoms_data[atom]
                    # if at['atom_name'] in ['N', 'CA', 'C', 'O']:
                    #     res_of_atom = (at['residue_id'], at['chain_id'], at['atom_name']) # residue id of the atom ("ALA 10", 'A', "CA")
                    # else:
                    #     res_of_atom = (at['residue_id'], at['chain_id']) # residue id of the atom ("ALA 10", 'A')
                    res_of_atom = (at['residue_id'], at['chain_id']) # residue id of the atom ("ALA 10", 'A')
                    res_of_atom_in_tetra.append(res_of_atom)
                    if atom in current_protein.terminal_atoms_indicies:
                        count_terminal_atoms += 1
                        atom_idx_in_res = res_of_atom[0] # residue id of the atom ("ALA 10 ")
                        res_of_atom_in_term_tetra.append(atom_idx_in_res)
                    # res_of_atom_in_term_tetra give the aa of each TA in the tetra: ["ALA 10", "ALA 10", "GLY 15", "SER 20"] if 4 TA, ["ALA 10", "GLY 15", "SER 20"] if 3 TA, etc.
                    # Well dissociated, count each tetra ONCE !

                    # == Number of dof ==
                    res_name = res_of_atom[0].split()[0] # "ALA"
                    atom_name = at['atom_name'] # "CA"
                    dof_atom = dom_database.get(res_name, {}).get(atom_name, 0) # get the dof for this atom, default to 0 if not found
                    dof_sum += dof_atom


                atoms_belonging_to_same_res = len(set((roa[0], roa[1]) for roa in res_of_atom_in_tetra))

                if exclude_internal_tetras:
                    if atoms_belonging_to_same_res > 1:# if we don't want to consider internal tetras
                        tetras_dof[pdb].append(dof_sum)
                        number_of_diff_res[pdb].append(atoms_belonging_to_same_res)
                        residues_in_tetra[pdb].append(res_of_atom_in_tetra)
                else:
                    tetras_dof[pdb].append(dof_sum)
                    number_of_diff_res[pdb].append(atoms_belonging_to_same_res)
                    residues_in_tetra[pdb].append(res_of_atom_in_tetra)

                i+=1

        return tetras_dof, number_of_diff_res, residues_in_tetra
    
    def calc_raw_data(self, nadr = [3,4]):
        """Exports the tetras with proba < threshold to vtu files for paraview visualization.
        Args:
            threshold (float): The probability threshold for exporting tetras. Tetras with a probability of having a dof sum above the threshold will be exported.
            to choose from [5,3,2]% of the distribution.
        """

        tetras_to_export = {}
        corresp_res_for_tetra = {}
        residues_involved = {}

        for pdb in self.protein_list:
            print("All tetrahedra with sdof sum above", self.dof_threshold[pdb], "will be selected for protein", pdb)
            tetras_to_export[pdb] = {num: [] for num in np.array([nadr]).flatten().tolist()}
            corresp_res_for_tetra[pdb] = {num: [] for num in np.array([nadr]).flatten().tolist()}
            residues_involved[pdb] = set()
            dom_threshold_prot = self.dof_threshold[pdb]

            for number_of_at_in_diff_res in np.array([nadr]).flatten().tolist():
                for dof, tetra, res_in_tetra, diff_res in zip(self.tetras_dof[pdb], self.tetras_by_prot[pdb], self.residues_in_tetras[pdb], self.number_of_diff_res[pdb]):
                    if dof >= dom_threshold_prot and diff_res == number_of_at_in_diff_res:
                        tetras_to_export[pdb][number_of_at_in_diff_res].append(tetra)
                        corresp_res_for_tetra[pdb][number_of_at_in_diff_res].append(res_in_tetra)
                        residues_involved[pdb].update(res_in_tetra)

        return residues_involved, tetras_to_export, corresp_res_for_tetra
        
    def get_high_sdof_clusters(self, print_info=False):
        # === Creating the residue-level network ===
        proteins_graphs = {}
        atom_graphs = {}

        for pdb in self.protein_list:
            proteins_graphs[pdb] = nx.Graph()
            atom_graphs[pdb] = nx.Graph()
            protein_atoms = self.PA_proteins[pdb]

            # Step 1: Create nodes for aa involved in high sdof tetras
            for aa_id in self.res_involved[pdb]:
                proteins_graphs[pdb].add_node(aa_id)

            # Step 1b: Create atom nodes with coordinates for the PyMOL viewer.
            # Node keys are atom indices to avoid collisions between residues.
            for number_of_at_in_diff_res in TetraAnalyser.tet_in_diff_res:
                for tetra in self.tetras_list[pdb][number_of_at_in_diff_res]:
                    for atom_idx in tetra:
                        atom_data = protein_atoms.atoms_data[atom_idx]
                        atom_graphs[pdb].add_node(
                            atom_idx,
                            pos=tuple(atom_data['coord']),
                            coord=tuple(atom_data['coord']),
                            atom_name=atom_data['atom_name'],
                            residue_id=atom_data['residue_id'],
                            chain_id=atom_data['chain_id'],
                            element=atom_data.get('element')
                        )

            # Step 2: Create edges between nodes based on the tetras and their importance
            for number_of_at_in_diff_res in TetraAnalyser.tet_in_diff_res:
                weight = 2 if number_of_at_in_diff_res == 4 else 1

                for tetra in self.corresp_res_in_tetra[pdb][number_of_at_in_diff_res]:
                    for i in range(len(tetra)):
                        for j in range(i + 1, len(tetra)):
                            node_i = tetra[i] if len(tetra[i]) == 3 else (tetra[i][0], tetra[i][1])
                            node_j = tetra[j] if len(tetra[j]) == 3 else (tetra[j][0], tetra[j][1])
                            if proteins_graphs[pdb].has_edge(node_i, node_j):
                                proteins_graphs[pdb][node_i][node_j]['weight'] += weight
                            else:
                                proteins_graphs[pdb].add_edge(node_i, node_j, weight=weight)

            # Step 2b: Build the atom-level graph by connecting all atom pairs inside each tetrahedron.
            for number_of_at_in_diff_res in TetraAnalyser.tet_in_diff_res:
                weight = 2 if number_of_at_in_diff_res == 4 else 1

                for tetra in self.tetras_list[pdb][number_of_at_in_diff_res]:
                    for i in range(len(tetra)):
                        for j in range(i + 1, len(tetra)):
                            atom_i = tetra[i]
                            atom_j = tetra[j]
                            if atom_graphs[pdb].has_edge(atom_i, atom_j):
                                atom_graphs[pdb][atom_i][atom_j]['weight'] += weight
                            else:
                                atom_graphs[pdb].add_edge(atom_i, atom_j, weight=weight, artificial=False)

            # Step 2c: Add artificial intra-residue edges so each residue stays connected.
            residue_to_atoms = {}
            for atom_idx, atom_data in atom_graphs[pdb].nodes(data=True):
                residue_key = (atom_data['chain_id'], atom_data['residue_id'])
                residue_to_atoms.setdefault(residue_key, []).append(atom_idx)

            for atoms_in_residue in residue_to_atoms.values():
                if len(atoms_in_residue) < 2:
                    continue
                for i in range(len(atoms_in_residue)):
                    for j in range(i + 1, len(atoms_in_residue)):
                        atom_i = atoms_in_residue[i]
                        atom_j = atoms_in_residue[j]
                        if atom_graphs[pdb].has_edge(atom_i, atom_j):
                            # atom_graphs[pdb][atom_i][atom_j]['artificial'] = True
                            continue
                        else:
                            atom_graphs[pdb].add_edge(
                                atom_i,
                                atom_j,
                                weight=0,
                                artificial=True,
                                intra_residue=True
                            )

            if print_info:
                print(f"Network created with {proteins_graphs[pdb].number_of_nodes()} nodes and {proteins_graphs[pdb].number_of_edges()} edges")
                print(f"Atom network created with {atom_graphs[pdb].number_of_nodes()} nodes and {atom_graphs[pdb].number_of_edges()} edges")

            import pickle
            import os

            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Outputs",pdb)
            affluent_data_file = os.path.join(output_dir, f"{pdb[:4]}_tetras_viz_pymol_atoms.pkl")

            with open(affluent_data_file, 'wb') as f:
                pickle.dump(atom_graphs[pdb], f)

        self.atom_graphs = atom_graphs
        return proteins_graphs
    

    def get_connected_components_info(self, print_info=False):
        """Calculates connected components for each protein graph and extracts information about them.

        Returns:
            dict: A dictionary where keys are protein names and values are dictionaries containing:
                - 'number_of_res': list of number of nodes in each connected component
                - 'weight': list of sum of edge weights in each connected component
                - 'amino_acids': list of lists, each containing the residues in each connected component
        """

        components_info = {}

        for pdb, graph in self.graphs.items():
            components_info[pdb] = {
                'number_of_res': [],
                'weight': [],
                'amino_acids': [],
                'tetras_4_res': [],
                'tetras_3_res': []
            }

            # Get connected components
            connected_comps = list(nx.connected_components(graph))

            for idx, component in enumerate(connected_comps):
                # Number of nodes (residues) in this component
                num_nodes = len(component)
                components_info[pdb]['number_of_res'].append(num_nodes)

                # Sum of weights for edges in this component
                subgraph = graph.subgraph(component)
                total_weight = sum(data['weight'] for u, v, data in subgraph.edges(data=True))
                components_info[pdb]['weight'].append(total_weight)

                # List of amino acids (residues) in this component
                amino_acids = sorted(list(component))
                components_info[pdb]['amino_acids'].append(amino_acids)

                # Count tetrahedra by type in this component
                tetras_4 = sum(1 for tetra in self.corresp_res_in_tetra[pdb][4] 
                              if all(res in amino_acids for res in tetra))
                tetras_3 = sum(1 for tetra in self.corresp_res_in_tetra[pdb][3] 
                              if all(res in amino_acids for res in tetra))

                components_info[pdb]['tetras_4_res'].append(tetras_4)
                components_info[pdb]['tetras_3_res'].append(tetras_3)

        return components_info
    
    def display_components_table(self, print_info=True):
        """Displays connected components information in a formatted table sorted by average residue number.
        """

        for pdb, comp_info in self.prot_sdof_carac.items():
            if print_info:
                print(f"\n{'='*80}")
                print(f"Connected Components for {pdb}")
                print(f"{'='*80}\n")

            # Prepare data for DataFrame
            rows = []
            for idx, (num_res, weight, amino_acids, tetras_4, tetras_3) in enumerate(zip(
                comp_info['number_of_res'],
                comp_info['weight'],
                comp_info['amino_acids'],
                comp_info['tetras_4_res'],
                comp_info['tetras_3_res']
            )):
                # Calculate average residue number for sorting
                res_numbers = [int(res[0].split()[1]) for res in amino_acids]
                avg_res_number = sum(res_numbers) / len(res_numbers) if res_numbers else 0

                # Format amino acids as readable string
                amino_acids_str = ", ".join([f"{res[0].split()[0]}{res[0].split()[1]} {res[1]}" for res in amino_acids])

                rows.append({
                    'Component': idx + 1,
                    'Avg Residue #': avg_res_number,
                    'Number of Residues': num_res,
                    'Total Weight': weight,
                    'Tetras (4 res)': tetras_4,
                    'Tetras (3 res)': tetras_3,
                    'Amino Acids': amino_acids_str
                })

            # Create DataFrame and sort by average residue number
            df = pd.DataFrame(rows)
            try:
                df = df.sort_values('Avg Residue #').reset_index(drop=True)
                df['Component'] = range(1, len(df) + 1)  # Re-number components after sorting
            except:
                print("Error sorting DataFrame, exit"
                        "df before sorting:")
                print(df)
                return df

            # Display the table
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)

            if print_info:
                print(df.to_string(index=False))
                print(f"\nTotal connected components: {len(df)}\n")
        return df

    def plot_dof_distribution(self):

        from scipy.optimize import curve_fit# Prepare data for fitting

        # Créer la figure avec 2 lignes et num_cols colonnes
        fig, axes = plt.subplots(nrows=2, ncols=self.num_cols, figsize=(6 * self.num_cols, 12))

        # Store fit parameters for each PDB
        fit_params = {}

        # Flatten axes
        axes = axes.flatten()

        
        for idx, pdb in enumerate(self.protein_list):
            ax = axes[idx]

            # Count occurrences (like in your histogram)
            counts = np.bincount(self.tetras_dof[pdb], minlength=21)

            ax.bar(range(21), counts, color='#278BF5', edgecolor='black', linewidth=0.5, label='Data')
            bars = ax.bar(range(21), counts, color='#278BF5', edgecolor='black', linewidth=0.5, label='Data')
            ax.bar_label(bars, fmt='%d', padding=3)

            # Add vertical line at x=dof_threshold
            thold = self.dof_threshold[pdb]
            ax.axvline(x=thold, color='green', linestyle='--', linewidth=2, label=f'DOF = {thold}')

            # Calculate proportion of tetras with DOF >= thold
            total_tetras = counts.sum()
            tetras_above_thold = counts[thold:].sum()
            proportion = (tetras_above_thold / total_tetras) * 100 if total_tetras > 0 else 0

            # Display proportion to the right of the vertical line
            ax.text(12.5, ax.get_ylim()[1] * 0.7, f'DOF ≥ {thold}:\n{tetras_above_thold}/{total_tetras}\n({proportion:.1f}%)',
                    fontsize=10, color='green', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

            # Add fit parameters
            ax.set_xlabel('Degrees of Freedom (DOF)', fontsize=12)
            ax.set_ylabel('Number of tetrahedra', fontsize=12)
            ax.set_title(f'Distrib. of tetrahedra dof + exp fit - {pdb}', fontsize=14)
            ax.legend(loc='upper center', fontsize=9)
        self.fit_params = fit_params
           

        # Hide unused axes
        for idx in range(len(self.protein_list), len(axes)):
            axes[idx].set_visible(False)

            plt.tight_layout()
            plt.show()

    def visualize_high_sdm_tetras(self, number_of_at_in_diff_res = 4, filter = False, negative_control = False):
        """Exports the tetras with proba > threshold to vtu files for paraview visualization.
        Args:
            threshold (float): The probability threshold for exporting tetras. Tetras with a probability of having a dof sum above the threshold will be exported.
            to choose from [5,3,2]% of the distribution.
        """
        import os

        
        for pdb in self.protein_list:
            threshold = self.dof_threshold[pdb]
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Outputs",pdb)
            os.makedirs(output_dir, exist_ok=True)
            name_txt = f"MCN-res_{threshold}_{pdb}_{number_of_at_in_diff_res}_filtered" if filter else f"tet_dof_sup_{threshold}_{pdb}_{number_of_at_in_diff_res}"  
            name_obj = f"MCN-tet_{threshold}_{pdb}_{number_of_at_in_diff_res}_filtered" if filter else f"tet_dof_sup_{threshold}_{pdb}_{number_of_at_in_diff_res}"  
            # Create paths for output files
            vtu_filepath = os.path.join(output_dir, f"{name_obj}.obj")
            txt_filepath = os.path.join(output_dir, f"{name_txt}.txt")

            # -----
            tetras_to_export = self.tetras_list[pdb][number_of_at_in_diff_res]
            residues_involved = self.corresp_res_in_tetra[pdb][number_of_at_in_diff_res]
            if tetras_to_export:
                self.PA_proteins[pdb].delaunay_to_vtu(filtered=True, filtered_simplices= tetras_to_export, filename= vtu_filepath)


            

            else:
                print(f"No tetrahedra with DOF >= {threshold} for {pdb}.")
            
            str_to_save =""

            residues_involved = set(res for tet in residues_involved for res in tet )  # Flatten the list of residues in tetras and get unique ones

            # Group residue sequence IDs by chain
            chain_residues = defaultdict(set)
            for res in residues_involved:
                resi = res[0].split()[1]  # sequence ID
                resc = res[1]             # chain ID
                chain_residues[resc].add(resi)

            # Build the string: "/A:1,2,3,4 /B:5,6,7"
            str_to_save = " ".join(
                f"/{chain}:{','.join(sorted(residues, key=int))}"
                for chain, residues in sorted(chain_residues.items())
            )
            with open(txt_filepath, "w") as f:
                f.write(str_to_save)

    def filter_components_by_size(self, size_threshold=4): 
        """Filters out components with weight < threshold and removes corresponding tetrahedra.

        Args:
            size_threshold (int): The size threshold for filtering components. Components with number of residues < size_threshold will be filtered out.
                                  THIS SIZE THRESHOLD IS AN IMPORTANT PARAMETER !!

        Updates:
            self.res_involved: Removes residues that belong to filtered components
            self.tetras_list: Removes tetrahedra containing filtered residues
            self.corresp_res_in_tetra: Removes residue correspondences for filtered tetrahedra
        """

        for pdb in self.protein_list:
            # Step 1: Identify amino acids in components with weight < threshold
            residues_to_filter = set()

            comp_info = self.prot_sdof_carac[pdb]
            for num_res, amino_acids in zip(comp_info['number_of_res'],comp_info['amino_acids']):

                if num_res <= size_threshold:
                    # Add all amino acids from this component to the filter set
                    residues_to_filter.update(amino_acids)

            # Step 2: Filter tetrahedra by removing those containing filtered residues
            for number_of_at_in_diff_res in np.array([self.tet_in_diff_res]).flatten().tolist():
                tetras_to_keep = []
                corresp_res_to_keep = []

                for tetra, res_in_tetra in zip(
                    self.tetras_list[pdb][number_of_at_in_diff_res],
                    self.corresp_res_in_tetra[pdb][number_of_at_in_diff_res]):
                    # Check if any residue in this tetra is in the filter set
                    has_filtered_residue = any(res in residues_to_filter for res in res_in_tetra)

                    if not has_filtered_residue:
                        tetras_to_keep.append(tetra)
                        corresp_res_to_keep.append(res_in_tetra)

                # Update tetras_list and corresp_res_in_tetra
                self.tetras_list[pdb][number_of_at_in_diff_res] = tetras_to_keep
                self.corresp_res_in_tetra[pdb][number_of_at_in_diff_res] = corresp_res_to_keep


            # Step 3: Update res_involved by removing filtered residues
            self.res_involved[pdb] = self.res_involved[pdb] - residues_to_filter



    def export_list_as_txt(self, data_list, filename):
        """Exports a list of data to a text file.

        Args:
            data_list (list): The list of data to be exported.
            filename (str): The name of the output text file.
        """
        with open(filename, 'w') as f:
            for item in data_list:
                f.write(f"{item}\n")

    def define_threshold(self, verbose = False, export_data = False):
        """Calculates the sdm threshold based on the diierence in the distributions btwn the protein and a random distriburtion."""

        def calc_rand_distrib(dom_dict, tet_num):

            I = tet_num  # number of repeated experiments

            counts = [val for val in dom_dict.values()]
            print(dom_dict)

            # =========================================================
            # TOTAL NUMBER OF BALLS
            # =========================================================

            N = sum(counts)

            # total number of possible 4-ball draws
            total_draws = math.comb(N, 4)

            # =========================================================
            # COMPUTE EXACT DISTRIBUTION
            # =========================================================

            # result[score] = number of favorable combinations
            result = defaultdict(int)

            # x0+x1+x2+x3+x4+x5 = 4 (I draw 4 balls(=atoms))
            # xi = number of balls of weight i

            for x0 in range(5):
                for x1 in range(5 - x0):
                    for x2 in range(5 - x0 - x1):
                        for x3 in range(5 - x0 - x1 - x2):
                            for x4 in range(5 - x0 - x1 - x2 - x3):
                            
                                x5 = 4 - x0 - x1 - x2 - x3 - x4

                                xs = [x0, x1, x2, x3, x4, x5]

                                # check enough balls exist
                                valid = True
                                for i in range(6):
                                    if xs[i] > counts[i]:
                                        valid = False
                                        break
                                    
                                if not valid:
                                    continue
                                
                                # compute score
                                score = sum(i * xs[i] for i in range(6))

                                # number of ways to realize this pattern
                                ways = 1
                                for i in range(6):
                                    ways *= math.comb(counts[i], xs[i])

                                result[score] += ways

            # =========================================================
            # DISPLAY RESULTS
            # =========================================================

            scores = list(range(21))
            expected_counts = []

            p_aggregate = 0
            for s in scores:
            
                p = result[s] / total_draws
                p_aggregate += p

                expected = I * p

                expected_counts.append(int(expected))

            return expected_counts
        
        def plot_rand_hist(expected_counts):
            scores = list(range(21))
            plt.figure(figsize=(10, 5))

            # plt.bar(scores, expected_counts)
            bars = plt.bar(scores, expected_counts)
            plt.bar_label(bars, fmt='%.0f', padding=3)

            plt.xlabel("SDM")
            plt.ylabel(f"Expected SDM occurency")
            plt.title("Expected number of tetra for each SDM")

            plt.xticks(scores)

            plt.show()

        def export_expected_counts(expected_counts, filename='expected_counts_sdm.txt'):
            """
            Exports expected_counts (list or array) to a tab-separated file.
            Columns: SDM  ExpectedFloat  ExpectedInt
            Lines beginning with '%' are treated as comments in TeX.
            """
            with open(filename,'w',encoding='utf-8') as f:
                f.write('% SDM\tExpectedInt\n')
                for s,val in enumerate(expected_counts):
                    f.write(f'{s}\t{int(round(val))}\n')
            print(f'Wrote {len(expected_counts)} rows to {filename}')

        dom_threshold = {}
        all_dom_prot = {}

        for pdb in self.protein_list:
            list_dict_chain = [Counter(dom_chain) for dom_chain in self.dof_detail[pdb].values()]
            total_counter = Counter()
            for c in list_dict_chain:
                total_counter.update(c)
            all_dom_prot[pdb] = dict(total_counter)
            number_of_tet = len(self.tetras_by_prot[pdb])
            random_counts = calc_rand_distrib(all_dom_prot[pdb], number_of_tet)

            if verbose:
                plot_rand_hist(random_counts)
            if export_data:
                export_expected_counts(random_counts, filename=f'{pdb}_expected_counts_rand_distrib.txt')

            prot_counts = np.bincount(self.tetras_dof[pdb], minlength=21)

            diff = np.zeros(len(random_counts))
            for i, (a_v, b_v) in enumerate(zip(prot_counts,random_counts)):
               if b_v == 0:
                  diff[i] = 0
               else:
                  diff[i] = np.round(a_v/b_v, 2)

            dom_threshold[pdb] = next(i for i in range(5, len(diff) - 1)
                                      if diff[i] >= 2.2 and diff[i + 1] >= 2.2)


        return dom_threshold

    def plot_aa_distrib_involved(self, export_as_txt = False):
        from collections import Counter

        res_involved =set()
        with open('structure_database.json', 'r') as f:
            aa_db = json.load(f)
        seq_in_tet = ""


        for pdb in self.protein_list:
            for i in [3,4]:
                residues_involved_list = self.corresp_res_in_tetra[pdb][i]
                residues_involved_set  = set(res for tet in residues_involved_list for res in tet )  # Flatten the list of residues in tetras and get unique ones
                res_involved.update(residues_involved_set)

            for res in res_involved:
                resn = res[0].split()[0]  # sequence ID
                
                seq_in_tet += aa_db[resn].get('1_letter_code', [])

            plot_aa_distribution(self.PA_proteins[pdb].sequence, seq_in_tet, pdb)

            if export_as_txt:
                prot_seq_count = Counter(self.PA_proteins[pdb].sequence)
                print(prot_seq_count)
                export_sequence(prot_seq_count, filename = f'{pdb[:4]}_prot_comp.txt')
                swarm_seq_count = Counter(seq_in_tet)
                export_sequence(swarm_seq_count, filename = f'{pdb[:4]}_swarm-res_comp.txt')












def plot_aa_distribution(sequence_pdb, sequence_tetra, pdb):
    from collections import Counter
    import matplotlib.pyplot as plt

    counts_pdb = sorted(sequence_pdb, key=str.lower)
    counts_pdb = Counter(counts_pdb)
    counts_tet = sorted(sequence_tetra, key=str.lower)
    counts_tet = Counter(counts_tet)

    plt.grid(which = 'major', axis='y')
    plt.grid(which = 'minor', axis='y', linestyle = '--')
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=False, top=False)
    plt.bar(counts_pdb.keys(), counts_pdb.values(), label = 'Sequence of the protein')
    plt.bar(counts_tet.keys(), counts_tet.values(), label = 'Residues involved in tetras')
    plt.xlabel("Amino acid")
    plt.ylabel("Occurrences")
    plt.ylim(top=50)
    plt.title(f"Distribution of amino acids in {pdb}")
    plt.legend()
    plt.show()


def export_sequence(sequence_counter, filename='res_conter.txt'):
    """
    Exports sequence_counter (list or array) to a tab-separated file.
    Columns: SDM  ExpectedFloat  ExpectedInt
    Lines beginning with '%' are treated as comments in TeX.
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    with open(filename,'w',encoding='utf-8') as f:
        f.write("count aa\n")
        for aa in amino_acids:
            f.write(f"{sequence_counter.get(aa, 0)} {aa}\n")
        print(f'Wrote {len(sequence_counter)} rows to {filename}')