# Affluent_residues
Identifies residues involved in energy flow through their side-chain in a protein, from the static structure of the protein (pdb structure).

## 1. Purpose

This code aims at identifying residues in a protein structure which are involved in energy flow. The method focuses on energy flow through side-chain. Residues involved in energy flow combine close atomic neighbors and high atomic mobility. For more details on the algorithm principle, see " ...". The algorithm select residues but is not able to specify their role in protein function.

The code outputs 2 kinds of files in [Outputs](/Outputs) folder. `.txt` files contain MCN residues ids and can be selected straightforwardly in ChimeraX for example. `.obj` files contain high *sum of degree of mobility* tetrahedra and can be visualized inb ChimeraX


## 2. Installation

  1. Clone the repository

  To clone the repository, run the following command in a terminal:

  ```Bash
    git clone https://github.com/QGoy69/Mobile_atoms_Close_Neighbors.git
  ```
  2. Install requirements

  Install in a new virtual environment the packages from `requirements.txt`

  ```Bash
    pip install -r requirements.txt
  ```

  3. Use the code

  Some examples of the code usage are provided in the notebook `MCN.ipynb`


## 3. How to use ?

The class `TetraAnalyser`from `tetra_analyser` module montains all the data to perform protein analysis. It has 2 arguments:

- `prot_info`: a list of pdb ids and chain ids on which the analysis is conducted. It is possible to pass several structures. The list syntax is as follows:
  ![image alt](https://github.com/QGoy69/Mobile_atoms_Close_Neighbors/blob/fd8693ab6b77eeb7f7f77162631c6aee24522eb1/prot_info.jpg)

- `fliter`: a boolean to filter (`=True`) or not (`=False`, default value) clusters of residues containing 4 or less residues

 📝 **Ex**: 

 ```
from modules import tetras_analyser
igps = tetra_analyser.TetraAnalyser([['1gpw.pdb',['A','B']]], filter = True)
```
Once the `TetraAnalyser` instance is created, one can access its attributes. In particular, `.final_df` prints a table containing clusters of residues and some of their characteristics. `.visualize_high_sdm_tetra` exports `.txt` and `.obj` files to visualize MCN residues in ChimeraX.

## How to cite ?

If you use this work, please consider citing "...".


