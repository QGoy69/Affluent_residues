import pickle
import os
from pymol import cmd
from chempy import Atom, Bond
from chempy.models import Indexed
import networkx as nx


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
NODE_RADIUS = 0.6              # sphere radius for nodes
EDGE_RADIUS = 0.2              # stick radius for edges
OBJECT_PREFIX = "graph_comp"   # graph_comp_001, graph_comp_002, ...
GRAY = (0.7, 0.7, 0.7)
# --------------------------------------------------


def get_pos(data):
    pos = data.get("pos")
    return pos if pos is not None else data.get("coord")


# --------------------------------------------------
# Load graph
# --------------------------------------------------
# ===== INSERT PDB NAME HERE ==========
pdb = '1gud.pdb'   #🟡🟡🟡🟡
# =====================================
file_name = f'{pdb[:4]}_tetras_viz_pymol_atoms.pkl'

data_dir = os.path.join(os.getcwd(), 'Outputs', pdb, file_name)

with open(data_dir, 'rb') as f:
    G = pickle.load(f)
print("Graph loaded:", G)

connected_comps = list(nx.connected_components(G))
print(f"Number of connected components: {len(connected_comps)}")

n_digits = max(3, len(str(len(connected_comps))))
obj_names = []


# --------------------------------------------------
# One molecular object per connected component
# --------------------------------------------------
for comp_idx, component in enumerate(connected_comps):
    subgraph = G.subgraph(component)

    color = GRAY

    model = Indexed()
    index_of = {}   # graph node -> atom index in this model

    # Nodes -> pseudoatoms
    for node, data in subgraph.nodes(data=True):
        pos = get_pos(data)
        if pos is None:
            print(f"Warning: no position for node {node}")
            continue

        a = Atom()
        a.symbol = "C"
        a.name = "N"
        a.resn = "GPH"
        a.resi = str(len(index_of) + 1)
        a.resi_number = len(index_of) + 1
        a.chain = "A"
        a.coord = [float(pos[0]), float(pos[1]), float(pos[2])]
        a.vdw = NODE_RADIUS
        a.hetatm = 1

        index_of[node] = len(model.atom)
        model.atom.append(a)

    if not model.atom:
        continue

    # Edges -> bonds
    for u, v, edata in subgraph.edges(data=True):
        if edata.get("artificial"):
            continue
        if u not in index_of or v not in index_of:
            continue
        b = Bond()
        b.index = [index_of[u], index_of[v]]
        b.order = 1
        model.bond.append(b)

    obj_name = f"{OBJECT_PREFIX}_{comp_idx + 1:0{n_digits}d}"
    cmd.load_model(model, obj_name)

    # Color
    col_name = f"{obj_name}_col"
    cmd.set_color(col_name, list(color))
    cmd.color(col_name, obj_name)

    # Representation
    cmd.hide("everything", obj_name)
    cmd.show("spheres", obj_name)
    cmd.show("sticks", obj_name)
    cmd.set("sphere_scale", 1.0, obj_name)
    cmd.set("stick_radius", EDGE_RADIUS, obj_name)

    obj_names.append(obj_name)

# Optional: group everything so it can be toggled as one
cmd.group("graph_components", " ".join(obj_names))

print(f"Loaded {len(obj_names)} component objects (group 'graph_components').")