import json
import yaml
import logging
import argparse
import numpy as np

from rdkit import Chem
from genbench3d import GenBench3D
from genbench3d.data.source import CSDDrug, CrossDocked, LigBoundConf
from genbench3d.data.structure import Protein, Pocket
from genbench3d.data import ComplexMinimizer
from genbench3d.utils import preprocess_mols
from genbench3d.geometry import ReferenceGeometry

from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(funcName)s: %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    filemode='w',
                    filename='benchmark.log', 
                    encoding='utf-8', 
                    level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_path", 
                    default='config/default.yaml', 
                    type=str,
                    help="Path to config file.")
parser.add_argument("-i", "--input_sdf", 
                    # default='examples/pocket2mol_generated_2z3h.sdf', 
                    type=str,
                    help="Path to sdf file containing molecules to benchmark.")
parser.add_argument("-o", "--output_json", 
                    # default='examples/results_pocket2mol_generated_2z3h.json', 
                    type=str,
                    help="Path to json file to store benchmark results.")
parser.add_argument("-s", "--source",
                    default='ligboundconf',
                    type=str,
                    help="Source of the reference geometry.",
                    choices=['csd_drug', 'crossdocked', 'ligboundconf'])
parser.add_argument("-m", "--minimize",
                    action='store_true',
                    help="Whether to minimize the molecules before benchmarking.")
parser.add_argument("-p", "--pdb_structure",
                    # default='test_set/BSD_ASPTE_1_130_0/2z3h_A_rec.pdb',
                    type=str,
                    help="PDB structure for the pocket used to generate the molecules")
parser.add_argument("-n", "--native_ligand_sdf",
                    # default='test_set/BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3.sdf',
                    help="Native ligand corresponding to the pocket used to generate the molecules")

args = parser.parse_args()

config = yaml.safe_load(open(args.config_path, 'r'))

if args.source == 'csd_drug':
    source = CSDDrug(subset_path=config['data']['csd_drug_subset_path'])
elif args.source == 'crossdocked':
    source = CrossDocked(root=config['benchmark_dirpath'],
                         config=config['data'],
                         subset='train')
elif args.source == 'ligboundconf':
    source = LigBoundConf(ligands_path=config['data']['ligboundconf_path'])
    
reference_geometry = ReferenceGeometry(source=source,
                                       root=config['benchmark_dirpath'],
                                       minimum_pattern_values=config['genbench3d']['minimum_pattern_values'],)

benchmark = GenBench3D(reference_geometry=reference_geometry,
                        config=config['genbench3d'])

if args.minimize:
    assert args.pdb_structure is not None, "PDB structure path is required for minimization."
    assert args.native_ligand_sdf is not None, "Native ligand path is required for minimization."
    original_structure_path = args.pdb_structure
    native_ligand_path = args.native_ligand_sdf
    native_ligand = [mol 
                        for mol in Chem.SDMolSupplier(native_ligand_path, 
                                                    removeHs=False)][0]
    native_ligand = Chem.AddHs(native_ligand, addCoords=True)

    protein = Protein(pdb_filepath=original_structure_path)
    pocket = Pocket(pdb_filepath=protein.protein_clean_filepath, 
                    native_ligand=native_ligand,
                    distance_from_ligand=config['pocket_distance_from_ligand'])
    complex_minimizer = ComplexMinimizer(pocket,
                                            config=config['minimization'])
    
gen_mols = Chem.SDMolSupplier(args.input_sdf, removeHs=False)
gen_mols = preprocess_mols(gen_mols) # Remove empty, None and fragmented RDKit molecules 
gen_mols_h = [Chem.AddHs(mol, addCoords=True) for mol in gen_mols]
if args.minimize:
    gen_mols = [complex_minimizer.minimize_ligand(mol) 
                for mol in gen_mols_h]
else:
    gen_mols = gen_mols_h
    
results = benchmark.get_results_for_mol_list(gen_mols)
    
with open(args.output_json, 'w') as f:
    json.dump(results, f)
    
summary = {}
for metric_name, values in results.items():
    if isinstance(values, dict): # e.g. Ring proportion
        for key, value in values.items():
            summary[metric_name + str(key)] = value
    elif isinstance(values, list):
        summary[metric_name] = np.nanmedian(values) # values can have nan
    else: # float or int
        summary[metric_name] = values
        
for metric_name, values in summary.items():
    print(f'{metric_name}: {np.around(values, 4)}')