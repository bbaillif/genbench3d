import pickle
import yaml
import logging
import argparse
import os

from tqdm import tqdm
from rdkit import Chem
from genbench3d import GenBench3D
from genbench3d.conf_ensemble import ConfEnsembleLibrary
from genbench3d.data.source import CSDDrug, CrossDocked
from genbench3d.sb_model import SBModel, LiGAN, ThreeDSBDD, Pocket2Mol, TargetDiff, DiffSBDD, ResGen
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
parser.add_argument("--config_path", 
                    default='config/default.yaml', 
                    type=str,
                    help="Path to config file.")
args = parser.parse_args()

config = yaml.safe_load(open(args.config_path, 'r'))

results_dirpath = config['results_dir']
if not os.path.exists(results_dirpath):
    os.mkdir(results_dirpath)

minimized_path = config['data']['minimized_path']
if not os.path.exists(minimized_path):
    os.mkdir(minimized_path)

train_crossdocked = CrossDocked(root=config['benchmark_dirpath'],
                                config=config['data'],
                                subset='train')
train_ligands = train_crossdocked.get_ligands()
training_mols = preprocess_mols(train_ligands)

source = CSDDrug(subset_path=config['data']['csd_drug_subset_path'])
reference_geometry = ReferenceGeometry(source=source,
                                       root=config['benchmark_dirpath'],
                                       minimum_pattern_values=config['genbench3d']['minimum_pattern_values'],)

benchmark = GenBench3D(reference_geometry=reference_geometry,
                        config=config['genbench3d'])

training_cel = ConfEnsembleLibrary.from_mol_list(training_mols)

training_mols_h = [Chem.AddHs(mol) for mol in training_mols]
training_cel_h = ConfEnsembleLibrary.from_mol_list(training_mols_h)
    
with open('test_set/ligand_filenames.txt', 'r') as f:
    lines = f.readlines()
all_ligand_filenames = [ligand_filename.strip() for ligand_filename in lines]

with open('test_set/ligand_filenames_subset.txt', 'r') as f:
    lines = f.readlines()
ligand_filenames_subset = [ligand_filename.strip() for ligand_filename in lines]
# ligand_filenames = test_crossdocked.get_ligand_filenames()

models: list[SBModel] = [
                        LiGAN(gen_path=config['models']['ligan_gen_dirpath'],
                              minimized_path=config['data']['minimized_path'],
                              ligand_filenames=all_ligand_filenames),
                        ThreeDSBDD(gen_path=config['models']['threedsbdd_gen_dirpath'],
                                   minimized_path=config['data']['minimized_path']),
                        Pocket2Mol(gen_path=config['models']['pocket2mol_gen_dirpath'],
                                   minimized_path=config['data']['minimized_path']),
                        TargetDiff(results_path=config['models']['targetdiff_results_filepath'],
                                   minimized_path=config['data']['minimized_path']),
                        DiffSBDD(gen_path=config['models']['diffsbdd_gen_dirpath'],
                                 minimized_path=config['data']['minimized_path']),
                        ResGen(gen_path=config['models']['resgen_gen_dirpath'],
                               minimized_path=config['data']['minimized_path'])
                        ]

# Benchmark models
minimizes = [False, True]
for minimize in minimizes:
    for model in tqdm(models):
        logging.info(model.name)
        
        # Compile generated molecules for all targets
        all_gen_mols = []
        n_total_mols = 0
        for ligand_filename in tqdm(ligand_filenames_subset):
            target_dirname, real_ligand_filename = ligand_filename.split('/')
            pdb_filename = f'{real_ligand_filename[:10]}.pdb'
            original_structure_path = os.path.join(config['data']['test_set_path'], 
                                                   target_dirname,
                                                   pdb_filename)
            native_ligand_path = os.path.join(config['data']['test_set_path'], 
                                            target_dirname,
                                            real_ligand_filename)
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
            
            gen_mols = model.get_generated_molecules(ligand_filename)
            n_total_mols += len(gen_mols)
            gen_mols = preprocess_mols(gen_mols)
            if len(gen_mols) > 0:
                gen_mols_h = [Chem.AddHs(mol, addCoords=True) for mol in gen_mols]
                if minimize:
                    gen_mols = model.get_minimized_molecules(ligand_filename,
                                                            gen_mols_h,
                                                            complex_minimizer)
                else:
                    gen_mols = gen_mols_h
                all_gen_mols.extend(gen_mols)
            
        print(model.name, len(all_gen_mols) / n_total_mols)
            
        # Run benchmark
        if minimize:
            benchmark.set_training_cel(training_cel_h)
        else:
            benchmark.set_training_cel(training_cel)
        results = benchmark.get_results_for_mol_list(all_gen_mols,
                                                      n_total_mols=n_total_mols)
        
        # Save results
        if minimize:
            results_path = os.path.join(results_dirpath,
                                        f'results_{model.name}_minimized.p')
        else:
            results_path = os.path.join(results_dirpath,
                                        f'results_{model.name}.p')
            
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)