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
parser.add_argument("config_path", 
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

source = CSDDrug(subset_path=config.csd_drug_subset_path)
reference_geometry = ReferenceGeometry(source=source,
                                       root=config['benchmark_dirpath'],
                                       minimum_pattern_values=config['minimum_pattern_values'],)

# Benchmark CrossDocked training set
model_name = 'CrossDocked_train'
print(model_name, len(training_mols) / len(train_crossdocked.get_split()))
genbench3D = GenBench3D(reference_geometry=reference_geometry,
                        config=config['genbench3d'])
results = genbench3D.get_results_for_mol_list(mols=training_mols)
results_path = os.path.join(f'{results_dirpath}/results_{model_name}.p')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

# Benchmark CSD Drug subset
csd_drug_ligands = [mol for mol in source]
csd_drug_ligands_clean = preprocess_mols(csd_drug_ligands)
model_name = 'CSDDrug'
print(model_name, len(csd_drug_ligands_clean) / len(source.subset_csd_ids))
results = genbench3D.get_results_for_mol_list(mols=csd_drug_ligands_clean)
results_path = os.path.join(f'{results_dirpath}/results_{model_name}.p')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

training_cel = ConfEnsembleLibrary.from_mol_list(training_mols)

training_mols_h = [Chem.AddHs(mol) for mol in training_mols]
training_cel_h = ConfEnsembleLibrary.from_mol_list(training_mols_h)

# Benchmark CrossDocked test set
model_name = 'CrossDocked_test'
test_crossdocked = CrossDocked(root=config['benchmark_dirpath'],
                                config=config['data'],
                                subset='test')
test_ligands = test_crossdocked.get_ligands()
test_mols = preprocess_mols(test_ligands)
print(model_name, len(test_mols) / len(test_crossdocked.get_split()))
genbench3D.set_training_cel(training_cel)
results = genbench3D.get_results_for_mol_list(mols=test_mols)
results_path = os.path.join(f'{results_dirpath}/results_{model_name}.p')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
    
ligand_filenames = test_crossdocked.get_ligand_filenames()

models: list[SBModel] = [
                        LiGAN(gen_path=config['models']['ligan_gen_dirpath'],
                              minimized_path=config['data']['minimized_path']),
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
        for ligand_filename in tqdm(ligand_filenames):
            original_structure_path = test_crossdocked.get_original_structure_path(ligand_filename)
            native_ligand = test_crossdocked.get_native_ligand(ligand_filename)
            native_ligand = Chem.AddHs(native_ligand, addCoords=True)
            
            protein = Protein(pdb_filepath=original_structure_path)
            pocket = Pocket(pdb_filepath=protein.protein_clean_filepath, 
                            native_ligand=native_ligand)
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
            genbench3D.set_training_cel(training_cel_h)
        else:
            genbench3D.set_training_cel(training_cel)
        results = genbench3D.get_results_for_mol_list(all_gen_mols,
                                                      n_total_mols=n_total_mols)
        
        # Save results
        if minimize:
            results_path = os.path.join(f'{results_dirpath}/results_{model.name}_minimized.p')
        else:
            results_path = os.path.join(f'{results_dirpath}/results_{model.name}.p')
            
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)