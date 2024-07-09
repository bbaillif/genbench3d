import pickle
import yaml
import logging
import argparse
import os


from rdkit import Chem
from genbench3d import GenBench3D
from genbench3d.conf_ensemble import ConfEnsembleLibrary
from genbench3d.data.source import CSDDrug, CrossDocked
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

# Benchmark CrossDocked training set
model_name = 'CrossDocked_train'
print(model_name, len(training_mols) / len(train_crossdocked.get_split()))
benchmark = GenBench3D(reference_geometry=reference_geometry,
                        config=config['genbench3d'])
results = benchmark.get_results_for_mol_list(mols=training_mols)
results_path = os.path.join(f'{results_dirpath}/results_{model_name}.p')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

# Benchmark CSD Drug subset
csd_drug_ligands = [mol for mol in source]
csd_drug_ligands_clean = preprocess_mols(csd_drug_ligands)
model_name = 'CSDDrug'
print(model_name, len(csd_drug_ligands_clean) / len(source.subset_csd_ids))
results = benchmark.get_results_for_mol_list(mols=csd_drug_ligands_clean)
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
benchmark.set_training_cel(training_cel)
results = benchmark.get_results_for_mol_list(mols=test_mols)
results_path = os.path.join(f'{results_dirpath}/results_{model_name}.p')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)