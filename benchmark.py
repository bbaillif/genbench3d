import pickle
import torch
import os
import gzip
import logging
import MDAnalysis as mda

from rdkit import Chem
from tqdm import tqdm
from genbench3d.genbench3d import GenBench3D
from utils.complex_minimizer import ComplexMinimizer

from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(filename='benchmark.log', encoding='utf-8', level=logging.DEBUG)

cd_dirpath = '/home/bb596/hdd/crossdocked_v1.1_rmsd1.0/crossdocked_v1.1_rmsd1.0/'

targetdiff_path = '../hdd/ThreeDGenMolBenchmark/targetdiff/targetdiff_vina_docked.pt'
ligan_post_path = '../hdd/ThreeDGenMolBenchmark/targetdiff/cvae_vina_docked.pt'
pocket2mol_path = '/home/bb596/hdd/ThreeDGenMolBenchmark/Pocket2Mol/test_set'

targetdiff_results = torch.load(targetdiff_path, map_location='cpu')
ligan_post_results = torch.load(ligan_post_path, map_location='cpu')

# the i-th target is making the IFP script crash
# MDA cannot read/transform into rdmol
crashing_target_i = [26, 32, 38, 44, 45, 52, 65]

i_to_ligand_filename = {}

logging.info('Compile generated mols')

# Gather TargetDiff generated mols, minimized mols and native ligands 
logging.info('TargetDiff')
targetdiff_gen_mols = []
targetdiff_mini_mols = []
native_ligands = {}
for i, target_results in enumerate(targetdiff_results):
    
    if i not in crashing_target_i: 
        gen_mols = []
        
        ligand_filename = target_results[0]['ligand_filename']
        i_to_ligand_filename[i] = ligand_filename

        native_ligand_filepath = os.path.join(cd_dirpath, ligand_filename)
        native_ligand = [mol for mol in Chem.SDMolSupplier(native_ligand_filepath, removeHs=False)][0]
        native_ligands[ligand_filename] = native_ligand
        
        for j, mol_results in enumerate(target_results):
            gen_mol = mol_results['mol']
            gen_mols.append(gen_mol)
            
        targetdiff_gen_mols.extend(gen_mols)
        
        mini_mols_path = os.path.join(pocket2mol_path, 
                                      ligand_filename.replace('.sdf', '_TargetDiff_mini.sdf'))
        if os.path.exists(mini_mols_path):
            mini_mols = [mol for mol in Chem.SDMolSupplier(mini_mols_path)]
            targetdiff_mini_mols.extend(mini_mols)
        
        
# Gather LiGAN posterior generated mols and minimized mols
logging.info('LiGAN posterior')
ligan_post_gen_mols = []
ligan_post_mini_mols = []
for i, target_results in enumerate(ligan_post_results):
    if i not in crashing_target_i: 
        gen_mols = []
        
        ligand_filename = target_results[0]['ligand_filename']
        
        for j, mol_results in enumerate(target_results):
            gen_mol = mol_results['mol']
            gen_mols.append(gen_mol)
            
        ligan_post_gen_mols.extend(gen_mols)
        
        mini_mols_path = os.path.join(pocket2mol_path, 
                                      ligand_filename.replace('.sdf', '_LiGAN_posterior_mini.sdf'))
        if os.path.exists(mini_mols_path):
            mini_mols = [mol for mol in Chem.SDMolSupplier(mini_mols_path)]
            ligan_post_mini_mols.extend(mini_mols)
        
# Gather 3D-SBDD generated mols and minimized mols
logging.info('3D-SBDD')
three_d_sbdd_gen_mols = []
three_d_sbdd_mini_mols = []
three_d_sbdd_path = '/home/bb596/hdd/ThreeDGenMolBenchmark/AR/test_set'
for i, ligand_filename in i_to_ligand_filename.items():
    gen_mols_filename = ligand_filename.replace('/', '/generated_').replace('.sdf', '_pocket10.pdb.sdf')
    gen_mols_filepath = os.path.join(three_d_sbdd_path, gen_mols_filename)
    gen_mols = [mol for mol in Chem.SDMolSupplier(gen_mols_filepath) if mol is not None]
    three_d_sbdd_gen_mols.extend(gen_mols)
    
    mini_mols_path = os.path.join(pocket2mol_path, 
                                    ligand_filename.replace('.sdf', '_3D-SBDD_mini.sdf'))
    if os.path.exists(mini_mols_path):
        mini_mols = [mol for mol in Chem.SDMolSupplier(mini_mols_path)]
        three_d_sbdd_mini_mols.extend(mini_mols)
    
# Gather Pocket2Mol generated mols and minimized mols
logging.info('Pocket2Mol')
pocket2mol_gen_mols = []
pocket2mol_mini_mols = []
for i, ligand_filename in i_to_ligand_filename.items():
    gen_mols_filename = ligand_filename.replace('/', '/generated_').replace('.sdf', '_pocket10.pdb.sdf')
    gen_mols_filepath = os.path.join(pocket2mol_path, gen_mols_filename)
    gen_mols = [mol for mol in Chem.SDMolSupplier(gen_mols_filepath) if mol is not None]
    pocket2mol_gen_mols.extend(gen_mols)
    
    mini_mols_path = os.path.join(pocket2mol_path, 
                                    ligand_filename.replace('.sdf', '_Pocket2Mol_mini.sdf'))
    if os.path.exists(mini_mols_path):
        mini_mols = [mol for mol in Chem.SDMolSupplier(mini_mols_path)]
        pocket2mol_mini_mols.extend(mini_mols)
    
# Gather DiffSBDD generated mols and minimized mols
logging.info('DiffSBDD')
diffsbdd_gen_mols = []
diffsbdd_mini_mols = []
diffsbdd_path = '/home/bb596/hdd/ThreeDGenMolBenchmark/DiffSBDD/crossdocked_fullatom_joint/'
for i, ligand_filename in i_to_ligand_filename.items():
    suffix = ligand_filename.split('/')[-1]
    pocket_part = suffix.replace('.sdf', '_pocket10').replace('_', '-')
    gen_part = suffix[:-4].replace('_', '-') + '_gen.sdf'
    gen_mols_filename = '_'.join([pocket_part, gen_part])
    gen_mols_filepath = os.path.join(diffsbdd_path, gen_mols_filename)
    gen_mols = [mol for mol in Chem.SDMolSupplier(gen_mols_filepath) if mol is not None]
    diffsbdd_gen_mols.extend(gen_mols)
    
    mini_mols_path = os.path.join(pocket2mol_path, 
                                    ligand_filename.replace('.sdf', '_DiffSBDD_mini.sdf'))
    if os.path.exists(mini_mols_path):
        mini_mols = [mol for mol in Chem.SDMolSupplier(mini_mols_path)]
        diffsbdd_mini_mols.extend(mini_mols)
    
# Gather LiGAN prior generated mols
logging.info('LiGAN prior')
ligan_prior_gen_mols = []
ligan_prior_mini_mols = []
ligan_prior_path = '/home/bb596/hdd/ThreeDGenMolBenchmark/LiGAN/molecules/'
for i, ligand_filename in i_to_ligand_filename.items():
    gen_mols_filepath = os.path.join(ligan_prior_path, f'Generated_{i}_lig_gen_fit_add.sdf.gz')
    gz_stream = gzip.open(gen_mols_filepath)
    with Chem.ForwardSDMolSupplier(gz_stream) as gzsuppl:
        mols = [mol 
                for mol in gzsuppl
                if (mol is not None) 
                    and (mol.GetNumAtoms() > 0)
                    and (not '.' in Chem.MolToSmiles(mol))]
        gen_mols = mols
        
    ligan_prior_gen_mols.extend(gen_mols)
    
    mini_mols_path = os.path.join(pocket2mol_path, 
                                    ligand_filename.replace('.sdf', '_LiGAN_prior_mini.sdf'))
    if os.path.exists(mini_mols_path):
        mini_mols = [mol for mol in Chem.SDMolSupplier(mini_mols_path)]
        ligan_prior_mini_mols.extend(mini_mols)
    
    
model_to_gen_mols = {
    'LiGAN (prior)': ligan_prior_gen_mols,
    'LiGAN (prior) minimized': ligan_prior_mini_mols,
    'LiGAN (posterior)': ligan_post_gen_mols,
    'LiGAN (posterior) minimized': ligan_post_mini_mols,
    '3D-SBDD': three_d_sbdd_gen_mols,
    '3D-SBDD minimized': three_d_sbdd_mini_mols,
    'Pocket2Mol': pocket2mol_gen_mols,
    'Pocket2Mol minimized': pocket2mol_mini_mols,
    'TargetDiff': targetdiff_gen_mols,
    'TargetDiff minimized': targetdiff_mini_mols,
    'DiffSBDD': diffsbdd_gen_mols,
    'DiffSBDD minimized': diffsbdd_mini_mols,
}

d_results = {}

for model_name, gen_mols in model_to_gen_mols.items():
    print(model_name, len(gen_mols))

for model_name, gen_mols in model_to_gen_mols.items():
    logging.info(model_name)
    metric_calculator = GenBench3D(show_plots=False)
    metric_calculator.n_total_mols = 10000
    metrics = metric_calculator.get_metrics_from_mol_list(gen_mols)
    d_results[model_name] = metrics
    
with open('benchmark_results.p', 'wb') as f:
    pickle.dump(d_results, f)