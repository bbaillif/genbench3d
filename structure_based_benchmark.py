import pickle
import os
import gzip
import logging

from copy import deepcopy
from rdkit import Chem
from tqdm import tqdm
from genbench3d import SBGenBench3D
from genbench3d.data import ComplexMinimizer
from genbench3d.data.structure import (Pocket, 
                                       VinaProtein, 
                                       Protein)

import torch

from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(funcName)s: %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    filemode='w',
                    filename='structure_based_benchmark.log', 
                    encoding='utf-8', 
                    level=logging.INFO)

cd_dirpath = '/home/bb596/hdd/crossdocked_v1.1_rmsd1.0/crossdocked_v1.1_rmsd1.0/'

minimized_path = '/home/bb596/hdd/ThreeDGenMolBenchmark/minimized/'
if not os.path.exists(minimized_path):
    os.mkdir(minimized_path)

targetdiff_path = '../hdd/ThreeDGenMolBenchmark/targetdiff/targetdiff_vina_docked.pt'
ligan_post_path = '../hdd/ThreeDGenMolBenchmark/targetdiff/cvae_vina_docked.pt'

targetdiff_results = torch.load(targetdiff_path, map_location='cpu')
ligan_post_results = torch.load(ligan_post_path, map_location='cpu')

# the i-th target is making the IFP script crash
# MDA cannot read/transform into rdmol
crashing_target_i = [26, 32, 38, 44, 45, 52, 65]
# crashing_target_i = []

i_to_ligand_filename = {}

logging.info('Compile generated mols')

# Gather TargetDiff generated mols and native ligands 
logging.info('TargetDiff')
targetdiff_gen_mols = {}
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
            
        targetdiff_gen_mols[ligand_filename] = gen_mols
        
# Gather LiGAN posterior generated mols
logging.info('LiGAN posterior')
ligan_post_gen_mols = {}
for i, target_results in enumerate(ligan_post_results):
    if i not in crashing_target_i: 
        gen_mols = []
        
        ligand_filename = target_results[0]['ligand_filename']
        
        for j, mol_results in enumerate(target_results):
            gen_mol = mol_results['mol']
            gen_mols.append(gen_mol)
            
        ligan_post_gen_mols[ligand_filename] = gen_mols
        
# Gather 3D-SBDD generated mols
logging.info('3D-SBDD')
three_d_sbdd_gen_mols = {}
three_d_sbdd_path = '/home/bb596/hdd/ThreeDGenMolBenchmark/AR/test_set'
for ligand_filename in targetdiff_gen_mols:
    gen_mols_filename = ligand_filename.replace('/', '/generated_').replace('.sdf', '_pocket10.pdb.sdf')
    gen_mols_filepath = os.path.join(three_d_sbdd_path, gen_mols_filename)
    gen_mols = [mol for mol in Chem.SDMolSupplier(gen_mols_filepath) if mol is not None]
    three_d_sbdd_gen_mols[ligand_filename] = gen_mols
    
# Gather Pocket2Mol generated mols
logging.info('Pocket2Mol')
pocket2mol_gen_mols = {}
pocket2mol_path = '/home/bb596/hdd/ThreeDGenMolBenchmark/Pocket2Mol/test_set'
for ligand_filename in targetdiff_gen_mols:
    gen_mols_filename = ligand_filename.replace('/', '/generated_').replace('.sdf', '_pocket10.pdb.sdf')
    gen_mols_filepath = os.path.join(pocket2mol_path, gen_mols_filename)
    gen_mols = [mol for mol in Chem.SDMolSupplier(gen_mols_filepath) if mol is not None]
    pocket2mol_gen_mols[ligand_filename] = gen_mols
    
# Gather DiffSBDD generated mols
logging.info('DiffSBDD')
diffsbdd_gen_mols = {}
diffsbdd_path = '/home/bb596/hdd/ThreeDGenMolBenchmark/DiffSBDD/crossdocked_fullatom_joint/'
for ligand_filename in targetdiff_gen_mols:
    suffix = ligand_filename.split('/')[-1]
    pocket_part = suffix.replace('.sdf', '_pocket10').replace('_', '-')
    gen_part = suffix[:-4].replace('_', '-') + '_gen.sdf'
    gen_mols_filename = '_'.join([pocket_part, gen_part])
    gen_mols_filepath = os.path.join(diffsbdd_path, gen_mols_filename)
    gen_mols = [mol for mol in Chem.SDMolSupplier(gen_mols_filepath) if mol is not None]
    diffsbdd_gen_mols[ligand_filename] = gen_mols
    
# Gather LiGAN prior generated mols
logging.info('LiGAN prior')
ligan_prior_gen_mols = {}
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
        
    ligan_prior_gen_mols[ligand_filename] = gen_mols
    
model_to_gen_mols = {
    'LiGAN (prior)': ligan_prior_gen_mols,
    'LiGAN (posterior)': ligan_post_gen_mols,
    '3D-SBDD': three_d_sbdd_gen_mols,
    'Pocket2Mol': pocket2mol_gen_mols,
    'TargetDiff': targetdiff_gen_mols,
    'DiffSBDD': diffsbdd_gen_mols,
}

d_results = {} # {target_name: {model_name: {rawOrProcessed: {metric: values_list}}}}

logging.info('Starting benchmark')
try:
    for i, ligand_filename in tqdm(list(i_to_ligand_filename.items())[2:]):
        
        try:
        
            logging.info(ligand_filename)
            
            d_target = {}
            
            native_ligand = native_ligands[ligand_filename]
            
            #ligand_filename is actually TARGET_NAME/ligand_filename.sdf
            target_dirname, real_ligand_filename = ligand_filename.split('/') 
            pdb_filename = f'{real_ligand_filename[:10]}.pdb'
            original_structure_path = os.path.join(cd_dirpath, 
                                                    target_dirname,
                                                    pdb_filename)
            
            protein = VinaProtein(pdb_filepath=original_structure_path)
            protein_clean = Protein(protein.protein_clean_filepath)
            pocket = Pocket(protein=protein_clean, 
                            native_ligand=native_ligand)
            
            sbgenbench3D = SBGenBench3D(protein,
                                        pocket,
                                        native_ligand)
            
            complex_minimizer = ComplexMinimizer(pocket)
            
            for model_name, ligand_filename_to_gen_mols in model_to_gen_mols.items():
            
                logging.info(model_name)
            
                gen_mols = ligand_filename_to_gen_mols[ligand_filename]
                gen_mols_h = [Chem.AddHs(mol, addCoords=True) for mol in gen_mols]
                results = sbgenbench3D.get_results_for_mol_list(mols=gen_mols_h,
                                                                n_total_mols=100)
                    
                d_model = {}
                d_model['raw'] = results
                
                # Minimize complexes
                clean_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
                minimized_target_path = os.path.join(minimized_path, target_dirname)
                if not os.path.exists(minimized_target_path):
                    os.mkdir(minimized_target_path)
                minimized_filename = 'generated_' + real_ligand_filename.replace('.sdf', 
                                                                            f'_{clean_model_name}_minimized.sdf')
                minimized_filepath = os.path.join(minimized_target_path,
                                                  minimized_filename)
                
                if not os.path.exists(minimized_filepath):
                    mini_gen_mols = []
                    for mol in gen_mols_h:
                        mini_mol = complex_minimizer.minimize_ligand(mol)
                        if mini_mol is not None:
                            mini_gen_mols.append(mini_mol)
                    logging.info(f'Saving minimized molecules in {minimized_filepath}')
                    with Chem.SDWriter(minimized_filepath) as writer:
                        for i, mol in enumerate(mini_gen_mols):
                            writer.write(mol)
                else:
                    logging.info(f'Loading minimized molecules from {minimized_filepath}')
                    mini_gen_mols = [mol for mol in Chem.SDMolSupplier(minimized_filepath)]
                    
                
                results = sbgenbench3D.get_results_for_mol_list(mols=mini_gen_mols,
                                                                n_total_mols=100)
                
                # import pdb;pdb.set_trace()
                
                d_model['minimized'] = results
                    
                d_target[model_name] = d_model
                
            d_results[ligand_filename] = d_target
        
        except Exception as e:
            logging.warning(f'Something went wrong: {e}')
            # import pdb;pdb.set_trace()
        
except KeyboardInterrupt:
    import pdb;pdb.set_trace()
    
with open('structure_based_benchmark_results.p', 'wb') as f:
    pickle.dump(d_results, f)