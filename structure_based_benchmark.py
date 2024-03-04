import warnings
warnings.filterwarnings("ignore")

import pickle
import os
import gzip
import logging

from rdkit import Chem
from tqdm import tqdm
from genbench3d import SBGenBench3D
from genbench3d.data import ComplexMinimizer
from genbench3d.data.structure import (Pocket, 
                                       VinaProtein, 
                                       GlideProtein,
                                       Protein)
from genbench3d.conf_ensemble import ConfEnsembleLibrary

from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(funcName)s: %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    filemode='w',
                    filename='structure_based_benchmark.log', 
                    encoding='utf-8', 
                    level=logging.INFO)

results_dirpath = '/home/bb596/hdd/ThreeDGenMolBenchmark/results/'
if not os.path.exists(results_dirpath):
    os.mkdir(results_dirpath)

cd_dirpath = '/home/bb596/hdd/crossdocked_v1.1_rmsd1.0/crossdocked_v1.1_rmsd1.0/'

minimized_path = '/home/bb596/hdd/ThreeDGenMolBenchmark/minimized/'
if not os.path.exists(minimized_path):
    os.mkdir(minimized_path)

targetdiff_path = '../hdd/ThreeDGenMolBenchmark/targetdiff/targetdiff_vina_docked.p'
ligan_post_path = '../hdd/ThreeDGenMolBenchmark/targetdiff/cvae_vina_docked.p'

# The results loading will be replaced to avoid using torch.
with open(targetdiff_path, 'rb') as f:
    targetdiff_results = pickle.load(f)
with open(ligan_post_path, 'rb') as f:
    ligan_post_results = pickle.load(f)

# the i-th target is making the IFP script crash
# MDA cannot read/transform into rdmol
# crashing_target_i = [26, 32, 38, 44, 45, 52, 65]
crashing_target_i = []

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
        mols = []
        for mol in gzsuppl:
            try:
                if (mol is not None) and (mol.GetNumAtoms() > 0) and (not '.' in Chem.MolToSmiles(mol)) :
                    mols.append(mol)
            except Exception as e:
                print(f'Mol not read exception: {e}')
        gen_mols = mols
        
    ligan_prior_gen_mols[ligand_filename] = gen_mols
    
    
# Gather ResGen generated mols
logging.info('ResGen')
resgen_gen_mols = {}
resgen_path = '/home/bb596/hdd/ThreeDGenMolBenchmark/ResGen/test_set'
for ligand_filename in targetdiff_gen_mols:
    gen_mols_filename = ligand_filename.replace('/', '/generated_').replace('.sdf', '_pocket10.pdb.sdf')
    gen_mols_filepath = os.path.join(resgen_path, gen_mols_filename)
    if os.path.exists(gen_mols_filepath):
        try:
            gen_mols = [mol for mol in Chem.SDMolSupplier(gen_mols_filepath) if mol is not None]
            resgen_gen_mols[ligand_filename] = gen_mols
        except:
            print('Wrong file ', ligand_filename)
    else:
        print(gen_mols_filepath, ' does not exists')
    
    
model_to_gen_mols = {
    'LiGAN (prior)': ligan_prior_gen_mols,
    # 'LiGAN (posterior)': ligan_post_gen_mols,
    '3D-SBDD': three_d_sbdd_gen_mols,
    'Pocket2Mol': pocket2mol_gen_mols,
    'TargetDiff': targetdiff_gen_mols,
    'DiffSBDD': diffsbdd_gen_mols,
    'ResGen': resgen_gen_mols
}

train_ligand_path = '../hdd/ThreeDGenMolBenchmark/train_ligand_cd.sdf'

train_ligands = [mol for mol in Chem.SDMolSupplier(train_ligand_path)][:10]
training_mols = []
for mol in train_ligands:
    if mol is not None:
        training_mols.append(Chem.AddHs(mol, addCoords=True))
training_cel = ConfEnsembleLibrary.from_mol_list(training_mols)

# d_results = {} # {target_name: {model_name: {rawOrProcessed: {metric: values_list}}}}

logging.info('Starting benchmark')
try:
    for i, ligand_filename in tqdm(list(i_to_ligand_filename.items())):

        try:
        
            logging.info(ligand_filename)
            
            # d_target = {}
            
            native_ligand = native_ligands[ligand_filename]
            native_ligand = Chem.AddHs(native_ligand, addCoords=True)
            
            #ligand_filename is actually TARGET_NAME/ligand_filename.sdf
            target_dirname, real_ligand_filename = ligand_filename.split('/') 
            pdb_filename = f'{real_ligand_filename[:10]}.pdb'
            original_structure_path = os.path.join(cd_dirpath, 
                                                    target_dirname,
                                                    pdb_filename)
            
            
            vina_protein = VinaProtein(pdb_filepath=original_structure_path)
            protein_clean = Protein(vina_protein.protein_clean_filepath)
            pocket = Pocket(protein=protein_clean, 
                            native_ligand=native_ligand)
            glide_protein = GlideProtein(pdb_filepath=vina_protein.protein_clean_filepath,
                                        native_ligand=native_ligand)

            complex_minimizer = ComplexMinimizer(pocket)
            
            minimized_target_path = os.path.join(minimized_path, target_dirname)
            if not os.path.exists(minimized_target_path):
                os.mkdir(minimized_target_path)
            
            minimized_filename = 'generated_' + real_ligand_filename.replace('.sdf', 
                                                                            f'_native_minimized.sdf')
            minimized_filepath = os.path.join(minimized_target_path,
                                                minimized_filename)
            
            if not os.path.exists(minimized_filepath):
            
                logging.info(f'Minimized native ligand in {minimized_filepath}')
                mini_native_ligand = complex_minimizer.minimize_ligand(native_ligand)
                with Chem.SDWriter(minimized_filepath) as writer:
                    writer.write(mini_native_ligand)
                
            else:
                
                logging.info(f'Loading minimized native ligand from {minimized_filepath}')
                mini_native_ligand = Chem.SDMolSupplier(minimized_filepath, removeHs=False)[0]
            
            sbgenbench3d = SBGenBench3D(vina_protein,
                                        glide_protein,
                                        pocket,
                                        native_ligand=mini_native_ligand)
            
            sbgenbench3d.set_training_cel(training_cel)
            
            for model_name, ligand_filename_to_gen_mols in model_to_gen_mols.items():
            
                results_filename = real_ligand_filename.replace('.sdf', 
                                                                f'_results_{model_name}.p')
                results_filepath = os.path.join(results_dirpath, results_filename)
            
                if not os.path.exists(results_filepath):
            
                    d_model = {}
                
                    logging.info(model_name)
                
                    gen_mols = ligand_filename_to_gen_mols[ligand_filename]
                    gen_mols_h = [Chem.AddHs(mol, addCoords=True) for mol in gen_mols]
                    
                    logging.info('raw')
                    results = sbgenbench3d.get_results_for_mol_list(mols=gen_mols_h,
                                                                    n_total_mols=100,
                                                                    do_conf_analysis=False)
                    
                    d_model['raw'] = results
                    
                    logging.info('raw_valid')
                    results = sbgenbench3d.get_results_for_mol_list(mols=gen_mols_h,
                                                                    n_total_mols=100,
                                                                    do_conf_analysis=True,
                                                                    valid_only=True)
                    
                    d_model['raw_valid'] = results
                    
                    # Minimize complexes
                    clean_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
                    
                    minimized_filename = 'generated_' + real_ligand_filename.replace('.sdf', 
                                                                                f'_{clean_model_name}_minimized.sdf')
                    minimized_filepath = os.path.join(minimized_target_path,
                                                    minimized_filename)
                    
                    if not os.path.exists(minimized_filepath):
                        mini_gen_mols = []
                        for mol_i, mol in enumerate(gen_mols_h):
                            logging.info(f'Minimizing molecule {mol_i}')
                            mini_mol = complex_minimizer.minimize_ligand(mol)
                            if mini_mol is not None:
                                mini_gen_mols.append(mini_mol)
                        logging.info(f'Saving minimized molecules in {minimized_filepath}')
                        with Chem.SDWriter(minimized_filepath) as writer:
                            for i, mol in enumerate(mini_gen_mols):
                                writer.write(mol)
                    else:
                        logging.info(f'Loading minimized molecules from {minimized_filepath}')
                        mini_gen_mols = [mol for mol in Chem.SDMolSupplier(minimized_filepath, 
                                                                        removeHs=False)]
                        
                    logging.info('minimized')
                    results = sbgenbench3d.get_results_for_mol_list(mols=mini_gen_mols,
                                                                    n_total_mols=100,
                                                                    do_conf_analysis=False)
                    
                    d_model['minimized'] = results
                    
                    logging.info('minimized_valid')
                    results = sbgenbench3d.get_results_for_mol_list(mols=mini_gen_mols,
                                                                    n_total_mols=100,
                                                                    do_conf_analysis=True,
                                                                    valid_only=True)
                    
                    d_model['minimized_valid'] = results
                
                    with open(results_filepath, 'wb') as f:
                        pickle.dump(d_model, f)
                
                # d_target[model_name] = d_model
                
            # d_results[ligand_filename] = d_target
        
        except Exception as e:
            logging.warning(f'Something went wrong for ligand {ligand_filename}: {e}')
        
except KeyboardInterrupt:
    import pdb;pdb.set_trace()
    
# with open('structure_based_benchmark_results.p', 'wb') as f:
#     pickle.dump(d_results, f)