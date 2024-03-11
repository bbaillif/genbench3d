import pickle
import os
import gzip
import logging
import pandas as pd

from rdkit import Chem
from genbench3d import GenBench3D

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

cd_dirpath = '/home/bb596/hdd/crossdocked_v1.1_rmsd1.0/crossdocked_v1.1_rmsd1.0/'

minimized_dirpath = '/home/bb596/hdd/ThreeDGenMolBenchmark/minimized/'

pocket2mol_path = '/home/bb596/hdd/ThreeDGenMolBenchmark/Pocket2Mol/test_set'
resgen_path = '/home/bb596/hdd/ThreeDGenMolBenchmark/ResGen/test_set'

cross_docked_path = '/home/bb596/hdd/CrossDocked/'
cross_docked_data_path = os.path.join(cross_docked_path, 'crossdocked_pocket10/')
split_path = os.path.join(cross_docked_path, 'split_by_name.pt')
train_ligand_path = '../hdd/ThreeDGenMolBenchmark/train_ligand_cd.sdf'

train_ligands = [mol for mol in Chem.SDMolSupplier(train_ligand_path)]

targetdiff_path = '../hdd/ThreeDGenMolBenchmark/targetdiff/targetdiff_vina_docked.p'
ligan_post_path = '../hdd/ThreeDGenMolBenchmark/targetdiff/cvae_vina_docked.p'

# The results loading will be replaced to avoid using torch.
with open(targetdiff_path, 'rb') as f:
    targetdiff_results = pickle.load(f)
with open(ligan_post_path, 'rb') as f:
    ligan_post_results = pickle.load(f)

# the i-th target is making the IFP script crash
# MDA cannot read/transform into rdmol
crashing_target_i = [26, 32, 38, 44, 45, 52, 65]
# crashing_target_i = []

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
        
        mini_mols_path = os.path.join(minimized_dirpath, 
                                      ligand_filename.replace('/', '/generated_').replace('.sdf', '_TargetDiff_minimized.sdf'))
        if os.path.exists(mini_mols_path):
            mini_mols = [mol for mol in Chem.SDMolSupplier(mini_mols_path, removeHs=False)]
            targetdiff_mini_mols.extend(mini_mols)
        else:
            print(mini_mols_path, 'does not exist')
        
        
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
        
        mini_mols_path = os.path.join(minimized_dirpath, 
                                      ligand_filename.replace('/', '/generated_').replace('.sdf', '_LiGAN_posterior_minimized.sdf'))
        if os.path.exists(mini_mols_path):
            mini_mols = [mol for mol in Chem.SDMolSupplier(mini_mols_path, removeHs=False)]
            ligan_post_mini_mols.extend(mini_mols)
        else:
            print(mini_mols_path, 'does not exist')
        
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
    
    mini_mols_path = os.path.join(minimized_dirpath, 
                                    ligand_filename.replace('/', '/generated_').replace('.sdf', '_3D-SBDD_minimized.sdf'))
    if os.path.exists(mini_mols_path):
        mini_mols = [mol for mol in Chem.SDMolSupplier(mini_mols_path, removeHs=False)]
        three_d_sbdd_mini_mols.extend(mini_mols)
    else:
        print(mini_mols_path, 'does not exist')
    
# Gather Pocket2Mol generated mols and minimized mols
logging.info('Pocket2Mol')
pocket2mol_gen_mols = []
pocket2mol_mini_mols = []
for i, ligand_filename in i_to_ligand_filename.items():
    gen_mols_filename = ligand_filename.replace('/', '/generated_').replace('.sdf', '_pocket10.pdb.sdf')
    gen_mols_filepath = os.path.join(pocket2mol_path, gen_mols_filename)
    gen_mols = [mol for mol in Chem.SDMolSupplier(gen_mols_filepath) if mol is not None]
    pocket2mol_gen_mols.extend(gen_mols)
    
    mini_mols_path = os.path.join(minimized_dirpath, 
                                    ligand_filename.replace('/', '/generated_').replace('.sdf', '_Pocket2Mol_minimized.sdf'))
    if os.path.exists(mini_mols_path):
        mini_mols = [mol for mol in Chem.SDMolSupplier(mini_mols_path, removeHs=False)]
        pocket2mol_mini_mols.extend(mini_mols)
    else:
        print(mini_mols_path, 'does not exist')
        
        
# Gather ResGen generated mols and minimized mols
logging.info('ResGen')
resgen_gen_mols = []
resgen_mini_mols = []
for i, ligand_filename in i_to_ligand_filename.items():
    gen_mols_filename = ligand_filename.replace('/', '/generated_').replace('.sdf', '_pocket10.pdb.sdf')
    gen_mols_filepath = os.path.join(resgen_path, gen_mols_filename)
    try:
        gen_mols = [mol for mol in Chem.SDMolSupplier(gen_mols_filepath) if mol is not None]
        resgen_gen_mols.extend(gen_mols)
        
        mini_mols_path = os.path.join(minimized_dirpath, 
                                        ligand_filename.replace('/', '/generated_').replace('.sdf', '_ResGen_minimized.sdf'))
        if os.path.exists(mini_mols_path):
            mini_mols = [mol for mol in Chem.SDMolSupplier(mini_mols_path, removeHs=False)]
            resgen_mini_mols.extend(mini_mols)
        else:
            print(mini_mols_path, 'does not exist')
    except:
        print(gen_mols_filepath, ' does not exist')
        
    
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
    
    mini_mols_path = os.path.join(minimized_dirpath, 
                                    ligand_filename.replace('/', '/generated_').replace('.sdf', '_DiffSBDD_minimized.sdf'))
    if os.path.exists(mini_mols_path):
        mini_mols = [mol for mol in Chem.SDMolSupplier(mini_mols_path, removeHs=False)]
        diffsbdd_mini_mols.extend(mini_mols)
    else:
        print(mini_mols_path, 'does not exist')
    
# Gather LiGAN prior generated mols
logging.info('LiGAN prior')
ligan_prior_gen_mols = []
ligan_prior_mini_mols = []
ligan_prior_path = '/home/bb596/hdd/ThreeDGenMolBenchmark/LiGAN/molecules/'
for i, ligand_filename in i_to_ligand_filename.items():
    gen_mols_filepath = os.path.join(ligan_prior_path, f'Generated_{i}_lig_gen_fit_add.sdf.gz')
    gz_stream = gzip.open(gen_mols_filepath)
    with Chem.ForwardSDMolSupplier(gz_stream) as gzsuppl:
        gen_mols = []
        for mol in gzsuppl:
            try:
                if (mol is not None) and (mol.GetNumAtoms() > 0) and (not '.' in Chem.MolToSmiles(mol)):
                    gen_mols.append(mol)
            except Exception as e:
                logging.warning(str(e))

        
    ligan_prior_gen_mols.extend(gen_mols)
    
    mini_mols_path = os.path.join(minimized_dirpath, 
                                    ligand_filename.replace('/', '/generated_').replace('.sdf', '_LiGAN_prior_minimized.sdf'))
    if os.path.exists(mini_mols_path):
        mini_mols = [mol for mol in Chem.SDMolSupplier(mini_mols_path, removeHs=False)]
        ligan_prior_mini_mols.extend(mini_mols)
    else:
        print(mini_mols_path, 'does not exist')
    
    
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
    'ResGen' : resgen_gen_mols,
    'ResGen minimized': resgen_mini_mols
}

d_results = {}

for model_name, gen_mols in model_to_gen_mols.items():
    print(model_name, len(gen_mols))

training_mols = []
for mol in train_ligands:
    if mol is not None:
        training_mols.append(Chem.AddHs(mol, addCoords=True))

model_name = 'CrossDocked (training)'
genbench3D = GenBench3D()
results = genbench3D.get_results_for_mol_list(mols=training_mols[:1000])
d_results[model_name] = results

dfs = []
for name, rows in genbench3D.validity3D_csd.validities.items():
    df = pd.DataFrame(rows)
    df['name'] = name
    dfs.append(df)
csd_df = pd.concat(dfs)

# dfs = []
# for name, rows in genbench3D.validity3D_crossdocked.validities.items():
#     df = pd.DataFrame(rows)
#     df['name'] = name
#     dfs.append(df)
# cd_df = pd.concat(dfs)

print(results['Validity3D CSD'])
# print(results['Validity3D CrossDocked'])
import pdb;pdb.set_trace()

# print(csd_df.sort_values('q-value'))
# print(cd_df.sort_values('q-value'))

for model_name, gen_mols in model_to_gen_mols.items():
    logging.info(model_name)
    genbench3D = GenBench3D(training_mols=training_mols)
    gen_mols_h = [Chem.AddHs(mol, addCoords=True) for mol in gen_mols]
    results = genbench3D.get_results_for_mol_list(gen_mols_h)
                                                  #, n_total_mols=10000)
    d_results[model_name] = results
    import pdb;pdb.set_trace()
    
with open('benchmark_results.p', 'wb') as f:
    pickle.dump(d_results, f)