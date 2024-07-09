import warnings
warnings.filterwarnings("ignore")

import pickle
import os
import logging
import yaml
import argparse

from rdkit import Chem
from tqdm import tqdm
from genbench3d import SBGenBench3D
from genbench3d.data import ComplexMinimizer
from genbench3d.data.structure import (Pocket, 
                                       VinaProtein, 
                                       GlideProtein)
from genbench3d.conf_ensemble import ConfEnsembleLibrary
from genbench3d.sb_model import TargetDiff
from genbench3d.data.source import CrossDocked
from genbench3d.sb_model import (SBModel,
                                 LiGAN,
                                 ThreeDSBDD,
                                 Pocket2Mol,
                                 DiffSBDD,
                                 TargetDiff,
                                 ResGen)
from genbench3d.utils import preprocess_mols
from genbench3d.geometry import ReferenceGeometry
from genbench3d.data.source import CSDDrug

from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(funcName)s: %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    filemode='w',
                    filename='structure_based_benchmark.log', 
                    encoding='utf-8', 
                    level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", 
                    default='config/default.yaml', 
                    type=str,
                    help="Path to config file.")
args = parser.parse_args()

config = yaml.safe_load(open(args.config_path, 'r'))
overwrite = config['genbench3d']['overwrite_results']

results_dirpath = config['results_dir']
if not os.path.exists(results_dirpath):
    os.mkdir(results_dirpath)

minimized_path = config['data']['minimized_path']
if not os.path.exists(minimized_path):
    os.mkdir(minimized_path)

source = CSDDrug(subset_path=config['data']['csd_drug_subset_path'])
reference_geometry = ReferenceGeometry(source=source,
                                       root=config['benchmark_dirpath'],
                                       minimum_pattern_values=config['genbench3d']['minimum_pattern_values'],)

train_crossdocked = CrossDocked(root=config['benchmark_dirpath'],
                                config=config['data'],
                                subset='train')
train_ligands = train_crossdocked.get_ligands()
training_mols = [Chem.AddHs(mol, addCoords=True) for mol in train_ligands]
training_cel = ConfEnsembleLibrary.from_mol_list(training_mols)

test_crossdocked = CrossDocked(root=config['benchmark_dirpath'],
                                config=config['data'],
                                subset='test')
# ligand_filenames = test_crossdocked.get_ligand_filenames()
with open('test_set/ligand_filenames.txt', 'r') as f:
    ligand_filenames = f.readlines()
all_ligand_filenames = [ligand_filename.strip() for ligand_filename in ligand_filenames]
# ligand_filenames_subset = ['PLCD1_RAT_134_756_0/1djy_A_rec_1djz_ip2_lig_tt_min_0.sdf',
#                            'DPP2_HUMAN_27_492_0/3jyh_A_rec_3n0t_opy_lig_tt_min_0.sdf',
#                            ]

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

d_results = {} # {target_name: {model_name: {rawOrProcessed: {metric: values_list}}}}

minimizes = [False, True]

logging.info('Starting benchmark')
try:
    for ligand_filename in tqdm(all_ligand_filenames):
        target_dirname, real_ligand_filename = ligand_filename.split('/')
        try:
            logging.info(ligand_filename)
            
            d_target = {}
            
            # We need native_ligand to setup pockets for scoring
            # native_ligand = test_crossdocked.get_native_ligand(ligand_filename)
            native_ligand_path = os.path.join(config['data']['test_set_path'], 
                                            target_dirname,
                                            real_ligand_filename)
            native_ligand = [mol 
                             for mol in Chem.SDMolSupplier(native_ligand_path, 
                                                           removeHs=False)][0]
            native_ligand = Chem.AddHs(native_ligand, addCoords=True)
            
            # original_structure_path = test_crossdocked.get_original_structure_path(ligand_filename)
            pdb_filename = f'{real_ligand_filename[:10]}.pdb'
            original_structure_path = os.path.join(config['data']['test_set_path'], 
                                                   target_dirname,
                                                   pdb_filename)
            
            vina_protein = VinaProtein(pdb_filepath=original_structure_path,
                                       prepare_receptor_bin_path=config['bin']['prepare_receptor_bin_path'],)
            glide_protein = GlideProtein(pdb_filepath=vina_protein.protein_clean_filepath,
                                        native_ligand=native_ligand,
                                        glide_output_dirpath=config['glide_working_dir'],
                                        glide_path=config['bin']['glide_path'],
                                        structconvert_path=config['bin']['structconvert_path'],)
            pocket = Pocket(pdb_filepath=vina_protein.protein_clean_filepath, 
                            native_ligand=native_ligand,
                            distance_from_ligand=config['pocket_distance_from_ligand'])
            complex_minimizer = ComplexMinimizer(pocket,
                                                 config=config['minimization'])
            
            for minimize in minimizes:
            
                if minimize:
                    set_name = 'minimized'
                    # native_ligand = test_crossdocked.get_minimized_native(ligand_filename,
                    #                                                       complex_minimizer)
                    minimized_target_path = os.path.join(config['data']['minimized_path'], target_dirname)
                    if not os.path.exists(minimized_target_path):
                        os.mkdir(minimized_target_path)
                    minimized_filename = 'generated_' + real_ligand_filename.replace('.sdf', 
                                                                        f'_native_minimized.sdf')
                    minimized_filepath = os.path.join(minimized_target_path,
                                                        minimized_filename)
                    
                    # Minimize native ligand
                    if not os.path.exists(minimized_filepath):
                        logging.info(f'Saving minimized native ligand in {minimized_filepath}')
                        mini_native_ligand = complex_minimizer.minimize_ligand(native_ligand)
                        with Chem.SDWriter(minimized_filepath) as writer:
                            writer.write(mini_native_ligand)
                    else:
                        logging.info(f'Loading minimized native ligand from {minimized_filepath}')
                        mini_native_ligand = Chem.SDMolSupplier(minimized_filepath, removeHs=False)[0]
                        
                    native_ligand = mini_native_ligand
                else:
                    set_name = 'raw'
            
                sbgenbench3d = SBGenBench3D(reference_geometry=reference_geometry,
                                            config=config['genbench3d'],
                                            pocket=pocket,
                                            native_ligand=native_ligand)
                sbgenbench3d.setup_vina(vina_protein,
                                        config['vina'],
                                       add_minimized=True)
                sbgenbench3d.setup_glide(glide_protein,
                                         glide_path=config['bin']['glide_path'],
                                         add_minimized=True)
                sbgenbench3d.setup_gold_plp(vina_protein)
                sbgenbench3d.set_training_cel(training_cel)
            
                for model in tqdm(models):
                    
                    logging.info(model.name)
                    if minimize:
                        results_filename = real_ligand_filename.replace('.sdf', 
                                                                        f'_results_{model.name}_minimized.p')
                    else:
                        results_filename = real_ligand_filename.replace('.sdf', 
                                                                        f'_results_{model.name}.p')
                    results_filepath = os.path.join(results_dirpath, results_filename)
                    
                    if not os.path.exists(results_filepath) or overwrite:
                    
                        gen_mols = model.get_generated_molecules(ligand_filename)
                        n_total_mols = len(gen_mols)
                        gen_mols = preprocess_mols(gen_mols)
                        if len(gen_mols) > 0:
                            gen_mols_h = [Chem.AddHs(mol, addCoords=True) for mol in gen_mols]
                            if minimize:
                                gen_mols = model.get_minimized_molecules(ligand_filename,
                                                                        gen_mols_h,
                                                                        complex_minimizer)
                            else:
                                gen_mols = gen_mols_h
                    
                            d_model = {}
                        
                            logging.info(set_name)
                            results = sbgenbench3d.get_results_for_mol_list(mols=gen_mols,
                                                                            n_total_mols=n_total_mols,
                                                                            do_conf_analysis=False)
                            
                            d_model[set_name] = results
                            
                            # Valid only results
                            logging.info(f'{set_name}_valid')
                            results = sbgenbench3d.get_results_for_mol_list(mols=gen_mols,
                                                                            n_total_mols=n_total_mols,
                                                                            do_conf_analysis=True,
                                                                            valid_only=True)
                            d_model[f'{set_name}_valid'] = results
                        
                            with open(results_filepath, 'wb') as f:
                                pickle.dump(d_model, f)
        
        except Exception as e:
            print(str(e))
            print(type(e))
            logging.warning(f'Something went wrong for ligand {ligand_filename}: {e}')
            import pdb;pdb.set_trace()
        
except KeyboardInterrupt:
    import pdb;pdb.set_trace()
