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
from genbench3d.sb_model import TargetDiff
from genbench3d.params import (MINIMIZED_DIRPATH, 
                               OVERWRITE)
from genbench3d.data.source import CrossDocked
from genbench3d.sb_model import (SBModel,
                                 LiGAN,
                                 ThreeDSBDD,
                                 Pocket2Mol,
                                 DiffSBDD,
                                 TargetDiff,
                                 ResGen)

from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(funcName)s: %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    filemode='w',
                    filename='structure_based_benchmark_v2.log', 
                    encoding='utf-8', 
                    level=logging.INFO)

overwrite = OVERWRITE

results_dirpath = '/home/bb596/hdd/ThreeDGenMolBenchmark/results/'
if not os.path.exists(results_dirpath):
    os.mkdir(results_dirpath)

minimized_path = MINIMIZED_DIRPATH
if not os.path.exists(minimized_path):
    os.mkdir(minimized_path)

# train_crossdocked = CrossDocked(subset='train')
# train_ligands = train_crossdocked.get_ligands()
# training_mols = [Chem.AddHs(mol, addCoords=True) for mol in train_ligands]
# training_cel = ConfEnsembleLibrary.from_mol_list(training_mols)

test_crossdocked = CrossDocked(subset='test')
ligand_filenames = test_crossdocked.get_ligand_filenames()

models: list[SBModel] = [TargetDiff(),
                        LiGAN(),
                        ThreeDSBDD(),
                        Pocket2Mol(),
                        DiffSBDD(),
                        ResGen()]

d_results = {} # {target_name: {model_name: {rawOrProcessed: {metric: values_list}}}}

minimizes = [False, True]

logging.info('Starting benchmark')
try:
    for ligand_filename in tqdm(ligand_filenames):
        target_dirname, real_ligand_filename = ligand_filename.split('/')
        try:
            logging.info(ligand_filename)
            
            d_target = {}
            
            # We need native_ligand to setup pockets for scoring
            native_ligand = test_crossdocked.get_native_ligand(ligand_filename)
            native_ligand = Chem.AddHs(native_ligand, addCoords=True)
            
            original_structure_path = test_crossdocked.get_original_structure_path(ligand_filename)
            
            vina_protein = VinaProtein(pdb_filepath=original_structure_path)
            glide_protein = GlideProtein(pdb_filepath=vina_protein.protein_clean_filepath,
                                        native_ligand=native_ligand)
            pocket = Pocket(pdb_filepath=vina_protein.protein_clean_filepath, 
                            native_ligand=native_ligand)
            complex_minimizer = ComplexMinimizer(pocket)
            
            for minimize in minimizes:
            
                if minimize:
                    set_name = 'minimized'
                    native_ligand = test_crossdocked.get_minimized_native(ligand_filename,
                                                                          complex_minimizer)
                else:
                    set_name = 'raw'
            
                sbgenbench3D = SBGenBench3D(pocket,
                                            native_ligand)
                sbgenbench3D.setup_vina(vina_protein)
                sbgenbench3D.setup_glide(glide_protein)
                sbgenbench3D.setup_gold_plp(vina_protein)
                # sbgenbench3D.set_training_cel(training_cel)
            
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
                        if len(gen_mols) > 0:
                            if minimize:
                                mini_gen_mols = model.get_minimized_molecules(ligand_filename,
                                                                            gen_mols_h,
                                                                            complex_minimizer)
                                gen_mols_h = mini_gen_mols
                            else:
                                gen_mols_h = [Chem.AddHs(mol, addCoords=True) for mol in gen_mols]
                    
                            d_model = {}
                        
                            logging.info(set_name)
                            results = sbgenbench3D.get_results_for_mol_list(mols=gen_mols_h,
                                                                            n_total_mols=100,
                                                                            do_conf_analysis=False)
                            
                            d_model[set_name] = results
                            
                            # Valid only results
                            logging.info(f'{set_name}_valid')
                            results = sbgenbench3D.get_results_for_mol_list(mols=gen_mols_h,
                                                                            n_total_mols=100,
                                                                            do_conf_analysis=True,
                                                                            valid_only=True)
                            d_model[f'{set_name}_valid'] = results
                        
                            with open(results_filepath, 'wb') as f:
                                pickle.dump(d_model, f)
        
        except Exception as e:
            logging.warning(f'Something went wrong for ligand {ligand_filename}: {e}')
            # import pdb;pdb.set_trace()
        
except KeyboardInterrupt:
    import pdb;pdb.set_trace()
