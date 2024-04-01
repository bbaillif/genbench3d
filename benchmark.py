import pickle
import os
import gzip
import logging
import pandas as pd

from tqdm import tqdm
from rdkit import Chem
from genbench3d import GenBench3D
from genbench3d.conf_ensemble import ConfEnsembleLibrary
from genbench3d.data.source import CSDDrug, CrossDocked
from genbench3d.sb_model import SBModel, LiGAN, ThreeDSBDD, Pocket2Mol, TargetDiff, DiffSBDD, ResGen
from genbench3d.data.structure import Protein, Pocket
from genbench3d.data import ComplexMinimizer

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

train_crossdocked = CrossDocked(subset='train')
train_ligands = train_crossdocked.get_ligands()
training_mols = train_ligands

# model_name = 'CrossDocked'
# genbench3D = GenBench3D()
# results = genbench3D.get_results_for_mol_list(mols=training_mols)
# with open(f'results/results_{model_name}.p', 'wb') as f:
#     pickle.dump(results, f)

# csd_drug = CSDDrug()
# csd_drug_ligands = [mol for mol in csd_drug]
# model_name = 'CSDDrug'
# genbench3D = GenBench3D()
# results = genbench3D.get_results_for_mol_list(mols=csd_drug_ligands)
# with open(f'results/results_{model_name}.p', 'wb') as f:
#     pickle.dump(results, f)

# training_cel = ConfEnsembleLibrary.from_mol_list(training_mols)

training_mols_h = [Chem.AddHs(mol) for mol in training_mols]
training_cel_h = ConfEnsembleLibrary.from_mol_list(training_mols_h)

test_crossdocked = CrossDocked(subset='test')
ligand_filenames = test_crossdocked.get_ligand_filenames()

models: list[SBModel] = [
                        LiGAN(),
                        ThreeDSBDD(),
                        Pocket2Mol(),
                        TargetDiff(),
                        DiffSBDD(),
                        ResGen()
                        ]

# minimizes = [False, True]
minimizes = [True]
for minimize in minimizes:
    for model in tqdm(models):
        logging.info(model.name)
        all_gen_mols = []
        for ligand_filename in tqdm(ligand_filenames):
            original_structure_path = test_crossdocked.get_original_structure_path(ligand_filename)
            native_ligand = test_crossdocked.get_native_ligand(ligand_filename)
            native_ligand = Chem.AddHs(native_ligand, addCoords=True)
            
            protein = Protein(pdb_filepath=original_structure_path)
            pocket = Pocket(pdb_filepath=protein.protein_clean_filepath, 
                            native_ligand=native_ligand)
            complex_minimizer = ComplexMinimizer(pocket)
            
            gen_mols = model.get_generated_molecules(ligand_filename)
            if len(gen_mols) > 0:
                if minimize:
                    gen_mols_h = [Chem.AddHs(mol) for mol in gen_mols]
                    gen_mols = model.get_minimized_molecules(ligand_filename,
                                                            gen_mols_h,
                                                            complex_minimizer)
                all_gen_mols.extend(gen_mols)
            
        genbench3D = GenBench3D()
        if minimize:
            genbench3D.set_training_cel(training_cel_h)
        # else:
            # genbench3D.set_training_cel(training_cel)
        results = genbench3D.get_results_for_mol_list(all_gen_mols,
                                                      n_total_mols=10000)
        
        if minimize:
            results_path = f'results/results_{model.name}_minimized.p'
        else:
            results_path = f'results/results_{model.name}.p'
            
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)