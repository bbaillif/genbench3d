import os
import argparse
import yaml

from tqdm import tqdm
from rdkit import Chem
from genbench3d.data.structure import (Pocket, 
                                       VinaProtein, 
                                       GlideProtein)
from genbench3d.sb_model import (SBModel,
                                 LiGAN,
                                 ThreeDSBDD,
                                 Pocket2Mol,
                                 DiffSBDD,
                                 TargetDiff,
                                 ResGen)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", 
                    default='config/default.yaml', 
                    type=str,
                    help="Path to config file.")
args = parser.parse_args()

config = yaml.safe_load(open(args.config_path, 'r'))

with open(os.path.join(config['test_set_dir'], 'ligand_filenames.txt'), 'r') as f:
    lines = f.readlines()
all_ligand_filenames = [ligand_filename.strip() for ligand_filename in lines]

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

ligand_filename_subset = []
for ligand_filename in tqdm(all_ligand_filenames):
    target_dirname, real_ligand_filename = ligand_filename.split('/')
    try:
        
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
        
        for model in models:
            gen_mols = model.get_generated_molecules(ligand_filename)
            if len(gen_mols) == 0:
                raise ValueError(f'No generated molecules for {model.name} {ligand_filename}')
        
        ligand_filename_subset.append(ligand_filename)
        
    except Exception as e:
        print(f'Error: {e}', ligand_filename)

with open(os.path.join(config['test_set_dir'], 'ligand_filenames_subset.txt'), 'w') as f:
    for ligand_filename in ligand_filename_subset:
        f.write(ligand_filename + '\n')