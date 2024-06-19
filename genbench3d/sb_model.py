import pickle
import os
import logging
import gzip

from rdkit import Chem
from rdkit.Chem import Mol
from abc import ABC, abstractmethod
from genbench3d.params import (CROSSDOCKED_DATA_PATH, 
                               MINIMIZED_DIRPATH, 
                               TARGETDIFF_RESULTS_FILEPATH,
                               THREEDSBDD_GEN_DIRPATH,
                               POCKET2MOL_GEN_DIRPATH,
                               DIFFSBDD_GEN_DIRPATH,
                               LIGAN_GEN_DIRPATH,
                               RESGEN_GEN_DIRPATH
                               )
from genbench3d.data import ComplexMinimizer
from genbench3d.data.source import CrossDocked
from genbench3d.data.source.cross_docked import CROSSDOCKED_SUBSETS

class SBModel(ABC):
    
    def __init__(self,
                 name: str,
                 minimized_path: str = MINIMIZED_DIRPATH,
                 crossdocked_subset: str = 'test') -> None:
        assert crossdocked_subset in CROSSDOCKED_SUBSETS
        self.name = name
        self.minimized_path = minimized_path
        
        self.crossdocked = CrossDocked(subset=crossdocked_subset)
    
    @abstractmethod
    def get_generated_molecules(self,
                                 ligand_filename: str) -> list[Mol]:
        pass
    
    def get_minimized_molecules(self,
                                 ligand_filename: str,
                                 gen_mols_h: list[Mol], # with hydrogens
                                 complex_minimizer: ComplexMinimizer):
        assert len(gen_mols_h) > 0, 'You must give a non-empty list of molecules with hydrogens'
        target_dirname, real_ligand_filename = ligand_filename.split('/') 
        minimized_target_path = os.path.join(self.minimized_path, target_dirname)
        if not os.path.exists(minimized_target_path):
            os.mkdir(minimized_target_path)
        minimized_filename = 'generated_' + real_ligand_filename.replace('.sdf', 
                                                                        f'_{self.name}_minimized.sdf')
        minimized_filepath = os.path.join(minimized_target_path,
                                        minimized_filename)
        
        if not os.path.exists(minimized_filepath):
            logging.info(f'Minimizing molecules for {ligand_filename}')
            mini_gen_mols = []
            for mol_i, mol in enumerate(gen_mols_h):
                logging.info(f'Minimizing molecule {mol_i}')
                try:
                    mini_mol = complex_minimizer.minimize_ligand(mol)
                    if mini_mol is not None:
                        mini_gen_mols.append(mini_mol)
                except:
                    logging.info(f'Minimization failed for {mol_i}')
            logging.info(f'Saving minimized molecules in {minimized_filepath}')
            with Chem.SDWriter(minimized_filepath) as writer:
                for mol in mini_gen_mols:
                    writer.write(mol)
        else:
            logging.info(f'Loading minimized molecules from {minimized_filepath}')
            mini_gen_mols = [mol for mol in Chem.SDMolSupplier(minimized_filepath, 
                                                            removeHs=False)]
            
        return mini_gen_mols

class TargetDiff(SBModel):
    
    def __init__(self,
                 name: str = 'TargetDiff',
                 minimized_path: str = MINIMIZED_DIRPATH,
                 results_path: str = TARGETDIFF_RESULTS_FILEPATH,
                 ) -> None:
        super().__init__(name,
                         minimized_path)
        self.results_path = results_path
        with open(results_path, 'rb') as f:
            self.results = pickle.load(f)
    
    def get_ligand_filenames(self):
        return list(self.results.keys())
    
    def get_generated_molecules(self, 
                                ligand_filename: str):
        gen_mols = []
    
        filename_found = False
        for target_results in self.results:
            if ligand_filename == target_results[0]['ligand_filename']:
                filename_found = True
                break
            
        if filename_found:
            for mol_results in target_results:
                gen_mol = mol_results['mol']
                gen_mols.append(gen_mol)
        else:
            logging.warning(f'Ligand filename {ligand_filename} not found in TargetDiff results')
    
        return gen_mols
    

class ThreeDSBDD(SBModel):
    
    def __init__(self,
                 name: str = '3D_SBDD',
                 minimized_path: str = MINIMIZED_DIRPATH,
                 gen_path = THREEDSBDD_GEN_DIRPATH
                 ) -> None:
        super().__init__(name,
                         minimized_path)
        self.gen_path = gen_path
    
    def get_generated_molecules(self, 
                                ligand_filename: str):
        gen_mols_filename = ligand_filename.replace('/', '/generated_').replace('.sdf', '_pocket10.pdb.sdf')
        gen_mols_filepath = os.path.join(self.gen_path, gen_mols_filename)
        if not os.path.exists(gen_mols_filepath):
            logging.warning(f'Ligand filename {ligand_filename} not found in 3D-SBDD results')
            gen_mols = []
        else:
            gen_mols = [mol 
                        for mol in Chem.SDMolSupplier(gen_mols_filepath) 
                        # if mol is not None
                        ]
    
        return gen_mols
    
class Pocket2Mol(SBModel):
    
    def __init__(self,
                 name: str = 'Pocket2Mol',
                 minimized_path: str = MINIMIZED_DIRPATH,
                 gen_path = POCKET2MOL_GEN_DIRPATH
                 ) -> None:
        super().__init__(name,
                         minimized_path)
        self.gen_path = gen_path
    
    def get_generated_molecules(self, 
                                ligand_filename: str):
        
        gen_mols_filename = ligand_filename.replace('/', '/generated_').replace('.sdf', '_pocket10.pdb.sdf')
        gen_mols_filepath = os.path.join(self.gen_path, gen_mols_filename)
        if not os.path.exists(gen_mols_filepath):
            logging.warning(f'Ligand filename {ligand_filename} not found in Pocket2Mol results')
            gen_mols = []
        else:
            gen_mols = [mol 
                        for mol in Chem.SDMolSupplier(gen_mols_filepath) 
                        # if mol is not None
                        ]
    
        return gen_mols


class DiffSBDD(SBModel):
    
    def __init__(self,
                 name: str = 'DiffSBDD',
                 minimized_path: str = MINIMIZED_DIRPATH,
                 gen_path = DIFFSBDD_GEN_DIRPATH
                 ) -> None:
        super().__init__(name,
                         minimized_path)
        self.gen_path = gen_path
    
    def get_generated_molecules(self, 
                                ligand_filename: str):
        
        suffix = ligand_filename.split('/')[-1]
        pocket_part = suffix.replace('.sdf', '_pocket10').replace('_', '-')
        gen_part = suffix[:-4].replace('_', '-') + '_gen.sdf'
        gen_mols_filename = '_'.join([pocket_part, gen_part])
        gen_mols_filepath = os.path.join(self.gen_path, gen_mols_filename)
        if not os.path.exists(gen_mols_filepath):
            logging.warning(f'Ligand filename {ligand_filename} not found in DiffSBDD results')
            gen_mols = []
        else:
            gen_mols = [mol 
                        for mol in Chem.SDMolSupplier(gen_mols_filepath)
                        # if mol is not None
                        ]
    
        return gen_mols
    
    
class LiGAN(SBModel):
    
    def __init__(self,
                 name: str = 'LiGAN',
                 minimized_path: str = MINIMIZED_DIRPATH,
                 gen_path = LIGAN_GEN_DIRPATH
                 ) -> None:
        super().__init__(name,
                         minimized_path)
        self.gen_path = gen_path
    
    def get_generated_molecules(self, 
                                ligand_filename: str):
        i = self.crossdocked.get_filename_i(ligand_filename)
        gen_mols_filepath = os.path.join(self.gen_path, f'Generated_{i}_lig_gen_fit_add.sdf.gz')
        if not os.path.exists(gen_mols_filepath):
            logging.warning(f'Ligand filename {ligand_filename} not found in LiGAN results')
            gen_mols = []
        else:
            gz_stream = gzip.open(gen_mols_filepath)
            with Chem.ForwardSDMolSupplier(gz_stream) as gzsuppl:
                # mols = []
                # for mol in gzsuppl:
                #     try:
                #         if (mol is not None) and (mol.GetNumAtoms() > 0) and (not '.' in Chem.MolToSmiles(mol)) :
                #             mols.append(mol)
                #     except Exception as e:
                #         print(f'Mol not read exception: {e}')
                mols = [mol 
                        for mol in gzsuppl 
                        # if mol is not None
                        ]
                gen_mols = mols
    
        return gen_mols
    

class ResGen(SBModel):
    
    def __init__(self,
                 name: str = 'ResGen',
                 minimized_path: str = MINIMIZED_DIRPATH,
                 gen_path = RESGEN_GEN_DIRPATH
                 ) -> None:
        super().__init__(name,
                         minimized_path)
        self.gen_path = gen_path
    
    def get_generated_molecules(self, 
                                ligand_filename: str):
        
        gen_mols_filename = ligand_filename.replace('/', '/generated_').replace('.sdf', '_pocket10.pdb.sdf')
        gen_mols_filepath = os.path.join(self.gen_path, gen_mols_filename)
        if not os.path.exists(gen_mols_filepath):
            logging.warning(f'Ligand filename {ligand_filename} not found in ResGen results')
            gen_mols = []
        else:
            gen_mols = [mol 
                        for mol in Chem.SDMolSupplier(gen_mols_filepath) 
                        # if mol is not None
                        ]
    
        return gen_mols
    
    
class Ymir(SBModel):
    
    def __init__(self,
                 name: str = 'Ymir',
                 minimized_path: str = MINIMIZED_DIRPATH,
                 gen_path = '/home/bb596/hdd/ymir/generated_cross_docked_late/'
                 ) -> None:
        super().__init__(name,
                         minimized_path)
        self.gen_path = gen_path
    
    def get_generated_molecules(self, 
                                ligand_filename: str):
        
        _, real_ligand_filename = ligand_filename.split('/')
        gen_mols_filename = real_ligand_filename.replace('.sdf', '.sdf_generated.sdf')
        gen_mols_filepath = os.path.join(self.gen_path, gen_mols_filename)
        if not os.path.exists(gen_mols_filepath):
            logging.warning(f'Ligand filename {ligand_filename} not found in Ymir results')
            gen_mols = []
        else:
            gen_mols = [mol 
                        for mol in Chem.SDMolSupplier(gen_mols_filepath) 
                        # if mol is not None
                        ]
            
        has_attach_l = []
        for gen_mol in gen_mols:
            has_attach = False
            for atom in gen_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetAtomicNum(1)
                    has_attach = True
            has_attach_l.append(has_attach)
            Chem.SanitizeMol(gen_mol)
            Chem.AssignStereochemistry(gen_mol)
                    
        logging.info(f'{ligand_filename}: has attach: {sum(has_attach_l)}')
    
        return gen_mols
    
    
class YmirRandom(SBModel):
    
    def __init__(self,
                 name: str = 'YmirRandom',
                 minimized_path: str = MINIMIZED_DIRPATH,
                 gen_path = '/home/bb596/hdd/ymir/generated_cross_docked_random/'
                 ) -> None:
        super().__init__(name,
                         minimized_path)
        self.gen_path = gen_path
    
    def get_generated_molecules(self, 
                                ligand_filename: str):
        
        _, real_ligand_filename = ligand_filename.split('/')
        gen_mols_filename = real_ligand_filename.replace('.sdf', '.sdf_generated.sdf')
        gen_mols_filepath = os.path.join(self.gen_path, gen_mols_filename)
        if not os.path.exists(gen_mols_filepath):
            logging.warning(f'Ligand filename {ligand_filename} not found in Ymir results')
            gen_mols = []
        else:
            gen_mols = [mol 
                        for mol in Chem.SDMolSupplier(gen_mols_filepath) 
                        # if mol is not None
                        ]
            
        has_attach_l = []
        for gen_mol in gen_mols:
            has_attach = False
            for atom in gen_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetAtomicNum(1)
                    has_attach = True
            has_attach_l.append(has_attach)
            Chem.SanitizeMol(gen_mol)
            Chem.AssignStereochemistry(gen_mol)
                    
        logging.info(f'{ligand_filename}: has attach: {sum(has_attach_l)}')
    
        return gen_mols
    
    
class YmirEarly(SBModel):
    
    def __init__(self,
                 name: str = 'YmirEarly',
                 minimized_path: str = MINIMIZED_DIRPATH,
                 gen_path = '/home/bb596/hdd/ymir/generated_cross_docked_early/'
                 ) -> None:
        super().__init__(name,
                         minimized_path)
        self.gen_path = gen_path
    
    def get_generated_molecules(self, 
                                ligand_filename: str):
        
        _, real_ligand_filename = ligand_filename.split('/')
        gen_mols_filename = real_ligand_filename.replace('.sdf', '.sdf_generated.sdf')
        gen_mols_filepath = os.path.join(self.gen_path, gen_mols_filename)
        if not os.path.exists(gen_mols_filepath):
            logging.warning(f'Ligand filename {ligand_filename} not found in Ymir results')
            gen_mols = []
        else:
            gen_mols = [mol 
                        for mol in Chem.SDMolSupplier(gen_mols_filepath) 
                        # if mol is not None
                        ]
            
        has_attach_l = []
        for gen_mol in gen_mols:
            has_attach = False
            for atom in gen_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetAtomicNum(1)
                    has_attach = True
            has_attach_l.append(has_attach)
            Chem.SanitizeMol(gen_mol)
            Chem.AssignStereochemistry(gen_mol)
                    
        logging.info(f'{ligand_filename}: has attach: {sum(has_attach_l)}')
    
        return gen_mols