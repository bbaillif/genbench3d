import os

from typing import List, Union
from rdkit import Chem
from rdkit.Chem import Mol
from meeko import MoleculePreparation
# from params import (VINA_BIN_FILEPATH, 
#                     VINA_URL, 
#                     VINA_DIRPATH)
from vina import Vina
from MDAnalysis import AtomGroup
from genbench3d.data.structure.protein import VinaProtein

DEFAULT_SF_NAME = 'vina'
DEFAULT_PREPARATION_METHOD = 'adfr'
DEFAULT_SIZE_BORDER = 35 # Angstrom
DEFAULT_N_CPUS = 4
SEED = 2023

class VinaScorer():
    
    def __init__(self,
                 vina_protein: VinaProtein,
                 sf_name: str = DEFAULT_SF_NAME,
                 preparation_method: str = DEFAULT_PREPARATION_METHOD,
                 n_cpus: int = DEFAULT_N_CPUS,
                 seed: int = SEED,
                 size_border: float = DEFAULT_SIZE_BORDER) -> None:
        
        self.vina_protein = vina_protein
        self.sf_name = sf_name
        self.preparation_method = preparation_method
        self.n_cpus = n_cpus
        self.seed = seed
        self.size_border = size_border
        
        self._vina = Vina(sf_name=sf_name,
                          cpu=n_cpus,
                          seed=seed)
        self._vina.set_receptor(self.vina_protein.pdbqt_filepath) 
        # will automatically perform protein preparation if 
        # the pdbqt file does not exist
        
        # at this stage, the Vina box is not defined
        # it is recommended to use the class methods to instantiate this class
        

        # if not os.path.exists(VINA_BIN_FILEPATH):
        #     self.download_vina()
       
    # @staticmethod
    # def download_vina() -> None:
    #     if not os.path.exists(VINA_DIRPATH):
    #         os.mkdir(VINA_DIRPATH)
    #     os.system(f'wget {VINA_URL} -O {VINA_BIN_FILEPATH}')
        
        
    @classmethod
    def from_ligand(cls,
                    ligand: Union[Mol, AtomGroup],
                    vina_protein: VinaProtein,
                    sf_name: str = DEFAULT_SF_NAME,
                    preparation_method: str = DEFAULT_PREPARATION_METHOD,
                    n_cpus: int = DEFAULT_N_CPUS,
                    seed: int = SEED,
                    size_border: float = DEFAULT_SIZE_BORDER) -> 'VinaScorer':
        vina_scorer = cls(vina_protein,
                         sf_name,
                         preparation_method,
                         n_cpus,
                         seed,
                         size_border)
        vina_scorer.set_box_from_ligand(ligand)
        return vina_scorer
        
    
    @classmethod
    def from_ligand_name_chain(cls,
                               ligand_name: str,
                                chain: str, 
                                vina_protein: VinaProtein,
                                sf_name: str = DEFAULT_SF_NAME,
                                preparation_method: str = DEFAULT_PREPARATION_METHOD,
                                n_cpus: int = DEFAULT_N_CPUS,
                                seed: int = SEED,
                                size_border: float = DEFAULT_SIZE_BORDER) -> 'VinaScorer':
        vina_scorer = cls(vina_protein,
                         sf_name,
                         preparation_method,
                         n_cpus,
                         seed,
                         size_border)
        ligand_filepath = vina_scorer.vina_protein.extract_ligand(universe=vina_scorer.protein.universe,
                                                            ligand_name=ligand_name,
                                                            chain=chain)
        ligand = Chem.MolFromPDBFile(ligand_filepath)
        vina_scorer.set_box_from_ligand(ligand, 
                                        size_border=size_border)
        return vina_scorer
    
        
    def set_box_from_ligand(self,
                            ligand: Union[Mol, AtomGroup],
                            size_border: float = DEFAULT_SIZE_BORDER):
        
        # with open(self.box_filepath, 'r') as f:
        #     lines: List[str] = [line.strip() for line in f.readlines()]
        # d = {}
        # for line in lines:
        #     l = line.split(' ')
        #     key = l[0]
        #     value = l[2]
        #     d[key] = float(value)
        # center = [d[f'center_{axis}'] for axis in 'xyz']
        # box_size = [d[f'size_{axis}'] for axis in 'xyz']
        
        if isinstance(ligand, Mol):
            conf = ligand.GetConformer()
            ligand_positions = conf.GetPositions()
        elif isinstance(ligand, AtomGroup):
            ligand_positions = ligand.positions
        else:
            import pdb;pdb.set_trace()
            raise Exception('Input ligand should be RDKit Mol or MD Universe')
        center = (ligand_positions.max(axis=0) + ligand_positions.min(axis=0)) / 2
        # box_size = ligand_positions.max(axis=0) - ligand_positions.min(axis=0) + size_border
        box_size = [size_border] * 3
        self._vina.compute_vina_maps(center=center, 
                                     box_size=box_size)
        
    # def write_box_from_ligand(self,
    #                              ligand: Union[Mol, AtomGroup],
    #                              size_border: float = DEFAULT_SIZE_BORDER, # Angstrom
    #                              ) -> None:
    #     with open(self.box_filepath, 'w') as f:
    #         for center_value, axis in zip(box_center, 'xyz'):
    #             f.write(f'center_{axis} = {center_value}')
    #             f.write('\n')
    #         for size_value, axis in zip(box_size, 'xyz'):
    #             f.write(f'size_{axis} = {size_value}')
    #             f.write('\n')
        
        
    def score_mol(self,
                  ligand: Mol,
                  minimized: bool = False,
                  output_filepath: str = None,
                  add_hydrogens: bool = True,
                  ) -> List[float]:
            
        assert self._vina._center is not None, \
            'You need to setup the pocket box using the set_box_from_ligand function'
            
        # Ligand preparation
        if add_hydrogens:
            ligand = Chem.AddHs(ligand, addCoords=True)
        
        preparator = MoleculePreparation()
        preparator.prepare(ligand)
        pdbqt_string = preparator.write_pdbqt_string()
        self._vina.set_ligand_from_string(pdbqt_string)
        
        if minimized:
            energies = self._vina.optimize()
        else:
            energies = self._vina.score()
        if output_filepath is not None:
            self._vina.write_pose(output_filepath, overwrite=True)
        return energies
        
        
    def score_pdbqt(self,
                    pdbqt_filepath: str,
                    minimized: bool = False,
                    ) -> List[float]:
            
        assert self._vina._center is not None, \
            'You need to setup the pocket box using the set_box_from_ligand function'
        assert pdbqt_filepath.endswith('pdbqt')
            
        self._vina.set_ligand_from_file(pdbqt_filepath)
        if minimized:
            energies = self._vina.optimize()
        else:
            energies = self._vina.score()
        return energies
    
    
    def score_sdf(self,
                  sdf_filepath: str,
                  ) -> List[List[float]]:
            
        assert self._vina._center is not None, \
            'You need to setup the pocket box using the set_box_from_ligand function'
        assert sdf_filepath.endswith('sdf')
        
        with Chem.SDMolSupplier(sdf_filepath) as suppl:
            mols = [mol for mol in suppl]
            
        all_energies = []
        for ligand in mols:
            energies = self.score_mol(ligand)
            all_energies.append(energies)
        
        return all_energies
    
    
    def score_pdb(self,
                  pdb_filepath: str,
                  minimized: bool = False,
                  ) -> List[float]:
            
        assert self._vina._center is not None, \
            'You need to setup the pocket box using the set_box_from_ligand function'
        assert pdb_filepath.endswith('pdb')
        
        ligand = Chem.MolFromPDBFile(pdb_filepath)
        energies = self.score_mol(ligand,
                                  minimized)
        
        return energies