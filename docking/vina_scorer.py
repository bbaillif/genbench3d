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
from data.preprocessing.protein_processor import ProteinProcessor


class VinaScorer():
    
    def __init__(self,
                 receptor_pdbqt_filepath: str,
                ligand_name: str = None,
                chain: str = None, 
                sf_name: str = 'vina',
                preparation_method: str = 'adfr',
                n_cpus: int = 4,
                seed: int = 2023,
                size_border: float = 20) -> None:
        
        assert receptor_pdbqt_filepath.endswith('.pdbqt'), \
            'Input file must be in pdbqt format'
        
        self.receptor_pdbqt_filepath = receptor_pdbqt_filepath
        self.ligand_name = ligand_name
        self.chain = chain
        self.sf_name = sf_name
        self.preparation_method = preparation_method
        self.n_cpus = n_cpus
        self.seed = seed
        self.size_border = size_border
        
        self.pdb_filepath = receptor_pdbqt_filepath.replace('.pdbqt',
                                                            '.pdb')
        self.box_filepath = receptor_pdbqt_filepath.replace('.pdbqt', 
                                                 '_vina_box.txt')
        
        self.protein_processor = ProteinProcessor(self.pdb_filepath)
        
        if not os.path.exists(receptor_pdbqt_filepath):
            self.protein_processor.vina_prepare_receptor(ligand_name,
                                                         chain,
                                                         size_border,
                                                         preparation_method)
        
        self._vina = Vina(sf_name=sf_name,
                          cpu=n_cpus,
                          seed=seed)
        self._vina.set_receptor(receptor_pdbqt_filepath)
        
        # Pocket = box definition
        if ligand_name is not None and chain is not None:
            self.set_box(receptor_pdbqt_filepath)
        
        # if not os.path.exists(VINA_BIN_FILEPATH):
        #     self.download_vina()
       
    # @staticmethod
    # def download_vina() -> None:
    #     if not os.path.exists(VINA_DIRPATH):
    #         os.mkdir(VINA_DIRPATH)
    #     os.system(f'wget {VINA_URL} -O {VINA_BIN_FILEPATH}')
        
        
    def setup_box(self,
                  ligand: Union[Mol, AtomGroup]):
        if not os.path.exists(self.box_filepath):
            self.write_box_from_ligand(ligand)
        self.set_box()
        
    def write_box_from_ligand(self,
                                 ligand: Union[Mol, AtomGroup],
                                 size_border: float = 25, # Angstrom
                                 ) -> None:
        if isinstance(ligand, Mol):
            conf = ligand.GetConformer()
            ligand_positions = conf.GetPositions()
        elif isinstance(ligand, AtomGroup):
            ligand_positions = ligand.positions
        else:
            import pdb;pdb.set_trace()
            raise Exception('Input ligand should be RDKit Mol or MD Universe')
        box_center = (ligand_positions.max(axis=0) + ligand_positions.min(axis=0)) / 2
        # box_size = ligand_positions.max(axis=0) - ligand_positions.min(axis=0) + size_border
        box_size = [size_border] * 3
        
        with open(self.box_filepath, 'w') as f:
            for center_value, axis in zip(box_center, 'xyz'):
                f.write(f'center_{axis} = {center_value}')
                f.write('\n')
            for size_value, axis in zip(box_size, 'xyz'):
                f.write(f'size_{axis} = {size_value}')
                f.write('\n')
        
        
    def score_mol(self,
                  ligand: Mol,
                  minimized: bool = False,
                  output_filepath: str = None,
                  add_hydrogens: bool = True,
                  ) -> List[float]:
            
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
            
        assert pdb_filepath.endswith('pdb')
        
        ligand = Chem.MolFromPDBFile(pdb_filepath)
        energies = self.score_mol(ligand,
                                  minimized)
        
        return energies
        
        
    def set_box(self) -> None:
        with open(self.box_filepath, 'r') as f:
            lines: List[str] = [line.strip() for line in f.readlines()]
        d = {}
        for line in lines:
            l = line.split(' ')
            key = l[0]
            value = l[2]
            d[key] = float(value)
        center = [d[f'center_{axis}'] for axis in 'xyz']
        box_size = [d[f'size_{axis}'] for axis in 'xyz']
        self._vina.compute_vina_maps(center=center, 
                                     box_size=box_size)
        