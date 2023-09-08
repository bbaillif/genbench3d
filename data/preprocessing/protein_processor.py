import os
import logging

from MDAnalysis import Universe
from params import PREPARE_RECEPTOR_BIN_PATH
from openbabel import pybel
from openbabel.pybel import Molecule

class ProteinProcessor():
    
    def __init__(self,
                 pdb_filepath: str) -> None:
        assert os.path.exists(pdb_filepath)
        assert pdb_filepath.endswith('.pdb')
        self.pdb_filepath = pdb_filepath
        
        self.pdbqt_filepath = pdb_filepath.replace('.pdb', 
                                                   '.pdbqt')
        self.protein_filepath = pdb_filepath.replace('.pdb', 
                                                     '_protein_only.pdb')
        self.protein_clean_filepath = pdb_filepath.replace('.pdb', 
                                                           '_protein_only_clean.pdb')
        
        
    def vina_prepare_receptor(self,
                              ligand_name: str = None,
                                chain: str = None,
                                preparation_method: str = 'adfr'):
        # inspired from teachopencadd talktorial 15 on protein_ligand_docking
        
        if ligand_name is None:
            logging.info('No ligand name is given, only computing protein files')
        else:
            logging.info(f'Extracting ligand data on {ligand_name}')
        
        complex = Universe(self.pdb_filepath)
        protein = complex.select_atoms('protein')
        protein.write(self.protein_filepath)
        if preparation_method == 'adfr':
            self.adfr_receptor_preparation()
        else:
            self.pdb_to_pdbqt()
            
        if ligand_name is not None and chain is not None:
            ligand = complex.select_atoms(f'resname {ligand_name} and segid {chain}')
            self.ligand_filepath = self.pdb_filepath.replace('.pdb', 
                                                             f'{ligand_name}_ligand_only.pdb')
            ligand.write(self.ligand_filepath)
            
            
    def adfr_receptor_preparation(self,
                                  pH: float = 7.4,
                                  ) -> None:
        
        molecule = list(pybel.readfile("pdb", str(self.pdb_filepath)))[0]
        molecule.OBMol.CorrectForPH(pH)
        molecule.addh()
        molecule.write("pdb", str(self.protein_clean_filepath), overwrite=True) 
        arg_list = [PREPARE_RECEPTOR_BIN_PATH,
                    f'-r {self.protein_clean_filepath}',
                    f'-o {self.pdbqt_filepath}']
        cmd = ' '.join(arg_list)
        os.system(cmd)
        
        
    def pdb_to_pdbqt(self,
                     pH: float = 7.4,
                     ) -> None:
        
        molecule = list(pybel.readfile("pdb", str(self.protein_filepath)))[0]
        self.ob_mol_to_pdbqt(molecule, pH)
        
        
    def ob_mol_to_pdbqt(self,
                        molecule: Molecule,
                        pH: float = 7.4,
                        ) -> None:
        
        # inspired from teachopencadd talktorial 15 on protein_ligand_docking
        
        # add hydrogens at given pH
        molecule.OBMol.CorrectForPH(pH)
        molecule.addh()
        # add partial charges to each atom
        for atom in molecule.atoms:
            atom.OBAtom.GetPartialCharge()
        molecule.write("pdb", str(self.protein_clean_filepath), overwrite=True) 
        molecule.write("pdbqt", str(self.pdbqt_filepath), overwrite=True)
        
        # Only keep ATOM and TER lines in pdbqt file
        with open(self.pdbqt_filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        new_lines = [line 
                     for line in lines 
                     if line.startswith('ATOM') or line.startswith('TER')]
        with open(self.pdbqt_filepath, 'w') as f:
            for line in new_lines:
                f.write(line)
                f.write('\n')
        
