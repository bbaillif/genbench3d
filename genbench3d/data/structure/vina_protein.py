import logging
import os

from MDAnalysis import (Universe,
                        AtomGroup)
from genbench3d.params import PREPARE_RECEPTOR_BIN_PATH
from openbabel import pybel
from openbabel.pybel import Molecule
from .protein import Protein
from pdbfixer import PDBFixer
from openmm.app import PDBFile

class VinaProtein(Protein):
    
    def __init__(self, 
                 pdb_filepath: str) -> None:
        super().__init__(pdb_filepath)
        
        self._protein_filepath = pdb_filepath.replace('.pdb', 
                                                     '_protein_only.pdb')
        self._protein_clean_filepath = pdb_filepath.replace('.pdb', 
                                                           '_protein_only_clean.pdb')
        self._pdbqt_filepath = pdb_filepath.replace('.pdb', 
                                                   '.pdbqt')
        
        
    @property
    def protein_filepath(self):
        if not os.path.exists(self._protein_filepath):
            self.extract_protein(universe=self.universe,
                                 output_pdb_filepath=self._protein_filepath)
        assert os.path.exists(self._protein_filepath), \
            'Something went wrong during protein extraction'
        return self._protein_filepath
    
    
    @property
    def protein_clean_filepath(self):
        if not os.path.exists(self._protein_clean_filepath):
            self.clean_protein(input_pdb_filepath=self.protein_filepath,
                                 output_pdb_filepath=self._protein_clean_filepath)
        assert os.path.exists(self._protein_clean_filepath), \
            'Something went wrong during protein cleaning'
        return self._protein_clean_filepath
    
    
    @property
    def pdbqt_filepath(self):
        if not os.path.exists(self._pdbqt_filepath):
            self.vina_prepare_receptor(universe=self.universe,
                                       output_pdbqt_filepath=self._pdbqt_filepath) # Using default configuration
        assert os.path.exists(self._pdbqt_filepath), \
            'Something went wrong during Vina receptor preparation'
        return self._pdbqt_filepath
        
        
    def vina_prepare_receptor(self,
                              universe: Universe,
                              output_pdbqt_filepath: str,
                              ligand_name: str = None,
                                chain: str = None,
                                preparation_method: str = 'adfr',
                                pH: float = 7.4
                                ) -> None:
        """
        inspired from teachopencadd talktorial 15 on protein_ligand_docking
        """
        
        self.extract_protein(universe=universe,
                             output_pdb_filepath=self.protein_filepath)
        
        self.clean_protein(input_pdb_filepath=self.protein_filepath,
                           output_pdb_filepath=self.protein_clean_filepath,
                           pH=pH)
        
        # self.clean_protein(input_pdb_filepath=self.pdb_filepath,
        #                    output_pdb_filepath=self._protein_clean_filepath,
        #                    pH=pH)
        
        if preparation_method == 'adfr':
            self.adfr_receptor_preparation(input_pdb_filepath=self.protein_clean_filepath,
                                           output_pdbqt_filepath=output_pdbqt_filepath)
        else:
            self.pdb_to_pdbqt()
            
        if ligand_name is not None and chain is not None:
            logging.info(f'Extracting ligand data on {ligand_name} and chain {chain}')
            self.current_ligand_filepath = self.extract_ligand(universe=universe,
                                                                ligand_name=ligand_name,
                                                                chain=chain)
        else:
            logging.info('No ligand name and/or chain is given, only computing protein files')
            
        
    def extract_protein(self,
                        universe: Universe,
                        output_pdb_filepath: str) -> None:
        """
        Extracts the protein only from an input pdb file that may also contains
        ligands, cofactors, metals...
        """
        logging.info(f'Extracting protein from universe to {output_pdb_filepath}')
        protein: AtomGroup = universe.select_atoms('protein')
        protein.write(output_pdb_filepath)
        
        
    def extract_ligand(self,
                       universe: str,
                       ligand_name: str,
                       chain: str) -> str:
        """
        Extracts the given ligand from an input pdb file that may also contains
        ligands, cofactors, metals...
        """
        ligand_filepath = self.pdb_filepath.replace('.pdb', 
                                                    f'{ligand_name}_{chain}_ligand_only.pdb')
        logging.info(f"""Extracting ligand {ligand_name} of chain {chain} 
                     from universe to {ligand_filepath}""")
        ligand_selection = f'resname {ligand_name} and segid {chain}'
        ligand: AtomGroup = universe.select_atoms(ligand_selection)
        ligand.write(ligand_filepath)
        return ligand_filepath
        
            
    def clean_protein(self,
                      input_pdb_filepath: str,
                      output_pdb_filepath: str,
                      pH: float = 7.4,
                      ) -> None:
        """
        Adds hydrogens to given pH for a pdb protein (input_filepath)
        """
        logging.info(f'Cleaning protein from {input_pdb_filepath} to {output_pdb_filepath}')
        
        fixer = PDBFixer(filename=input_pdb_filepath)
        # We only use PDBFixer to reinitialize the chains and segment id combinations
        # (e.g. chain A and B, having both segment A and B, with be renamed chain A to D)
        # fixer.findMissingResidues() # it cannot find them with CrossDocked because it does not contain sequence info
        # fixer.findNonstandardResidues()
        # fixer.replaceNonstandardResidues()
        # fixer.findMissingAtoms()
        # fixer.addMissingAtoms()
        # fixer.addMissingHydrogens(pH=pH)
        # fixer.removeHeterogens(keepWater=False)
        intermediate_filepath = output_pdb_filepath.replace('.pdb', '_no_h.pdb')
        PDBFile.writeFile(fixer.topology, fixer.positions, open(intermediate_filepath, 'w'))
            
        # pdbfixer might miss some hydrogens because of missing atoms
        molecule = list(pybel.readfile("pdb", str(intermediate_filepath)))[0]
        # # import pdb;pdb.set_trace()
        # molecule.OBMol.CorrectForPH(pH) # the correct functions fails for some groups: -NH3+ becomes -NH4+
        # molecule.removeh() # some pdb files have hydrogens, these might mess up the next step
        molecule.addh() 
        molecule.write("pdb", str(output_pdb_filepath), overwrite=True) 
            
            
    def adfr_receptor_preparation(self,
                                  input_pdb_filepath: str,
                                  output_pdbqt_filepath: str,
                                  ) -> None:
        """
        input_pdb_filepath must be a pbd file that only contains the protein with
        hydrogens
        """
        logging.info(f'Preparing protein from {input_pdb_filepath} to {output_pdbqt_filepath}')
        arg_list = [PREPARE_RECEPTOR_BIN_PATH,
                    f'-r {input_pdb_filepath}',
                    f'-o {output_pdbqt_filepath}']
        cmd = ' '.join(arg_list)
        os.system(cmd)
        
        
    def pdb_to_pdbqt(self,
                     ph: float = 7.4,
                     ) -> None:
        molecule = list(pybel.readfile("pdb", str(self._protein_filepath)))[0]
        self.ob_mol_to_pdbqt(molecule, ph)
        
        
    def ob_mol_to_pdbqt(self,
                        molecule: Molecule,
                        ph: float = 7.4,
                        ) -> None:
        
        # add hydrogens at given pH
        molecule.OBMol.CorrectForPH(ph)
        molecule.addh()
        # add partial charges to each atom
        for atom in molecule.atoms:
            atom.OBAtom.GetPartialCharge()
        molecule.write("pdb", str(self._protein_clean_filepath), overwrite=True) 
        molecule.write("pdbqt", str(self._pdbqt_filepath), overwrite=True)
        
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