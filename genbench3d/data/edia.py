import os
import subprocess
import pandas as pd

from typing import Dict
from .pdbredo import PDBREDO
from .pdbe import PDBe
from .pdbbind import PDBbind
from genbench3d.params import (EDIA_BIN_FILEPATH, 
                    EDIA_DATA_DIRPATH,
                    EDIA_LICENSE_FILEPATH)

class EDIAFailedException(Exception):
    
    pass


class EDIA():
    
    def __init__(self,
                 edia_bin: str = EDIA_BIN_FILEPATH,
                 root: str = EDIA_DATA_DIRPATH,
                 license_filepath: str = EDIA_LICENSE_FILEPATH,
                 use_pdb_redo: bool = False):
        self.edia_bin = edia_bin
        self.root = root
        self.license_filepath = license_filepath
        self.use_pdb_redo = use_pdb_redo
        
        if not os.path.exists(root):
            os.mkdir(root)
            
        self.license = self.get_license_from_file(license_filepath)
        
        if use_pdb_redo:
            self.pdb_source = PDBREDO()
        else:
            self.pdb_source = PDBbind()
            
        self.pdbe = PDBe()
        
        
    def get_edias(self,
                 pdb_id: str) -> Dict[str, pd.DataFrame]:
        output_dirpath = self.get_output_dirpath(pdb_id,
                                                 run_edia=True) # Runs EDIA
        filenames = os.listdir(output_dirpath)
        if len(filenames) != 3:
            raise EDIAFailedException(f'EDIA failed for {pdb_id}')
        for filename in filenames:
            if filename.endswith('.pdb'):
                edia_pdb_filename = filename
                edia_pdb_filepath = os.path.join(output_dirpath,
                                                 edia_pdb_filename)
            elif filename.endswith('atomscores.csv'):
                atom_scores_filename = filename
                atom_scores_filepath = os.path.join(output_dirpath, 
                                                    atom_scores_filename)
            elif filename.endswith('structurescores.csv'):
                structure_scores_filename = filename
                structure_scores_filepath = os.path.join(output_dirpath,
                                                         structure_scores_filename)
            else:
                raise EDIAFailedException(f'We have an unknown file in the {pdb_id} directory')
        
        # atom_scores_df = pd.read_csv(atom_scores_filepath)
        structure_scores_df = pd.read_csv(structure_scores_filepath)
        # return structure_scores_df[structure_scores_df['Structure specifier'] == 'm']
        return structure_scores_df
        
        
    def get_output_dirpath(self,
                           pdb_id: str,
                           run_edia: bool = True) -> str:
        if self.use_pdb_redo:
            output_dirpath = os.path.join(self.root, f'{pdb_id}_redo')
        else:
            output_dirpath = os.path.join(self.root, pdb_id)
        if run_edia: # default True, we want the results stored in the directory
            run = True
            if os.path.exists(output_dirpath):
                filenames = os.listdir(output_dirpath)
                if len(filenames) == 3: # if not, it means EDIA has not been run, or failed before
                    run = False
                
            if run: 
                pdb_filepath = self.pdb_source.get_pdb_filepath(pdb_id)
                ccp4_filepath = self.pdbe.get_ccp4_filepath(pdb_id)
                if not os.path.exists(output_dirpath):
                    os.mkdir(output_dirpath)
                self.run_edia(pdb_filepath=pdb_filepath,
                              ccp4_filepath=ccp4_filepath,
                              output_dirpath=output_dirpath)
                
                
        return output_dirpath
        
    def run_edia(self,
                 pdb_filepath: str,
                 ccp4_filepath: str,
                 output_dirpath: str,
                 ) -> str:
        cmd = f'{self.edia_bin} -t {pdb_filepath} -d {ccp4_filepath} -o {output_dirpath} -l {self.license}'
        subprocess.run(cmd.split())
        
    def get_license_from_file(self,
                              license_filepath):
        with open(license_filepath, 'r') as f:
            lines = f.readlines()
        return lines[-1].strip()