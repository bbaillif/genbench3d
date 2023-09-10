import os
import subprocess
from genbench3d.params import PDB_REDO_DIRPATH

class PDBREDO():
    
    def __init__(self,
                 root: str = PDB_REDO_DIRPATH):
        self.root = root
        
    def get_pdb_filepath(self,
                         pdb_id: str) -> str:
        pdb_redo_filename = f'{pdb_id}_final.pdb'
        pdb_redo_filepath = os.path.join(self.root, pdb_redo_filename)
        if not os.path.exists(pdb_redo_filepath):
            self.download_pdb_file(pdb_id, pdb_redo_filepath)
        return pdb_redo_filepath
        
    def download_pdb_file(self,
                     pdb_id: str,
                     pdb_redo_filepath: str) -> None: # TODO: handle errors
        cmd = f'wget -O {pdb_redo_filepath} https://pdb-redo.eu/db/{pdb_id}/{pdb_id}_final.pdb'
        subprocess.run(cmd.split())
        
        