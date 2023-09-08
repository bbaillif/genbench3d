import os
import subprocess
import gemmi

from params import CCP4_DIRPATH
from pdbredo import PDBREDO

class WGETException(Exception):
    pass

class PDBe():
    
    def __init__(self,
                 root: str = CCP4_DIRPATH):
        self.root = root
        self.pdbredo = PDBREDO()
        
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        
    def get_ccp4_filepath(self,
                          pdb_id: str,
                          centre: bool = True) -> str:
        ccp4_filename = f'{pdb_id}.ccp4'
        ccp4_filepath = os.path.join(self.root, ccp4_filename)
        if not os.path.exists(ccp4_filepath):
            self.download_ccp4_file(pdb_id, ccp4_filepath)
            
        if centre:
            ccp4_centred_filename = f'{pdb_id}_centred.ccp4'
            ccp4_centred_filepath = os.path.join(self.root, ccp4_centred_filename)
            pdb_filepath = self.pdbredo.get_pdb_filepath(pdb_id)
            if not os.path.exists(ccp4_centred_filepath):
                self.generate_ccp4_centred(ccp4_filepath,
                                        ccp4_centred_filepath,
                                        pdb_filepath)
            return ccp4_centred_filepath
        else:
            return ccp4_filepath
    
    def download_ccp4_file(self,
                           pdb_id: str,
                           ccp4_filepath: str,) -> None:
        cmd = f'wget -O {ccp4_filepath} https://www.ebi.ac.uk/pdbe/entry-files/{pdb_id}.ccp4'
        process = subprocess.run(cmd.split())
        if process.returncode == 8:
            if os.path.exists(ccp4_filepath):
                os.remove(ccp4_filepath)
            raise WGETException('File could not be downloaded (probably 404 not found)')
            
        
    def generate_ccp4_centred(self,
                              ccp4_filepath: str,
                              ccp4_centred_filepath: str,
                              pdb_filepath: str) -> None:
        ccp4 = gemmi.read_ccp4_map(ccp4_filepath)
        ccp4.setup(0)
        
        structure = gemmi.read_structure(pdb_filepath)
        ccp4.set_extent(structure.calculate_fractional_box(margin=5))
        ccp4.write_ccp4_map(ccp4_centred_filepath)