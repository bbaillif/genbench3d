import os

from MDAnalysis import Universe

class Protein():

    def __init__(self,
                 pdb_filepath: str) -> None:
        assert os.path.exists(pdb_filepath)
        assert pdb_filepath.endswith('.pdb')
        self.pdb_filepath = pdb_filepath
        
        self._universe = None
        
    
    @property
    def universe(self):
        if self._universe is None:
            self._universe = Universe(self.pdb_filepath)
        return self._universe
        
