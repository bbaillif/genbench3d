from ..metric import Metric
from genbench3d.conf_ensemble import GeneratedCEL
from genbench3d.data.structure import Pocket
from rdkit import Chem
from genbench3d.geometry import GeometryExtractor
from rdkit.Chem import Mol
from collections import defaultdict

class StericClash(Metric):
    
    def __init__(self, 
                 pocket: Pocket,
                 name: str = 'Steric clash',
                 ) -> None:
        super().__init__(name)
        self.pocket = pocket
        self.geometry_extractor = GeometryExtractor()
        self.clashes = None
        self.valid_pldist_conf_ids = None
        
    
    def get(self,
            cel: GeneratedCEL) -> float:
        
        self.clashes = {}
        self.valid_pldist_conf_ids = defaultdict(list)
        self.n_valid = 0
        
        for name, ce in cel.items():
            ce_clashes = []
            # mols = ce.to_mol_list()
            
            conf_ids = [conf.GetId() for conf in ce.mol.GetConformers()]
            for conf_id in conf_ids:
                
                mol = Mol(ce.mol, confId=conf_id)
                ligand = Chem.AddHs(mol, addCoords=True)
                complx = Chem.CombineMols(self.pocket.mol, ligand)
                atoms = [atom for atom in complx.GetAtoms()]
                pocket_atoms = atoms[:self.pocket.mol.GetNumAtoms()]
                ligand_atoms = atoms[self.pocket.mol.GetNumAtoms():]
                distance_matrix = Chem.Get3DDistanceMatrix(mol=complx)
                
                is_clashing = False
            
                for atom1 in pocket_atoms:
                    idx1 = atom1.GetIdx()
                    symbol1 = atom1.GetSymbol()
                    for atom2 in ligand_atoms:
                        idx2 = atom2.GetIdx()
                        symbol2 = atom2.GetSymbol()
                        
                        vdw1 = self.geometry_extractor.get_vdw_radius(symbol1)
                        vdw2 = self.geometry_extractor.get_vdw_radius(symbol2)
                        
                        if symbol1 == 'H':
                            min_distance = vdw2
                        elif symbol2 == 'H':
                            min_distance = vdw1
                        else:
                            # min_distance = vdw1 + vdw2 - self.clash_tolerance
                            min_distance = vdw1 + vdw2
                            
                        min_distance = min_distance * 0.75
                            
                        distance = distance_matrix[idx1, idx2]
                        if distance < min_distance:
                            is_clashing = True
                            invalid_d = {
                            'conf_id': conf_id,
                            'atom_idx1': idx1,
                            'atom_idx2': idx2,
                            'atom_symbol1': symbol1,
                            'atom_symbol2': symbol2,
                            'distance': distance,
                            }
                            ce_clashes.append(invalid_d)
                                    
                if not is_clashing:
                    self.valid_pldist_conf_ids[name].append(conf_id)
                    self.n_valid += 1
                    
            if len(ce_clashes) > 0:
                self.clashes[name] = ce_clashes
                    
        self.value = self.n_valid / cel.n_total_confs
                    
        return self.value