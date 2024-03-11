from rdkit import Chem
from rdkit.Chem import Conformer, Mol
from genbench3d.geometry import GeometryExtractor
from typing import NamedTuple

class Clash(NamedTuple):
    i: int
    symbol_i: str
    j: int
    symbol_j: str
    distance: float
    min_distance: float
    

class ClashChecker():
    
    def __init__(self,
                 safety_ratio: float = 0.75,
                 consider_hs: bool = False) -> None:
        self.safety_ratio = safety_ratio
        self.consider_hs = consider_hs
        
        self.vdw_min_distances = {}
        self.geometry_extractor = GeometryExtractor()
        
        
    def get_vdw_min_distance(self,
                             symbol1, 
                             symbol2):
        vdw1 = self.geometry_extractor.get_vdw_radius(symbol1)
        vdw2 = self.geometry_extractor.get_vdw_radius(symbol2)
        
        if symbol1 == 'H':
            min_distance = vdw2
        elif symbol2 == 'H':
            min_distance = vdw1
        else:
            # min_distance = vdw1 + vdw2 - self.clash_tolerance
            min_distance = vdw1 + vdw2
            
        min_distance = min_distance * self.safety_ratio
        
        return min_distance
    
    
    def get_clashes(self,
                    mol: Mol,
                    conf_id: int,
                    excluded_pairs: list[tuple[int, int]] = []
                    ) -> list[Clash]:
        
        assert all(i <= j for i, j in excluded_pairs)
        distance_matrix = Chem.Get3DDistanceMatrix(mol=mol, confId=conf_id)
        atoms = [atom for atom in mol.GetAtoms()]
        
        clashes = []
        for i, atom1 in enumerate(atoms):
            for j, atom2 in enumerate(atoms[i+1:]):
                j = i + 1 + j
                if not ((i, j) in excluded_pairs):
                    symbol1 = atom1.GetSymbol()
                    symbol2 = atom2.GetSymbol()
                    
                    if symbol1 > symbol2:
                        symbol1, symbol2 = symbol2, symbol1
                            
                    symbol_tuple = (symbol1, symbol2)
                    
                    evaluate = True
                    if not self.consider_hs and 'H' in symbol_tuple:
                        evaluate = False
                        
                    if evaluate:
                        
                        if symbol_tuple in self.vdw_min_distances:
                            min_distance = self.vdw_min_distances[symbol_tuple]
                        else:
                            min_distance = self.get_vdw_min_distance(symbol1, symbol2)
                            self.vdw_min_distances[symbol_tuple] = min_distance
                            
                        distance = distance_matrix[i, j]
                        if distance < min_distance:
                            clash = Clash(i, symbol1, 
                                        j, symbol2, 
                                        distance, min_distance)
                            clashes.append(clash)
                        
        return clashes
        
        
    # def ymir_clash(self):
        
        
    #     ps = Chem.CombineMols(self.pocket_mol, ligand)
    #     atoms = [atom for atom in ps.GetAtoms()]
    #     n_pocket_atoms = self.pocket_mol.GetNumAtoms()
    #     pocket_atoms = atoms[:n_pocket_atoms]
    #     ligand_atoms = atoms[n_pocket_atoms:]
    #     distance_matrix = Chem.Get3DDistanceMatrix(mol=ps)
    #     ps_distance_matrix = distance_matrix[:n_pocket_atoms, n_pocket_atoms:]
        
    #     # Chem.MolToMolFile(ps, 'pocket_and_seed.mol')
    #     # import pdb;pdb.set_trace()
        
    #     # TOO SLOW
    #     # pocket_pos = self.pocket_mol.GetConformer().GetPositions()
    #     # pocket_atoms = self.pocket_mol.GetAtoms()
    #     # fragment_pos = fragment.GetConformer().GetPositions()
    #     # fragment_atoms = fragment.GetAtoms()
    #     # distance_matrix = cdist(pocket_pos, fragment_pos)
    #     # from sklearn import metrics
    #     # distance_matrix = metrics.pairwise_distances(pocket_pos, fragment_pos)
        
    #     # import pdb;pdb.set_trace()
        
    #     has_clash = False
    
    #     for idx1, atom1 in enumerate(pocket_atoms):
    #         for idx2, atom2 in enumerate(ligand_atoms):
                
    #             symbol1 = atom1.GetSymbol()
    #             symbol2 = atom2.GetSymbol()
    #             if symbol2 == 'R':
    #                 symbol2 = 'H' # we mimic that the fragment will bind to a Carbon
                    
    #             if symbol1 > symbol2:
    #                 symbol1, symbol2 = symbol2, symbol1
                
    #             if symbol1 in vdw_distances:
    #                 if symbol2 in vdw_distances[symbol1]:
    #                     min_distance = vdw_distances[symbol1][symbol2]
    #                 else:
    #                     min_distance = get_vdw_min_distance(symbol1, symbol2)
    #                     vdw_distances[symbol1][symbol2] = min_distance
    #             else:
    #                 min_distance = get_vdw_min_distance(symbol1, symbol2)
    #                 vdw_distances[symbol1][symbol2] = min_distance
                    
    #             distance = ps_distance_matrix[idx1, idx2]
    #             if distance < min_distance:
    #                 has_clash = True
    #                 break
                
    #         if has_clash:
    #             break
                    
    #     # Seed - Seed clashes
                    
    #     if not has_clash:
            
    #         ss_distance_matrix = distance_matrix[n_pocket_atoms:, n_pocket_atoms:]
            
    #         # import pdb;pdb.set_trace()

    #         bonds = self.geometry_extractor.get_bonds(ligand)
    #         bond_idxs = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    #                     for bond in bonds]
    #         bond_idxs = [t 
    #                     if t[0] < t[1] 
    #                     else (t[1], t[0])
    #                     for t in bond_idxs ]
    #         angle_idxs = self.geometry_extractor.get_angles_atom_ids(ligand)
    #         two_hop_idxs = [(t[0], t[2])
    #                                 if t[0] < t[2]
    #                                 else (t[2], t[0])
    #                                 for t in angle_idxs]
    #         torsion_idxs = self.geometry_extractor.get_torsions_atom_ids(ligand)
    #         three_hop_idxs = [(t[0], t[3])
    #                             if t[0] < t[3]
    #                             else (t[3], t[0])
    #                             for t in torsion_idxs]

    #         for idx1, atom1 in enumerate(ligand_atoms):
    #             for idx2, atom2 in enumerate(ligand_atoms[idx1+1:]):
    #                 idx2 = idx2 + idx1 + 1
    #                 not_bond = (idx1, idx2) not in bond_idxs
    #                 not_angle = (idx1, idx2) not in two_hop_idxs
    #                 not_torsion = (idx1, idx2) not in three_hop_idxs
    #                 if not_bond and not_angle and not_torsion:
    #                     symbol1 = atom1.GetSymbol()
    #                     symbol2 = atom2.GetSymbol()
    #                     if symbol1 == 'R':
    #                         symbol1 = 'H'
    #                     if symbol2 == 'R':
    #                         symbol2 = 'H'
                            
    #                     if symbol1 > symbol2:
    #                         symbol1, symbol2 = symbol2, symbol1
                        
    #                     if symbol1 in vdw_distances:
    #                         if symbol2 in vdw_distances[symbol1]:
    #                             min_distance = vdw_distances[symbol1][symbol2]
    #                         else:
    #                             min_distance = get_vdw_min_distance(symbol1, symbol2)
    #                             vdw_distances[symbol1][symbol2] = min_distance
    #                     else:
    #                         min_distance = get_vdw_min_distance(symbol1, symbol2)
    #                         vdw_distances[symbol1][symbol2] = min_distance
                            
    #                     distance = ss_distance_matrix[idx1, idx2]
    #                     if distance < min_distance:
    #                         has_clash = True
    #                         break
                    
    #             if has_clash:
    #                 break
                    
    #     return has_clash