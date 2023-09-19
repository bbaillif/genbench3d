import numpy as np
import pandas as pd

from typing import Tuple, List, Dict, Any
from genbench3d.geometry import GeometryExtractor, ReferenceGeometry, LigBoundConfGeometry
from rdkit import Chem
from rdkit.Chem import Mol
from ..metric import Metric
from genbench3d.conf_ensemble import GeneratedCEL

class Validity3D(Metric):
    
    def __init__(self,
                 name: str = 'Validity3D',
                 reference_geometry: ReferenceGeometry = LigBoundConfGeometry(),
                 clash_tolerance: float = 0.5 # Angstrom
                 ) -> None:
        super().__init__(name)
        self.geometry_extractor = GeometryExtractor()
        self.reference_geometry = reference_geometry
        self.clash_tolerance = clash_tolerance
        self.valid_conf_ids = {}
        self.invalidities = {}
        self.new_patterns = {}
        self.validities = {}
        self.clashes = {}
        self.n_valid_confs = None
        self.invalidity_df = None
    
    
    def get(self,
            cel: GeneratedCEL) -> float:
        self.n_valid_confs = 0
        for name, ce in cel.items():
            mol = ce.mol
            valid_conf_ids = self.get_valid_conf_ids_for_mol(mol, 
                                                            name)
            if len(valid_conf_ids) > 0:
                self.n_valid_confs += len(valid_conf_ids)
        self.value = self.n_valid_confs / cel.n_total_confs
        
        return self.value
    
    
    def get_valid_conf_ids_for_mol(self, 
                                mol: Mol,
                                name: str = None,
                                # complete_evaluation: bool = True,
                                analyze_torsions: bool = False,
                                ) -> List[int]:
        assert name is not None, 'Name must be given'
        assert name not in self.valid_conf_ids, 'Name already tested'
        
        valid_bond_conf_ids, bond_invalidities, new_bond_patterns = self.analyze_bonds(mol)
        valid_conf_ids = set(valid_bond_conf_ids)
        invalidities = list(bond_invalidities)
        new_patterns = [('bond', p) for p in new_bond_patterns]
        
        valid_angle_conf_ids, angle_invalidities, new_angle_patterns = self.analyze_angles(mol)
        valid_conf_ids = valid_conf_ids.intersection(valid_angle_conf_ids)
        invalidities.extend(angle_invalidities)
        new_patterns.extend([('angle', p) for p in new_angle_patterns])
        
        if analyze_torsions: # this is slower than bonds and angles
            valid_torsion_conf_ids, torsion_invalidities, new_torsion_patterns = self.analyze_torsions(mol)
            valid_conf_ids = valid_conf_ids.intersection(valid_torsion_conf_ids)
            invalidities.extend(torsion_invalidities)
            new_patterns.extend([('torsion', p) for p in new_torsion_patterns])
        
        valid_nonbonddistance_conf_ids, clashes = self.analyze_clashes(mol)
        valid_conf_ids = valid_conf_ids.intersection(valid_nonbonddistance_conf_ids)
        
        validity = len(valid_conf_ids) / mol.GetNumConformers()
        
        self.valid_conf_ids[name] = valid_conf_ids
        self.invalidities[name] = invalidities
        self.new_patterns[name] = new_patterns
        self.validities[name] = validity
        self.clashes[name] = clashes
        
        return valid_conf_ids
    
    
    def analyze_bonds(self,
                      mol: Mol,
                      ) -> Tuple[List[int], 
                                 List[Dict[str, Any]], 
                                 List[str]]:
        bonds = self.geometry_extractor.get_bonds(mol)
        
        valid_bond_conf_ids = []
        bond_invalidies = []
        new_bond_patterns = set()
        for conf in mol.GetConformers():
            conf_is_valid = True
            conf_id = conf.GetId()
        
            for bond in bonds:
                bond_string = self.geometry_extractor.get_bond_string(bond)
                bond_value = self.geometry_extractor.get_bond_length(conf, bond)
                
                bond_is_valid, new_pattern = self.reference_geometry.geometry_is_valid(string=bond_string,
                                                                    value=bond_value,
                                                                    geometry='bond')
                
                if not bond_is_valid:
                    invalid_d = {
                        'conf_id': conf_id,
                        'geometry_type': 'bond',
                        'string': bond_string,
                        'value': bond_value
                    }
                    bond_invalidies.append(invalid_d)
                    conf_is_valid = False
                    
                if new_pattern:
                    new_bond_patterns.update(bond_string)
            
            if conf_is_valid:
                valid_bond_conf_ids.append(conf_id)
                
        return valid_bond_conf_ids, bond_invalidies, new_bond_patterns
    
    
    def analyze_angles(self,
                       mol: Mol,
                       ) -> Tuple[List[int], 
                                 List[Dict[str, Any]], 
                                 List[str]]:
        
        angles_atom_ids = self.geometry_extractor.get_angles_atom_ids(mol)
        
        valid_angle_conf_ids = []
        angle_invalidies = []
        new_angle_patterns = []
        for conf in mol.GetConformers():
            conf_is_valid = True
            conf_id = conf.GetId()
        
            for begin_atom_idx, second_atom_idx, end_atom_idx in angles_atom_ids:
                angle_string = self.geometry_extractor.get_angle_string(mol, 
                                                                        begin_atom_idx, 
                                                                        second_atom_idx, 
                                                                        end_atom_idx)
                angle_value = self.geometry_extractor.get_angle_value(conf, 
                                                                        begin_atom_idx, 
                                                                        second_atom_idx, 
                                                                        end_atom_idx)
                
                angle_is_valid, new_pattern = self.reference_geometry.geometry_is_valid(string=angle_string,
                                                                                    value=angle_value,
                                                                                    geometry='angle')
                
                if not angle_is_valid:
                    invalid_d = {
                        'conf_id': conf_id,
                        'geometry_type': 'angle',
                        'string': angle_string,
                        'value': angle_value
                    }
                    angle_invalidies.append(invalid_d)
                    conf_is_valid = False
                    
                if new_pattern:
                    new_angle_patterns.append(angle_string)
            
            if conf_is_valid:
                valid_angle_conf_ids.append(conf_id)
                
        return valid_angle_conf_ids, angle_invalidies, new_angle_patterns
        
    
    def analyze_torsions(self,
                       mol: Mol,
                       ) -> Tuple[List[int], 
                                 List[Dict[str, Any]], 
                                 List[str]]:
        
        torsions_atom_ids = self.geometry_extractor.get_torsions_atom_ids(mol)
        
        valid_torsion_conf_ids = []
        torsion_invalidies = []
        new_torsion_patterns = []
        for conf in mol.GetConformers():
            conf_is_valid = True
            conf_id = conf.GetId()
        
            for begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx in torsions_atom_ids:
                torsion_string = self.geometry_extractor.get_torsion_string(mol, 
                                                                        begin_atom_idx, 
                                                                        second_atom_idx, 
                                                                        third_atom_idx,
                                                                        end_atom_idx)
                torsion_value = self.geometry_extractor.get_torsion_value(conf, 
                                                                        begin_atom_idx, 
                                                                        second_atom_idx, 
                                                                        third_atom_idx,
                                                                        end_atom_idx)
                
                torsion_is_valid, new_pattern = self.reference_geometry.geometry_is_valid(string=torsion_string,
                                                                                    value=torsion_value,
                                                                                    geometry='torsion')
                
                if not torsion_is_valid:
                    invalid_d = {
                        'conf_id': conf_id,
                        'geometry_type': 'torsion',
                        'string': torsion_string,
                        'value': torsion_value
                    }
                    torsion_invalidies.append(invalid_d)
                    conf_is_valid = False
                    
                if new_pattern:
                    new_torsion_patterns.append(torsion_string)
            
            if conf_is_valid:
                valid_torsion_conf_ids.append(conf_id)
                
        return valid_torsion_conf_ids, torsion_invalidies, new_torsion_patterns
    
    
    def analyze_clashes(self,
                        mol: Mol) -> Tuple[List[int], 
                                            Dict[str, Any]]:
        
        try:
        
            clashes = []
            valid_interbonddist_conf_ids = []
            
            bonds = self.geometry_extractor.get_bonds(mol)
            bond_idxs = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        for bond in bonds]
            bond_idxs = [t 
                        if t[0] < t[1] 
                        else (t[1], t[0])
                        for t in bond_idxs ]
            angle_idxs = self.geometry_extractor.get_angles_atom_ids(mol)
            two_hop_idxs = [(t[0], t[2])
                                      if t[0] < t[2]
                                      else (t[2], t[0])
                                      for t in angle_idxs]
            torsion_idxs = self.geometry_extractor.get_torsions_atom_ids(mol)
            three_hop_idxs = [(t[0], t[3])
                                if t[0] < t[3]
                                else (t[3], t[0])
                                for t in torsion_idxs]
            atoms = [atom for atom in mol.GetAtoms()]
            for conf in mol.GetConformers():
                conf_id = conf.GetId()
                distance_matrix = Chem.Get3DDistanceMatrix(mol=mol, confId=conf_id)
                is_clashing = False
            
                for i, atom1 in enumerate(atoms):
                    idx1 = atom1.GetIdx()
                    if i != idx1:
                        print('Check atom indices')
                        import pdb;pdb.set_trace()
                    symbol1 = atom1.GetSymbol()
                    for j, atom2 in enumerate(atoms[i+1:]):
                        idx2 = atom2.GetIdx()
                        if i+1+j != idx2:
                            print('Check atom indices')
                            import pdb;pdb.set_trace()
                        symbol2 = atom2.GetSymbol()
                        not_bond = (idx1, idx2) not in bond_idxs
                        not_angle = (idx1, idx2) not in two_hop_idxs
                        not_torsion = (idx1, idx2) not in three_hop_idxs
                        if not_bond and not_angle and not_torsion:
                            vdw1 = self.geometry_extractor.get_vdw_radius(symbol1)
                            vdw2 = self.geometry_extractor.get_vdw_radius(symbol2)
                            
                            if symbol1 == 'H':
                                min_distance = vdw2
                            elif symbol2 == 'H':
                                min_distance = vdw1
                            else:
                                # min_distance = vdw1 + vdw2 - self.clash_tolerance
                                min_distance = (vdw1 + vdw2) * 0.8
                                
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
                                clashes.append(invalid_d)
                
                if not is_clashing:
                    valid_interbonddist_conf_ids.append(conf_id)
                    
        except Exception as e:
            print(e)
            import pdb;pdb.set_trace()
                
        return valid_interbonddist_conf_ids, clashes
            
    
    def get_invalid_bonds_angles(self):
        all_n_inv_bonds = []
        all_n_inv_angles = []
        invalidity_dfs = []
        for name, invalidity_list in self.invalidities.items():
            # columns: name, conf_id, geometry_type, string, value
            invalidity_df = pd.DataFrame(invalidity_list) 
            invalidity_df['name'] = name
            if invalidity_df.shape[0] > 0:
                invalid_numbers_df = invalidity_df.pivot_table(values='value', 
                                                                index=['name', 'conf_id'],
                                                                columns=['geometry_type'],
                                                                aggfunc='count')
                if 'bond' in invalid_numbers_df.columns:
                    all_n_inv_bonds.extend(invalid_numbers_df['bond'].values)
                if 'angle' in invalid_numbers_df.columns: 
                    all_n_inv_angles.extend(invalid_numbers_df['angle'].values)
                
                invalidity_dfs.append(invalid_numbers_df)
            
        if len(invalidity_dfs) > 0:
            self.invalidity_df = pd.concat(invalidity_dfs)
                    
            # if self.show_plots:
            #     sns.histplot(x=n_invalid_bonds_l)
            #     plt.xlabel('Number of invalid bonds')
            #     plt.show()
            
            #     sns.histplot(x=n_invalid_angles_l)
            #     plt.xlabel('Number of invalid angles')
            #     plt.show()
        
        self.median_n_inv_bonds = np.nanmedian(all_n_inv_bonds)
        self.median_n_inv_angles = np.nanmedian(all_n_inv_angles)
        
        return self.median_n_inv_bonds, self.median_n_inv_angles