import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Tuple, List, Dict, Any
from genbench3d.geometry import (GeometryExtractor, 
                                 ReferenceGeometry, 
                                 LigBoundConfGeometry,
                                 CSDGeometry)
from rdkit import Chem
from rdkit.Chem import Mol
from ..metric import Metric
from genbench3d.conf_ensemble import GeneratedCEL
from genbench3d.utils import rdkit_conf_to_ccdc_mol
from ccdc.conformer import GeometryAnalyser
from ccdc.io import Molecule
from scipy.stats.mstats import gmean

class Validity3D(Metric):
    
    def __init__(self,
                 name: str = 'Validity3D',
                 reference_geometry: ReferenceGeometry = CSDGeometry(),
                 backend: str = 'reference',
                 q_value_threshold: float = 0.001,
                 include_torsions: float = False,
                 ) -> None:
        super().__init__(name)
        self.geometry_extractor = GeometryExtractor()
        self.reference_geometry = reference_geometry
        assert backend in ['reference', 'CSD']
        self.backend = backend
        if backend == 'CSD':
            self.geometry_analyser = GeometryAnalyser()
        self.q_value_threshold = q_value_threshold
        self.include_torsions = include_torsions
        
        self.valid_conf_ids = {}
        self.new_patterns = {}
        self.validities = {}
        self.clashes = {}
        self.n_valid_confs = None
        self.invalidity_df = None
    
    
    def get(self,
            cel: GeneratedCEL) -> float:
        
        self.n_valid_confs = 0
        # for name, ce in tqdm(cel.items()):
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
                                analyze_torsions: bool = True,
                                ) -> List[int]:
        assert name is not None, 'Name must be given'
        assert name not in self.valid_conf_ids, 'Name already tested'
        
        if self.backend == 'reference':
        
            valid_bond_conf_ids, bond_validities, new_bond_patterns = self.analyze_bonds(mol)
            valid_conf_ids = set(valid_bond_conf_ids)
            validities = list(bond_validities)
            new_patterns = [('bond', p) for p in new_bond_patterns]
            
            valid_angle_conf_ids, angle_validities, new_angle_patterns = self.analyze_angles(mol)
            valid_conf_ids = valid_conf_ids.intersection(valid_angle_conf_ids)
            validities.extend(angle_validities)
            new_patterns.extend([('angle', p) for p in new_angle_patterns])
            
            if analyze_torsions: # this is slower than bonds and angles
                valid_torsion_conf_ids, torsion_validities, new_torsion_patterns = self.analyze_torsions(mol)
                validities.extend(torsion_validities)
                new_patterns.extend([('torsion', p) for p in new_torsion_patterns])
                if self.include_torsions:
                    valid_conf_ids = valid_conf_ids.intersection(valid_torsion_conf_ids)
                
        elif self.backend == 'CSD':
            
            valid_conf_ids = []
            for conf in mol.GetConformers():
                conf_id = conf.GetId()
                molecule = rdkit_conf_to_ccdc_mol(rdkit_mol=mol, 
                                                  conf_id=conf_id)
                molecule.assign_bond_types(which='unknown')
                molecule.standardise_aromatic_bonds()
                molecule.standardise_delocalised_bonds()
                molecule.add_hydrogens()
                analysed_molecule: Molecule = self.geometry_analyser.analyse_molecule(molecule)
                conf_bond_invalidities = [{'conf_id': conf_id,
                                           'geometry_type': 'bond',
                                           'string': bond.fragment_label,
                                           'value': bond.value,
                                           'z-score': bond.z_score}
                                          for bond in analysed_molecule.analysed_bonds
                                          if bond.unusual]
                conf_angle_invalidities = [{'conf_id': conf_id,
                                           'geometry_type': 'angle',
                                           'string': angle.fragment_label,
                                           'value': angle.value,
                                           'z-score': angle.z_score}
                                          for angle in analysed_molecule.analysed_angles
                                          if angle.unusual]
                conf_torsion_invalidities = [{'conf_id': conf_id,
                                           'geometry_type': 'torsion',
                                           'string': torsion.fragment_label,
                                           'value': torsion.value}
                                          for torsion in analysed_molecule.analysed_torsions
                                          if torsion.unusual]
                conf_ring_invalidities = [{'conf_id': conf_id,
                                           'geometry_type': 'ring',
                                           'string': ring.fragment_label,
                                           'value': ring.value}
                                          for ring in analysed_molecule.analysed_rings
                                          if ring.unusual]
                conf_invalidities = (conf_bond_invalidities 
                                     + conf_angle_invalidities
                                     + conf_torsion_invalidities
                                     + conf_ring_invalidities)
                n_invalidities = len(conf_invalidities)
                is_valid = n_invalidities == 0
                if is_valid: 
                    valid_conf_ids.append(conf_id)
                    
        else:
            print('Invalid backend')
        
        valid_nonbonddistance_conf_ids, clashes = self.analyze_clashes(mol)
        valid_conf_ids = valid_conf_ids.intersection(valid_nonbonddistance_conf_ids)
        
        # validity = len(valid_conf_ids) / mol.GetNumConformers()
        
        self.valid_conf_ids[name] = valid_conf_ids
        self.validities[name] = validities
        self.new_patterns[name] = new_patterns
        self.clashes[name] = clashes
        
        return valid_conf_ids
    
    
    def analyze_bonds(self,
                      mol: Mol,
                      ) -> Tuple[List[int], 
                                 List[Dict[str, Any]], 
                                 List[str]]:
        bonds = self.geometry_extractor.get_bonds(mol, 
                                                  consider_hydrogens=False)
        
        valid_bond_conf_ids = []
        bond_validities = []
        new_bond_patterns = set()
        for conf in mol.GetConformers():
            conf_is_valid = True
            conf_id = conf.GetId()
        
            for bond in bonds:
                bond_tuple = self.geometry_extractor.get_bond_tuple(bond)
                bond_value = self.geometry_extractor.get_bond_length(conf, bond)
                
                q_value, new_pattern = self.reference_geometry.geometry_is_valid(geometry_tuple=bond_tuple,
                                                                                value=bond_value,
                                                                                geometry='bond')
                
                if not new_pattern:
                    bond_is_valid = q_value > self.q_value_threshold
                else: # When pattern is new, we cannot state on validity, default behaviour is to accept
                    assert np.isnan(q_value)
                    bond_is_valid = True
                
                validity_d = {'conf_id': conf_id,
                            'geometry_type': 'bond',
                            'tuple': bond_tuple,
                            'value': bond_value,
                            'q-value': q_value,
                            }
                
                bond_validities.append(validity_d)
                
                if not bond_is_valid:
                    # import pdb;pdb.set_trace()
                    conf_is_valid = False
                    
                if new_pattern:
                    new_bond_patterns.update(bond_tuple)
            
            if conf_is_valid:
                valid_bond_conf_ids.append(conf_id)
                
        return valid_bond_conf_ids, bond_validities, new_bond_patterns
    
    
    def analyze_angles(self,
                       mol: Mol,
                       ) -> Tuple[List[int], 
                                 List[Dict[str, Any]], 
                                 List[str]]:
        
        angles_atom_ids = self.geometry_extractor.get_angles_atom_ids(mol,
                                                                      consider_hydrogens=False)
        
        valid_angle_conf_ids = []
        angle_validities = []
        new_angle_patterns = []
        for conf in mol.GetConformers():
            conf_is_valid = True
            conf_id = conf.GetId()
        
            for begin_atom_idx, second_atom_idx, end_atom_idx in angles_atom_ids:
                angle_tuple = self.geometry_extractor.get_angle_tuple(mol, 
                                                                        begin_atom_idx, 
                                                                        second_atom_idx, 
                                                                        end_atom_idx)
                angle_value = self.geometry_extractor.get_angle_value(conf, 
                                                                        begin_atom_idx, 
                                                                        second_atom_idx, 
                                                                        end_atom_idx)
                
                q_value, new_pattern = self.reference_geometry.geometry_is_valid(geometry_tuple=angle_tuple,
                                                                                value=angle_value,
                                                                                geometry='angle')
                
                if not new_pattern:
                    angle_is_valid = q_value > self.q_value_threshold
                else: # When pattern is new, we cannot state on validity, default behaviour is to accept
                    assert np.isnan(q_value)
                    angle_is_valid = True
                
                validity_d = {
                        'conf_id': conf_id,
                        'geometry_type': 'angle',
                        'tuple': angle_tuple,
                        'value': angle_value,
                        'q-value': q_value,
                    }
                angle_validities.append(validity_d)
                
                if not angle_is_valid:
                    # import pdb;pdb.set_trace()
                    conf_is_valid = False
                    
                if new_pattern:
                    new_angle_patterns.append(angle_tuple)
            
            if conf_is_valid:
                valid_angle_conf_ids.append(conf_id)
                
        return valid_angle_conf_ids, angle_validities, new_angle_patterns
        
    
    def analyze_torsions(self,
                       mol: Mol,
                       ) -> Tuple[List[int], 
                                 List[Dict[str, Any]], 
                                 List[str]]:
        
        torsions_atom_ids = self.geometry_extractor.get_torsions_atom_ids(mol,
                                                                          consider_hydrogens=False)
        
        valid_torsion_conf_ids = []
        torsion_validities = []
        new_torsion_patterns = []
        for conf in mol.GetConformers():
            conf_is_valid = True
            conf_id = conf.GetId()
        
            for begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx in torsions_atom_ids:
                torsion_tuple = self.geometry_extractor.get_torsion_tuple(mol, 
                                                                        begin_atom_idx, 
                                                                        second_atom_idx, 
                                                                        third_atom_idx,
                                                                        end_atom_idx)
                torsion_value = self.geometry_extractor.get_torsion_value(conf, 
                                                                        begin_atom_idx, 
                                                                        second_atom_idx, 
                                                                        third_atom_idx,
                                                                        end_atom_idx)
                
                q_value, new_pattern = self.reference_geometry.geometry_is_valid(geometry_tuple=torsion_tuple,
                                                                                value=torsion_value,
                                                                                geometry='torsion')
                
                if not new_pattern:
                    torsion_is_valid = q_value > self.q_value_threshold
                else: # When pattern is new, we cannot state on validity, default behaviour is to accept
                    assert np.isnan(q_value)
                    torsion_is_valid = True
                
                validity_d = {
                        'conf_id': conf_id,
                        'geometry_type': 'torsion',
                        'tuple': torsion_tuple,
                        'value': torsion_value,
                        'q-value': q_value
                    }
                torsion_validities.append(validity_d)
                
                if not torsion_is_valid:
                    conf_is_valid = False
                    
                if new_pattern:
                    new_torsion_patterns.append(torsion_tuple)
            
            if conf_is_valid:
                valid_torsion_conf_ids.append(conf_id)
                
        return valid_torsion_conf_ids, torsion_validities, new_torsion_patterns
    
    
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
            
            # import pdb;pdb.set_trace()
            
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
                                min_distance = (vdw1 + vdw2) * 0.75
                                
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
        for name, validity_list in self.validities.items():
            # columns: name, conf_id, geometry_type, string, value
            validity_df = pd.DataFrame(validity_list) 
            if len(validity_df) > 0:
                invalidity_df = validity_df[validity_df['q-value'] < self.q_value_threshold].copy()
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
        
        # self.median_n_inv_bonds = np.nanmedian(all_n_inv_bonds)
        # self.median_n_inv_angles = np.nanmedian(all_n_inv_angles)
        
        return all_n_inv_bonds, all_n_inv_angles
    
    
    def get_min_q_values_for_geometry(self,
                                     geometry_type: str = 'bond'):
        return self.get_agg_q_values_for_geometries(agg_func=np.min,
                                                  geometry_types=[geometry_type])
    
    
    def get_geo_mean_q_values_for_geometry(self,
                                          geometry_type: str = 'bond'):
        return self.get_agg_q_values_for_geometries(agg_func=gmean,
                                                  geometry_types=[geometry_type])
    
    
    def get_min_q_values_for_geometries(self,
                                        geometry_types: list[str] = ['bond', 'angle', 'torsion']):
        return self.get_agg_q_values_for_geometries(agg_func=np.min,
                                                  geometry_types=geometry_types)
    
    
    def get_geo_mean_q_values_for_geometries(self,
                                            geometry_types: list[str] = ['bond', 'angle', 'torsion']):
        return self.get_agg_q_values_for_geometries(agg_func=gmean,
                                                  geometry_types=geometry_types)
    
    
    def get_agg_q_values_for_geometries(self,
                                        agg_func: callable,
                                        geometry_types: list[str] = ['bond', 'angle', 'torsion']):
        
        agg_q_values = []
        for name, validity_list in self.validities.items():
            validity_df = pd.DataFrame(validity_list) 
            if len(validity_df) > 0:
                geometry_validity_df = validity_df[validity_df['geometry_type'].isin(geometry_types)]
                geometry_validity_df = geometry_validity_df.dropna(subset=['q-value'])
                if geometry_validity_df.shape[0] > 1:
                    try:
                        q_value_series = geometry_validity_df.groupby('conf_id')['q-value'].agg(agg_func)
                    except Exception as e:
                        print(e)
                        import pdb;pdb.set_trace()
                    agg_q_values.extend(q_value_series.values)
                else:
                    # import pdb;pdb.set_trace()
                    agg_q_values.extend([np.nan] * len(validity_df['conf_id'].unique()))
        
        return agg_q_values
        