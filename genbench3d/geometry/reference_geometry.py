import os
import numpy as np
import pickle
# import logging

from abc import abstractmethod, ABC
from typing import List, Dict
from tqdm import tqdm
from collections import defaultdict
from genbench3d.params import DATA_DIRPATH
from rdkit.Chem import Mol
from .geometry_extractor import GeometryExtractor


class ReferenceGeometry(ABC):
    
    def __init__(self,
                 root: str = DATA_DIRPATH,
                 source_name: str = None) -> None:
        self.root = root
        self.source_name = source_name
        
        self.geometry_extractor = GeometryExtractor()
        
        self.values_filepath = os.path.join(root, f'{source_name}_geometry_values.p')
        self.ranges_filepath = os.path.join(root, f'{source_name}_geometry_ranges.p')
        if (not os.path.exists(self.values_filepath)) and (not os.path.exists(self.ranges_filepath)):
            self.compute_ranges()
        self.ranges = self.read_ranges()
    
    
    @abstractmethod
    def load_ligands(self) -> List[Mol]:
        raise NotImplementedError()
    
    
    def read_values(self):
        with open(self.values_filepath, 'rb') as f:
            values = pickle.load(f)
        return values
    
    
    def read_ranges(self):
        with open(self.ranges_filepath, 'rb') as f:
            ranges = pickle.load(f)
        return ranges
    
    
    def compute_ranges(self):
        
        print(f'Compiling geometry ranges for {self.source_name}')
        
        ligands = self.load_ligands()
        bond_values = self.compile_bond_lengths(ligands)
        angle_values = self.compile_angle_values(ligands)
        torsion_values = self.compile_torsion_values(ligands)
        values = {'bond_length': bond_values,
                  'angle_values': angle_values,
                  'torsion_values': torsion_values}
        
        with open(self.values_filepath, 'wb') as f:
            pickle.dump(values, f)
        
        ranges = self.get_ranges_from_values(bond_values,
                                             angle_values,
                                             torsion_values)
        
        with open(self.ranges_filepath, 'wb') as f:
            pickle.dump(ranges, f)
            
            
    def get_ranges_from_values(self,
                               bond_values: Dict[str, List[float]],
                                angle_values: Dict[str, List[float]],
                                torsion_values: Dict[str, List[float]]):
        
        bond_ranges = {}
        for bond_string, values in bond_values.items():
            ranges = self.compute_authorized_ranges_bond(values)
            bond_ranges[bond_string] = ranges
        
        angle_ranges = {}
        for angle_string, values in angle_values.items():
            ranges = self.compute_authorized_ranges_angle(values)
            angle_ranges[angle_string] = ranges
            
        torsion_ranges = {}
        for torsion_string, values in torsion_values.items():
            ranges = self.compute_authorized_ranges_torsion(values)
            torsion_ranges[torsion_string] = ranges
        
        ranges = {
            'bond_values': bond_ranges,
            'angle_values': angle_ranges,
            'torsion_values': torsion_ranges,
            }
        
        return ranges
              
                        
    def compile_bond_lengths(self,
                             ligands: List[Mol],
                             ) -> Dict[str, List[float]]:
        
        print('Compiling bond lengths')
        
        bond_values = defaultdict(list)
        for i, mol in enumerate(tqdm(ligands)):
            for bond in mol.GetBonds():
                bond_string = self.geometry_extractor.get_bond_string(bond)
                for conf in mol.GetConformers():
                    bond_length = self.geometry_extractor.get_bond_length(conf, bond)
                    bond_values[bond_string].append(bond_length)
                    
        return bond_values
    
    
    def compile_angle_values(self,
                              ligands: List[Mol],
                             ) -> Dict[str, List[float]]:
        
        print('Compiling valence angle values')
        
        angle_values = defaultdict(list)
        for i, mol in enumerate(tqdm(ligands)):
            angles_atom_ids = self.geometry_extractor.get_angles_atom_ids(mol)
            for begin_atom_idx, second_atom_idx, end_atom_idx in angles_atom_ids :
                
                angle_string = self.geometry_extractor.get_angle_string(mol, 
                                                                        begin_atom_idx, 
                                                                        second_atom_idx, 
                                                                        end_atom_idx)
                    
                for conf in mol.GetConformers():
                    angle_value = self.geometry_extractor.get_angle_value(conf,
                                                                          begin_atom_idx,
                                                                          second_atom_idx,
                                                                          end_atom_idx)
                    angle_values[angle_string].append(angle_value)
                    
        return angle_values
                    
                    
    def compile_torsion_values(self,
                              ligands: List[Mol],
                             ) -> Dict[str, List[float]]:
        
        print('Compiling torsion angle values')
        
        torsion_values = defaultdict(list)
        for i, mol in enumerate(tqdm(ligands)):
            torsion_atom_ids = self.geometry_extractor.get_torsions_atom_ids(mol)
            for begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx in torsion_atom_ids :
                
                torsion_string = self.geometry_extractor.get_torsion_string(mol,
                                                                            begin_atom_idx,
                                                                            second_atom_idx,
                                                                            third_atom_idx,
                                                                            end_atom_idx)
                    
                for conf in mol.GetConformers():
                    torsion_value = self.geometry_extractor.get_torsion_value(conf,
                                                                              begin_atom_idx,
                                                                              second_atom_idx,
                                                                              third_atom_idx,
                                                                              end_atom_idx)
                    if not np.isnan(torsion_value):
                        torsion_values[torsion_string].append(torsion_value)
                        
        return torsion_values
    
    
    def compute_authorized_ranges_angle(self,
                                        values, 
                                        nbins: int = 36, 
                                        binrange: List[float] = [0.0, 180.0], 
                                        authorized_distance: float = 2.5 # Deg
                                        ) -> List[tuple]:
        if len(values) > 20:
            counts, binticks = np.histogram(values, 
                                            bins=nbins, 
                                            range=binrange)
            authorized_ranges = []
            in_range = False # cursor to indicate whether we are inside possible value range
            # We scan all bins of the histogram, and merge bins accordinly to create ranges
            for i, (value, tick) in enumerate(zip(counts, binticks)) :
                if value > 0:
                    if not in_range: # if we have values and we were not in range, we enter in range
                        start = tick
                        in_range = True
                else:
                    if in_range: # if we have no values and we were in range, we exit range
                        end = tick
                        authorized_ranges.append((start, end))
                        in_range = False
            if in_range:
                end = binticks[-1]
                authorized_ranges.append((start, end))

            new_ranges = []
            for start, end in authorized_ranges:
                new_start = max(start - authorized_distance, binrange[0])
                new_end = min(end + authorized_distance, binrange[1])
                new_range = (new_start, new_end)
                new_ranges.append(new_range)

            corrected_ranges = []
            previous_range = new_ranges[0]
            for new_range in new_ranges[1:]:
                if new_range[0] <= previous_range[1]:
                    previous_range = (previous_range[0], new_range[1])
                else:
                    corrected_ranges.append(previous_range)
                    previous_range = new_range
            corrected_ranges.append(previous_range)

            return corrected_ranges
        else:
            return [tuple(binrange)]
        
    
    def compute_authorized_ranges_bond(self,
                                       values, 
                                       authorized_distance: float = 0.025):
        min_value = np.around(np.min(values) - authorized_distance, 3)
        max_value = np.around(np.max(values) + authorized_distance, 3)
        return [(min_value, max_value)]
    
    
    def compute_authorized_ranges_torsion(self,
                                          values, 
                                        nbins: int = 36, 
                                        binrange: List[float] = [0.0, 180.0], 
                                        authorized_distance: float = 2.5,
                                        absolute_torsion: bool = True):
        if absolute_torsion:
            values = np.abs(values)
        return self.compute_authorized_ranges_angle(values, 
                                                    nbins,
                                                    binrange, 
                                                    authorized_distance)
        
    
    def geometry_is_valid(self, 
                          string: str, 
                          value: float, 
                          geometry: str = 'bond',
                          new_pattern_is_valid: bool = True,
                          ) -> bool:
        
        assert geometry in ['bond', 'angle', 'torsion']
        
        if geometry == 'bond':
            ranges = self.ranges['bond_values']
        elif geometry == 'angle' :
            ranges = self.ranges['angle_values']
        elif geometry == 'torsion':
            ranges = self.ranges['torsion_values']
            value = np.abs(value) # this method is done on absolute torsion ranges [0, 180]
        else:
            raise RuntimeError()
        
        in_range = False
        new_pattern = False
        if string in ranges:
            current_ranges = ranges[string]

            for range in current_ranges:
                if value >= range[0] and value <= range[1]:
                    in_range = True
                    break
                
        else:
            new_pattern = True
            if new_pattern_is_valid:
                in_range = True
            
        return in_range, new_pattern
    
        