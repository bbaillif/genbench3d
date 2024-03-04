import numpy as np

from rdkit.Chem import (Bond, 
                        Mol,
                        Conformer,
                        GetPeriodicTable,
                        Atom)
from rdkit.Chem.rdMolTransforms import (GetBondLength, 
                                        GetAngleDeg, 
                                        GetDihedralDeg)
from rdkit.Chem.rdmolops import FindAllPathsOfLengthN

class GeometryExtractor():
    
    def __init__(self) -> None:
        self.periodic_table = GetPeriodicTable()
    
    
    def get_vdw_radius(self,
                       symbol: str):
        return self.periodic_table.GetRvdw(symbol)
    
    @staticmethod
    def get_bond_order_str(bond_order: float) -> str:
        if bond_order == 1:
            return '-'
        elif bond_order == 2:
            return '='
        elif bond_order == 3:
            return '#'
        else :
            return '~'
        
        
    # @staticmethod
    # def get_atom_tuple(atom: Atom) -> tuple[str, str, int]:
    #     atom_symbol = atom.GetSymbol()
    #     atom_hybridization = atom.GetHybridization()
    #     atom_charge = atom.GetFormalCharge()
    #     atom_tuple = (atom_symbol, str(atom_hybridization), atom_charge)
    #     return atom_tuple
    
    @staticmethod
    def get_atom_tuple(atom: Atom) -> tuple[str, int]:
        atom_symbol = atom.GetSymbol()
        atom_charge = atom.GetFormalCharge()
        atom_tuple = (atom_symbol, atom_charge)
        return atom_tuple
    
    
    def get_neighbor_tuple(self,
                           atom: Atom,
                           excluded_atoms: list[Atom]) -> tuple:
        excluded_atom_idxs = [a.GetIdx() for a in excluded_atoms]
        neighbors = atom.GetNeighbors()
        neighbors = [neighbor 
                    for neighbor in neighbors 
                    if neighbor.GetIdx() not in excluded_atom_idxs]
        neighbor_symbols = [atom.GetSymbol() for atom in neighbors]
        # neighbor_symbols_1 = np.array(neighbor_symbols_1)
        # order = neighbor_symbols_1.argsort()
        # sorted_neighbors_1 = neighbor_symbols_1[order]
        
        mol = atom.GetOwningMol()
        neighbor_bonds = [mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                            for neighbor in neighbors]
        neighbor_bond_types = [bond.GetBondTypeAsDouble() for bond in neighbor_bonds]
        # neighbor_1_bonds = np.array()
        sorting_idx = np.lexsort((neighbor_bond_types, neighbor_symbols))
        # neighbor_tuple = tuple((self.get_atom_tuple(neighbors[i]), 
        #                         self.get_bond_order_str(neighbor_bond_types[i]))
        #                        for i in sorting_idx)
        neighbor_tuple = tuple((neighbor_symbols[i], 
                                self.get_bond_order_str(neighbor_bond_types[i]))
                               for i in sorting_idx)
        return neighbor_tuple
    
    
    def get_bonds(self,
                  mol: Mol,
                  consider_hydrogens: bool = True):
        bonds = mol.GetBonds()
        if not consider_hydrogens:
            returned_bonds = []
            for bond in bonds:
                atom_symbols = [bond.GetBeginAtom().GetSymbol(),
                                bond.GetEndAtom().GetSymbol()]
                if not 'H' in atom_symbols:
                    returned_bonds.append(bond)
        else:
            returned_bonds = bonds
            
        return returned_bonds
    
    
    def get_bond_length(self,
                        conf: Conformer,
                        bond: Bond):
        return GetBondLength(conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    
    def get_bond_tuple(self, 
                        bond: Bond):
        bond_order = bond.GetBondTypeAsDouble()
        bond_order_str = self.get_bond_order_str(bond_order)
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        atom_symbol_1 = begin_atom.GetSymbol()
        atom_symbol_2 = end_atom.GetSymbol()
        
        # TODO: detection of 3 and 4 membered-rings
        
        atom_tuple_1 = self.get_atom_tuple(begin_atom)
        atom_tuple_2 = self.get_atom_tuple(end_atom)
        
        neighbor_tuple_1 = self.get_neighbor_tuple(begin_atom, 
                                                   excluded_atoms=[end_atom])
        neighbor_tuple_2 = self.get_neighbor_tuple(end_atom,
                                                   excluded_atoms=[begin_atom])
        
        if atom_symbol_1 < atom_symbol_2:
            bond_tuple = (neighbor_tuple_1, atom_tuple_1, bond_order_str, 
                          atom_tuple_2, neighbor_tuple_2)
        else:
            bond_tuple = (neighbor_tuple_2, atom_tuple_2, bond_order_str, 
                          atom_tuple_1, neighbor_tuple_1)
            
        # import pdb;pdb.set_trace()
            
        return bond_tuple
    
    
    def get_angles_atom_ids(self,
                           mol: Mol,
                            consider_hydrogens: bool = True):
        # angles are defined by three consecutive atom in the graph, we take their idx
        paths = FindAllPathsOfLengthN(mol, 3, useBonds=False, useHs=True)
        
        if not consider_hydrogens:
            returned_tuples = []
            for path in paths:
                begin_atom_idx, second_atom_idx, end_atom_idx = path
                atom_symbols = [mol.GetAtomWithIdx(begin_atom_idx).GetSymbol(),
                                mol.GetAtomWithIdx(second_atom_idx).GetSymbol(),
                                mol.GetAtomWithIdx(end_atom_idx).GetSymbol()]
                if not 'H' in atom_symbols:
                    returned_tuples.append(path)
        else:
            returned_tuples = [tuple(path) for path in paths]
            
        return returned_tuples
    
    
    def get_angle_tuple(self, 
                         mol: Mol,
                         begin_atom_idx: int, 
                         second_atom_idx: int, 
                         end_atom_idx: int) -> str:
        begin_atom = mol.GetAtomWithIdx(begin_atom_idx)
        second_atom = mol.GetAtomWithIdx(second_atom_idx)
        end_atom = mol.GetAtomWithIdx(end_atom_idx)
        
        atom_symbol_1 = begin_atom.GetSymbol()
        bond1 = mol.GetBondBetweenAtoms(begin_atom_idx, second_atom_idx)
        bond_order_1 = bond1.GetBondTypeAsDouble()
        atom_symbol_2 = second_atom.GetSymbol()
        bond2 = mol.GetBondBetweenAtoms(second_atom_idx, end_atom_idx)
        bond_order_2 = bond2.GetBondTypeAsDouble()
        atom_symbol_3 = end_atom.GetSymbol()
        
        bond_order_str_1 = self.get_bond_order_str(bond_order_1)
        bond_order_str_2 = self.get_bond_order_str(bond_order_2)
        
        atom_tuple_1 = self.get_atom_tuple(begin_atom)
        atom_tuple_2 = self.get_atom_tuple(second_atom)
        atom_tuple_3 = self.get_atom_tuple(end_atom)
        
        neighbor_tuple_1 = self.get_neighbor_tuple(begin_atom, 
                                                   excluded_atoms=[end_atom, second_atom])
        neighbor_tuple_2 = self.get_neighbor_tuple(second_atom,
                                                   excluded_atoms=[begin_atom, end_atom])
        neighbor_tuple_3 = self.get_neighbor_tuple(end_atom,
                                                   excluded_atoms=[begin_atom, second_atom])
        
        ascending_order = None
        if atom_symbol_1 < atom_symbol_3:
            ascending_order = True
        elif atom_symbol_1 == atom_symbol_3:
            if bond_order_1 < bond_order_2:
                ascending_order = True
            else:
                ascending_order = False
        else:
            ascending_order = False
            
        if ascending_order:
            angle_tuple = (neighbor_tuple_1, atom_tuple_1, bond_order_str_1, 
                           atom_tuple_2, neighbor_tuple_2,
                           bond_order_str_2, atom_tuple_3, neighbor_tuple_3)
        else:
            angle_tuple = (neighbor_tuple_3, atom_tuple_3, bond_order_str_2, 
                           atom_tuple_2, neighbor_tuple_2,
                           bond_order_str_1, atom_tuple_1, neighbor_tuple_1)
            
        return angle_tuple
    
    
    def get_angle_value(self,
                        conf: Conformer,
                        begin_atom_idx: int,
                        second_atom_idx: int,
                        end_atom_idx: int,
                        ):
        return GetAngleDeg(conf, begin_atom_idx, second_atom_idx, end_atom_idx)
    
    
    def get_torsions_atom_ids(self,
                             mol: Mol,
                             consider_hydrogens: bool = True):
        # dihedrals are defined by four consecutive atom in the graph, we take their idx
        paths = FindAllPathsOfLengthN(mol, 4, useBonds=False, useHs=consider_hydrogens)
        
        # if not consider_hydrogens:
        #     returned_tuples = []
        #     for path in paths:
        #         begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx = path
        #         atom_symbols = [mol.GetAtomWithIdx(begin_atom_idx).GetSymbol(),
        #                         mol.GetAtomWithIdx(second_atom_idx).GetSymbol(),
        #                         mol.GetAtomWithIdx(third_atom_idx).GetSymbol(),
        #                         mol.GetAtomWithIdx(end_atom_idx).GetSymbol()]
        #         if not 'H' in atom_symbols:
        #             returned_tuples.append(path)
        # else:
        returned_tuples = [tuple(path) for path in paths]
        
        return returned_tuples
    
    
    def get_torsion_tuple(self,
                             mol: Mol,
                            begin_atom_idx: int, 
                            second_atom_idx: int, 
                            third_atom_idx: int,
                            end_atom_idx: int):
        
        begin_atom = mol.GetAtomWithIdx(begin_atom_idx)
        second_atom = mol.GetAtomWithIdx(second_atom_idx)
        third_atom = mol.GetAtomWithIdx(third_atom_idx)
        end_atom = mol.GetAtomWithIdx(end_atom_idx)
        
        atom_symbol_1 = begin_atom.GetSymbol()
        bond1 = mol.GetBondBetweenAtoms(begin_atom_idx, second_atom_idx)
        bond_order_1 = bond1.GetBondTypeAsDouble()
        atom_symbol_2 = second_atom.GetSymbol()
        bond2 = mol.GetBondBetweenAtoms(second_atom_idx, third_atom_idx)
        bond_order_2 = bond2.GetBondTypeAsDouble()
        atom_symbol_3 = third_atom.GetSymbol()
        bond3 = mol.GetBondBetweenAtoms(third_atom_idx, end_atom_idx)
        bond_order_3 = bond3.GetBondTypeAsDouble()
        atom_symbol_4 = end_atom.GetSymbol()
        
        bond_order_str_1 = self.get_bond_order_str(bond_order_1)
        bond_order_str_2 = self.get_bond_order_str(bond_order_2)
        bond_order_str_3 = self.get_bond_order_str(bond_order_3)
        
        atom_tuple_1 = self.get_atom_tuple(begin_atom)
        atom_tuple_2 = self.get_atom_tuple(second_atom)
        atom_tuple_3 = self.get_atom_tuple(third_atom)
        atom_tuple_4 = self.get_atom_tuple(end_atom)
        
        neighbor_tuple_1 = self.get_neighbor_tuple(begin_atom, 
                                                   excluded_atoms=[second_atom, third_atom, end_atom])
        neighbor_tuple_2 = self.get_neighbor_tuple(second_atom,
                                                   excluded_atoms=[begin_atom, third_atom, end_atom])
        neighbor_tuple_3 = self.get_neighbor_tuple(third_atom,
                                                   excluded_atoms=[begin_atom, second_atom, end_atom])
        neighbor_tuple_4 = self.get_neighbor_tuple(end_atom,
                                                   excluded_atoms=[begin_atom, second_atom, third_atom])
        
        ascending_order = None
        if atom_symbol_2 < atom_symbol_3:
            ascending_order = True
        elif atom_symbol_2 == atom_symbol_3:
            if bond_order_1 < bond_order_3:
                ascending_order = True
            elif bond_order_1 == bond_order_3:
                if atom_symbol_1 < atom_symbol_4:
                    ascending_order = True
                else:
                    ascending_order = False
            else:
                ascending_order = False
        else:
            ascending_order = False
            
        if ascending_order:
            torsion_tuple = (atom_tuple_1, bond_order_str_1, neighbor_tuple_2, 
                             atom_tuple_2, bond_order_str_2, atom_tuple_3, 
                             neighbor_tuple_3, bond_order_str_3, atom_tuple_4)
        else:
            torsion_tuple = (atom_tuple_4, bond_order_str_3, neighbor_tuple_3,
                              atom_tuple_3, bond_order_str_2, atom_tuple_2, 
                              neighbor_tuple_2, bond_order_str_1, atom_tuple_1)
            
        # import pdb;pdb.set_trace()
            
        return torsion_tuple
    
    
    def get_torsion_value(self,
                        conf: Conformer,
                        begin_atom_idx: int,
                        second_atom_idx: int,
                        third_atom_idx: int,
                        end_atom_idx: int,
                        ):
        return GetDihedralDeg(conf, begin_atom_idx, second_atom_idx, 
                              third_atom_idx, end_atom_idx)