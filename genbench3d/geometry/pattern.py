import re

from rdkit.Chem import GetPeriodicTable
from typing import (NamedTuple, 
                    Union)
from functools import cmp_to_key
from rdkit.Chem import GetPeriodicTable

AtomicSymbol = str
AtomicNum = int
FormalCharge = int
BondType = float

periodic_table = GetPeriodicTable()

class CentralAtomTuple(NamedTuple):
    
    atomic_num: AtomicNum
    formal_charge: FormalCharge
        
    def __lt__(self, other: 'CentralAtomTuple') -> bool:
        if self.atomic_num < other.atomic_num:
            return True
        elif (self.atomic_num == other.atomic_num) and (self.formal_charge < other.formal_charge):
            return True
        else:
            return False
    
    def __gt__(self, other: 'CentralAtomTuple') -> bool:
        if self.atomic_num > other.atomic_num:
            return True
        elif (self.atomic_num == other.atomic_num) and (self.formal_charge > other.formal_charge):
            return True
        else:
            return False
    
    def __le__(self, other: 'CentralAtomTuple') -> bool:
        return not self > other
    
    def __ge__(self, other: 'CentralAtomTuple') -> bool:
        return not self < other
        
    def __eq__(self, other: 'CentralAtomTuple') -> bool:
        return (self.atomic_num == other.atomic_num) and (self.formal_charge == other.formal_charge)
    
    def __ne__(self, other: 'CentralAtomTuple') -> bool:
        return not self == other
    
    
class NeighborAtomTuple(NamedTuple):
    
    atomic_num: AtomicNum
    bond_type: BondType
        
    def __lt__(self, other: 'NeighborAtomTuple') -> bool:
        if self.atomic_num < other.atomic_num:
            return True
        elif (self.atomic_num == other.atomic_num) and (self.bond_type < other.bond_type):
            return True
        else:
            return False
    
    def __gt__(self, other: 'NeighborAtomTuple') -> bool:
        if self.atomic_num > other.atomic_num:
            return True
        elif (self.atomic_num == other.atomic_num) and (self.bond_type > other.bond_type):
            return True
        else:
            return False
    
    def __le__(self, other: 'NeighborAtomTuple') -> bool:
        return not self > other
    
    def __ge__(self, other: 'NeighborAtomTuple') -> bool:
        return not self < other
        
    def __eq__(self, other: 'NeighborAtomTuple') -> bool:
        return (self.atomic_num == other.atomic_num) and (self.bond_type == other.bond_type)
    
    def __ne__(self, other: 'NeighborAtomTuple') -> bool:
        return not self == other
    
class NeighborhoodTuple(NamedTuple):
    
    neighbor_tuples: tuple[NeighborAtomTuple, ...]
        
    def __lt__(self, other: 'NeighborhoodTuple') -> bool:
        if len(self.neighbor_tuples) < len(other.neighbor_tuples):
            return True
        elif len(self.neighbor_tuples) == len(other.neighbor_tuples):
            tuple_zip = zip(self.neighbor_tuples, other.neighbor_tuples)
            for neighbor_atom_tuple1, neighbor_atom_tuple2 in tuple_zip:
                if neighbor_atom_tuple1 < neighbor_atom_tuple2:
                    return True
            return False
        else:
            return False
    
    def __gt__(self, other: 'NeighborhoodTuple') -> bool:
        if len(self.neighbor_tuples) > len(other.neighbor_tuples):
            return True
        elif len(self.neighbor_tuples) == len(other.neighbor_tuples):
            tuple_zip = zip(self.neighbor_tuples, other.neighbor_tuples)
            for neighbor_atom_tuple1, neighbor_atom_tuple2 in tuple_zip:
                if neighbor_atom_tuple1 > neighbor_atom_tuple2:
                    return True
            return False
        else:
            return False
    
    def __le__(self, other: 'NeighborhoodTuple') -> bool:
        return not self > other
    
    def __ge__(self, other: 'NeighborhoodTuple') -> bool:
        return not self < other
        
    def __eq__(self, other: 'NeighborhoodTuple') -> bool:
        if len(self.neighbor_tuples) != len(other.neighbor_tuples):
            return False
        else:
            tuple_zip = zip(self.neighbor_tuples, other.neighbor_tuples)
            for neighbor_atom_tuple1, neighbor_atom_tuple2 in tuple_zip:
                if neighbor_atom_tuple1 != neighbor_atom_tuple2:
                    return False
            return True
                
    def __ne__(self, other: 'NeighborhoodTuple') -> bool:
        return not self == other
    
    @classmethod
    def from_string(cls,
                    string: str) -> 'NeighborhoodTuple':
        assert string.startswith('(') and string.endswith(')')
        string = string[1:-1]
        if string == '':
            return cls(())
        else:
            neighbor_tuples = []
            substrings = string.split(',')
            for string in substrings:
                bond_type = string[0]
                symbol = string[1:]
                bond_type = get_bond_type_float(bond_type)
                atomic_num = periodic_table.GetAtomicNumber(symbol)
                neighbor_tuple = NeighborAtomTuple(atomic_num, bond_type)
                neighbor_tuples.append(neighbor_tuple)
            neighbor_tuples = tuple(neighbor_tuples)
            return NeighborhoodTuple(neighbor_tuples)
    
    
def sort_neighbor_tuples(neighbor_tuples: list[NeighborAtomTuple],
                         reverse=True,
                         ):
    
    def cmp(nt1, nt2):
        if nt1 < nt2:
            return -1
        elif nt1 == nt2:
            return 0
        else:
            return 1
    
    return sorted(neighbor_tuples, 
                  key=cmp_to_key(cmp),
                  reverse=reverse)

def get_charge_string(charge: int) -> str:
    if charge > 0:
        charge = '+' + str(charge)
        charge = f'[{charge}]'
    elif charge < 0:
        charge = str(charge)
        charge = f'[{charge}]'
    else:
        charge = ''
    return charge
    
def get_charge_int(charge: str) -> int:
    if (charge is None) or (charge == ''):
        return 0
    else:
        assert charge.startswith('[') and charge.endswith(']')
        charge = charge[1:-1]
        sign = charge[0]
        value = charge[1:]
        if sign == '+':
            return int(value)
        elif sign == '-':
            return -int(value)
        else:
            raise Exception('Wrong sign')
    
def get_bond_type_string(bond_type: float) -> str:
    if bond_type == 1:
        return '-'
    elif bond_type == 2:
        return '='
    elif bond_type == 3:
        return '#'
    else :
        return '~'
    
def get_bond_type_float(bond_type: str) -> float:
    if bond_type == '-':
        return 1.0
    elif bond_type == '=':
        return 2.0
    elif bond_type == '#':
        return 3.0
    else:
        return 1.5
    
def get_neighbor_string(neighbor: NeighborAtomTuple) -> str:
    symbol = periodic_table.GetElementSymbol(neighbor.atomic_num)
    bond_type_str = get_bond_type_string(neighbor.bond_type)
    return bond_type_str + symbol


def get_central_atom_string(atom_tuple: CentralAtomTuple, 
                            neighborhood_tuple: NeighborAtomTuple):
    symbol = periodic_table.GetElementSymbol(atom_tuple.atomic_num)
    charge = get_charge_string(atom_tuple.formal_charge)
    neighbor_strings = [get_neighbor_string(neighbor)
                            for neighbor in neighborhood_tuple.neighbor_tuples]
    neighbors_string = f"({','.join(neighbor_strings)})"
    return symbol + charge + neighbors_string

atom_symbol_regex = r"(\w+)"
charge_regex = r"(\[.+\])?"
neighborhood_regex = r"(\(.*\))"
bond_type_regex = r"(-|=|\#|~)"

bond_regex = bond_type_regex.join([atom_symbol_regex 
                                    + charge_regex 
                                    + neighborhood_regex] * 2)
angle_regex = bond_type_regex.join([atom_symbol_regex 
                                    + charge_regex 
                                    + neighborhood_regex] * 3)
torsion_regex = bond_type_regex.join([atom_symbol_regex 
                                      + charge_regex 
                                      + neighborhood_regex] * 4)

class BondPattern(NamedTuple):
    
    central_atom_tuple_1: CentralAtomTuple
    neighborhood_tuple_1: NeighborhoodTuple
    bond_type_12: BondType
    central_atom_tuple_2: CentralAtomTuple
    neighborhood_tuple_2: NeighborhoodTuple
    
    def generalize(self) -> 'BondPattern':
        """ We do generalization by setting the neighborhood tuple to empty"""
        
        neighborhood_tuple_1 = NeighborhoodTuple(())
        neighborhood_tuple_2 = NeighborhoodTuple(())
            
        generalized_pattern = BondPattern(self.central_atom_tuple_1,
                                            neighborhood_tuple_1,
                                            self.bond_type_12,
                                            self.central_atom_tuple_2,
                                            neighborhood_tuple_2)
            
        return generalized_pattern
    
    def to_string(self) -> str:
        central_atom_string_1 = get_central_atom_string(self.central_atom_tuple_1,
                                                        self.neighborhood_tuple_1)
        bond_type_12 = get_bond_type_string(self.bond_type_12)
        central_atom_string_2 = get_central_atom_string(self.central_atom_tuple_2,
                                                        self.neighborhood_tuple_2)
        return central_atom_string_1 + bond_type_12 + central_atom_string_2
    
    @classmethod
    def from_string(cls,
                    string: str):
        
        pat = re.compile(bond_regex)
        results = pat.search(string)
        group_iter = iter(results.groups())
        atom_symbol_1 = next(group_iter)
        charge_1 = next(group_iter)
        neighbors_1 = next(group_iter)
        bond_type_12 = next(group_iter)
        atom_symbol_2 = next(group_iter)
        charge_2 = next(group_iter)
        neighbors_2 = next(group_iter)
        
        atomic_num_1 = periodic_table.GetAtomicNumber(atom_symbol_1)
        formal_charge_1 = get_charge_int(charge_1)
        central_atom_tuple_1 = CentralAtomTuple(atomic_num=atomic_num_1,
                                                formal_charge=formal_charge_1)
        neighborhood_tuple_1 = NeighborhoodTuple.from_string(neighbors_1)
        
        bond_type_12 = get_bond_type_float(bond_type_12)
        
        atomic_num_2 = periodic_table.GetAtomicNumber(atom_symbol_2)
        formal_charge_2 = get_charge_int(charge_2)
        central_atom_tuple_2 = CentralAtomTuple(atomic_num=atomic_num_2,
                                                formal_charge=formal_charge_2)
        neighborhood_tuple_2 = NeighborhoodTuple.from_string(neighbors_2)
        
        return cls(central_atom_tuple_1, 
                   neighborhood_tuple_1,
                   bond_type_12,
                   central_atom_tuple_2,
                   neighborhood_tuple_2)
        
    
class AnglePattern(NamedTuple):
    
    central_atom_tuple_1: CentralAtomTuple
    neighborhood_tuple_1: NeighborhoodTuple
    bond_type_12: BondType
    central_atom_tuple_2: CentralAtomTuple
    neighborhood_tuple_2: NeighborhoodTuple
    bond_type_23: BondType
    central_atom_tuple_3: CentralAtomTuple
    neighborhood_tuple_3: NeighborhoodTuple
    
    def generalize(self,
                   inner_neighbors: bool = False,
                   outer_neighbors: bool = True) -> 'AnglePattern':
        """ We do generalization by setting the neighborhood tuple to empty"""
        
        if (not outer_neighbors) and inner_neighbors:
            print('It is not advised to generalize the inner neighbors only')
        
        if outer_neighbors:
            neighborhood_tuple_1 = NeighborhoodTuple(())
            neighborhood_tuple_3 = NeighborhoodTuple(())
        else:
            neighborhood_tuple_1 = self.neighborhood_tuple_1
            neighborhood_tuple_3 = self.neighborhood_tuple_3
            
        if inner_neighbors:
            neighborhood_tuple_2 = NeighborhoodTuple(())
        else:
            neighborhood_tuple_2 = self.neighborhood_tuple_2
            
        generalized_pattern = AnglePattern(self.central_atom_tuple_1,
                                            neighborhood_tuple_1,
                                            self.bond_type_12,
                                            self.central_atom_tuple_2,
                                            neighborhood_tuple_2,
                                            self.bond_type_23,
                                            self.central_atom_tuple_3,
                                            neighborhood_tuple_3,)
            
        return generalized_pattern
    
    def to_string(self) -> str:
        central_atom_string_1 = get_central_atom_string(self.central_atom_tuple_1,
                                                        self.neighborhood_tuple_1)
        bond_type_12 = get_bond_type_string(self.bond_type_12)
        central_atom_string_2 = get_central_atom_string(self.central_atom_tuple_2,
                                                        self.neighborhood_tuple_2)
        bond_type_23 = get_bond_type_string(self.bond_type_23)
        central_atom_string_3 = get_central_atom_string(self.central_atom_tuple_3,
                                                        self.neighborhood_tuple_3)
        return central_atom_string_1 + bond_type_12 + central_atom_string_2 + bond_type_23 + central_atom_string_3
    
    @classmethod
    def from_string(cls,
                    string: str):
        
        pat = re.compile(angle_regex)
        results = pat.search(string)
        group_iter = iter(results.groups())
        atom_symbol_1 = next(group_iter)
        charge_1 = next(group_iter)
        neighbors_1 = next(group_iter)
        bond_type_12 = next(group_iter)
        atom_symbol_2 = next(group_iter)
        charge_2 = next(group_iter)
        neighbors_2 = next(group_iter)
        bond_type_23 = next(group_iter)
        atom_symbol_3 = next(group_iter)
        charge_3 = next(group_iter)
        neighbors_3 = next(group_iter)
        
        atomic_num_1 = periodic_table.GetAtomicNumber(atom_symbol_1)
        formal_charge_1 = get_charge_int(charge_1)
        central_atom_tuple_1 = CentralAtomTuple(atomic_num=atomic_num_1,
                                                formal_charge=formal_charge_1)
        neighborhood_tuple_1 = NeighborhoodTuple.from_string(neighbors_1)
        
        bond_type_12 = get_bond_type_float(bond_type_12)
        
        atomic_num_2 = periodic_table.GetAtomicNumber(atom_symbol_2)
        formal_charge_2 = get_charge_int(charge_2)
        central_atom_tuple_2 = CentralAtomTuple(atomic_num=atomic_num_2,
                                                formal_charge=formal_charge_2)
        neighborhood_tuple_2 = NeighborhoodTuple.from_string(neighbors_2)
        
        bond_type_23 = get_bond_type_float(bond_type_23)
        
        atomic_num_3 = periodic_table.GetAtomicNumber(atom_symbol_3)
        formal_charge_3 = get_charge_int(charge_3)
        central_atom_tuple_3 = CentralAtomTuple(atomic_num=atomic_num_3,
                                                formal_charge=formal_charge_3)
        neighborhood_tuple_3 = NeighborhoodTuple.from_string(neighbors_3)
        
        return cls(central_atom_tuple_1, 
                   neighborhood_tuple_1,
                   bond_type_12,
                   central_atom_tuple_2,
                   neighborhood_tuple_2,
                   bond_type_23,
                   central_atom_tuple_3,
                   neighborhood_tuple_3)
    
class TorsionPattern(NamedTuple):
    
    central_atom_tuple_1: CentralAtomTuple
    neighborhood_tuple_1: NeighborhoodTuple
    bond_type_12: BondType
    central_atom_tuple_2: CentralAtomTuple
    neighborhood_tuple_2: NeighborhoodTuple
    bond_type_23: BondType
    central_atom_tuple_3: CentralAtomTuple
    neighborhood_tuple_3: NeighborhoodTuple
    bond_type_34: BondType
    central_atom_tuple_4: CentralAtomTuple
    neighborhood_tuple_4: NeighborhoodTuple
    
    def generalize(self,
                   inner_neighbors: bool = False,
                   outer_neighbors: bool = True) -> 'TorsionPattern':
        """ We do generalization by setting the neighborhood tuple to empty"""
        
        if (not outer_neighbors) and inner_neighbors:
            print('It is not advised to generalize the inner neighbors only')
        
        if outer_neighbors:
            neighborhood_tuple_1 = NeighborhoodTuple(())
            neighborhood_tuple_4 = NeighborhoodTuple(())
        else:
            neighborhood_tuple_1 = self.neighborhood_tuple_1
            neighborhood_tuple_4 = self.neighborhood_tuple_4
            
        if inner_neighbors:
            neighborhood_tuple_2 = NeighborhoodTuple(())
            neighborhood_tuple_3 = NeighborhoodTuple(())
        else:
            neighborhood_tuple_2 = self.neighborhood_tuple_2
            neighborhood_tuple_3 = self.neighborhood_tuple_3
            
        generalized_torsion_pattern = TorsionPattern(self.central_atom_tuple_1,
                                                     neighborhood_tuple_1,
                                                     self.bond_type_12,
                                                     self.central_atom_tuple_2,
                                                     neighborhood_tuple_2,
                                                     self.bond_type_23,
                                                     self.central_atom_tuple_3,
                                                     neighborhood_tuple_3,
                                                     self.bond_type_34,
                                                     self.central_atom_tuple_4,
                                                     neighborhood_tuple_4)
            
        return generalized_torsion_pattern
    
    def to_string(self) -> str:
        central_atom_string_1 = get_central_atom_string(self.central_atom_tuple_1,
                                                        self.neighborhood_tuple_1)
        bond_type_12 = get_bond_type_string(self.bond_type_12)
        central_atom_string_2 = get_central_atom_string(self.central_atom_tuple_2,
                                                        self.neighborhood_tuple_2)
        bond_type_23 = get_bond_type_string(self.bond_type_23)
        central_atom_string_3 = get_central_atom_string(self.central_atom_tuple_3,
                                                        self.neighborhood_tuple_3)
        bond_type_34 = get_bond_type_string(self.bond_type_34)
        central_atom_string_4 = get_central_atom_string(self.central_atom_tuple_4,
                                                        self.neighborhood_tuple_4)
        return central_atom_string_1 + bond_type_12 + central_atom_string_2 + bond_type_23 + central_atom_string_3 + bond_type_34 + central_atom_string_4
    
    @classmethod
    def from_string(cls,
                    string: str):
        
        pat = re.compile(torsion_regex)
        results = pat.search(string)
        group_iter = iter(results.groups())
        atom_symbol_1 = next(group_iter)
        charge_1 = next(group_iter)
        neighbors_1 = next(group_iter)
        bond_type_12 = next(group_iter)
        atom_symbol_2 = next(group_iter)
        charge_2 = next(group_iter)
        neighbors_2 = next(group_iter)
        bond_type_23 = next(group_iter)
        atom_symbol_3 = next(group_iter)
        charge_3 = next(group_iter)
        neighbors_3 = next(group_iter)
        bond_type_34 = next(group_iter)
        atom_symbol_4 = next(group_iter)
        charge_4 = next(group_iter)
        neighbors_4 = next(group_iter)
        
        atomic_num_1 = periodic_table.GetAtomicNumber(atom_symbol_1)
        formal_charge_1 = get_charge_int(charge_1)
        central_atom_tuple_1 = CentralAtomTuple(atomic_num=atomic_num_1,
                                                formal_charge=formal_charge_1)
        neighborhood_tuple_1 = NeighborhoodTuple.from_string(neighbors_1)
        
        bond_type_12 = get_bond_type_float(bond_type_12)
        
        atomic_num_2 = periodic_table.GetAtomicNumber(atom_symbol_2)
        formal_charge_2 = get_charge_int(charge_2)
        central_atom_tuple_2 = CentralAtomTuple(atomic_num=atomic_num_2,
                                                formal_charge=formal_charge_2)
        neighborhood_tuple_2 = NeighborhoodTuple.from_string(neighbors_2)
        
        bond_type_23 = get_bond_type_float(bond_type_23)
        
        atomic_num_3 = periodic_table.GetAtomicNumber(atom_symbol_3)
        formal_charge_3 = get_charge_int(charge_3)
        central_atom_tuple_3 = CentralAtomTuple(atomic_num=atomic_num_3,
                                                formal_charge=formal_charge_3)
        neighborhood_tuple_3 = NeighborhoodTuple.from_string(neighbors_3)
        
        bond_type_34 = get_bond_type_float(bond_type_34)
        
        atomic_num_4 = periodic_table.GetAtomicNumber(atom_symbol_4)
        formal_charge_4 = get_charge_int(charge_4)
        central_atom_tuple_4 = CentralAtomTuple(atomic_num=atomic_num_4,
                                                formal_charge=formal_charge_4)
        neighborhood_tuple_4 = NeighborhoodTuple.from_string(neighbors_4)
        
        return cls(central_atom_tuple_1, 
                   neighborhood_tuple_1,
                   bond_type_12,
                   central_atom_tuple_2,
                   neighborhood_tuple_2,
                   bond_type_23,
                   central_atom_tuple_3,
                   neighborhood_tuple_3,
                   bond_type_34,
                   central_atom_tuple_4,
                   neighborhood_tuple_4)
        
GeometryPattern = Union[BondPattern, AnglePattern, TorsionPattern]