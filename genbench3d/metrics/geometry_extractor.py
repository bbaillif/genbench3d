from rdkit.Chem import Bond, Mol, Conformer
from rdkit.Chem.rdMolTransforms import GetBondLength, GetAngleDeg, GetDihedralDeg
from rdkit.Chem.rdmolops import FindAllPathsOfLengthN

class GeometryExtractor():
    
    def __init__(self) -> None:
        pass
    
    
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
    
    
    def get_bonds(self,
                  mol: Mol):
        return mol.GetBonds()
    
    
    def get_bond_length(self,
                        conf: Conformer,
                        bond: Bond):
        return GetBondLength(conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    
    def get_bond_string(self, 
                        bond: Bond):
        bond_order = bond.GetBondTypeAsDouble()
        bond_order_str = self.get_bond_order_str(bond_order)
        atom_symbol_1 = bond.GetBeginAtom().GetSymbol()
        atom_symbol_2 = bond.GetEndAtom().GetSymbol()
        if atom_symbol_1 < atom_symbol_2:
            bond_string = atom_symbol_1 + bond_order_str + atom_symbol_2
        else:
            bond_string = atom_symbol_2 + bond_order_str + atom_symbol_1
        return bond_string
    
    
    def get_angles_atom_ids(self,
                           mol: Mol):
        # angles are defined by three consecutive atom in the graph, we take their idx
        return [tuple(path) for path in FindAllPathsOfLengthN(mol, 3, useBonds=False, useHs=True)]
    
    
    def get_angle_string(self, 
                         mol: Mol,
                         begin_atom_idx: int, 
                         second_atom_idx: int, 
                         end_atom_idx: int) -> str:
        begin_atom = mol.GetAtomWithIdx(begin_atom_idx)
        second_atom = mol.GetAtomWithIdx(second_atom_idx)
        end_atom = mol.GetAtomWithIdx(end_atom_idx)
        
        atom_symbol_1 = begin_atom.GetSymbol()
        bond_order_1 = mol.GetBondBetweenAtoms(begin_atom_idx, second_atom_idx).GetBondTypeAsDouble()
        atom_symbol_2 = second_atom.GetSymbol()
        bond_order_2 = mol.GetBondBetweenAtoms(second_atom_idx, end_atom_idx).GetBondTypeAsDouble()
        atom_symbol_3 = end_atom.GetSymbol()
        
        bond_order_str_1 = self.get_bond_order_str(bond_order_1)
        bond_order_str_2 = self.get_bond_order_str(bond_order_2)
        
        if atom_symbol_1 < atom_symbol_3:
            angle_string = atom_symbol_1 + bond_order_str_1 + atom_symbol_2 + bond_order_str_2 + atom_symbol_3
        elif atom_symbol_1 == atom_symbol_3:
            if bond_order_1 < bond_order_2:
                angle_string = atom_symbol_1 + bond_order_str_1 + atom_symbol_2 + bond_order_str_2 + atom_symbol_3
            else:
                angle_string = atom_symbol_3 + bond_order_str_2 + atom_symbol_2 + bond_order_str_1 + atom_symbol_1
        else:
            angle_string = atom_symbol_3 + bond_order_str_2 + atom_symbol_2 + bond_order_str_1 + atom_symbol_1
            
        return angle_string
    
    
    def get_angle_value(self,
                        conf: Conformer,
                        begin_atom_idx: int,
                        second_atom_idx: int,
                        end_atom_idx: int,
                        ):
        return GetAngleDeg(conf, begin_atom_idx, second_atom_idx, end_atom_idx)
    
    
    def get_torsions_atom_ids(self,
                             mol: Mol):
        # dihedrals are defined by four consecutive atom in the graph, we take their idx
        return [tuple(path) for path in FindAllPathsOfLengthN(mol, 4, useBonds=False, useHs=True)]
    
    
    def get_torsion_string(self,
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
        bond_order_1 = mol.GetBondBetweenAtoms(begin_atom_idx, second_atom_idx).GetBondTypeAsDouble()
        atom_symbol_2 = second_atom.GetSymbol()
        bond_order_2 = mol.GetBondBetweenAtoms(second_atom_idx, third_atom_idx).GetBondTypeAsDouble()
        atom_symbol_3 = third_atom.GetSymbol()
        bond_order_3 = mol.GetBondBetweenAtoms(third_atom_idx, end_atom_idx).GetBondTypeAsDouble()
        atom_symbol_4 = end_atom.GetSymbol()
        
        bond_order_str_1 = self.get_bond_order_str(bond_order_1)
        bond_order_str_2 = self.get_bond_order_str(bond_order_2)
        bond_order_str_3 = self.get_bond_order_str(bond_order_3)
        
        if atom_symbol_2 < atom_symbol_3:
            torsion_string = atom_symbol_1 + bond_order_str_1 + atom_symbol_2 + bond_order_str_2 + atom_symbol_3 + bond_order_str_3 + atom_symbol_4
        elif atom_symbol_2 == atom_symbol_3:
            if bond_order_1 < bond_order_3:
                torsion_string = atom_symbol_1 + bond_order_str_1 + atom_symbol_2 + bond_order_str_2 + atom_symbol_3 + bond_order_str_3 + atom_symbol_4
            elif bond_order_1 == bond_order_3:
                if atom_symbol_1 < atom_symbol_4:
                    torsion_string = atom_symbol_1 + bond_order_str_1 + atom_symbol_2 + bond_order_str_2 + atom_symbol_3 + bond_order_str_3 + atom_symbol_4
                else:
                    torsion_string = atom_symbol_4 + bond_order_str_3 + atom_symbol_3 + bond_order_str_2 + atom_symbol_2 + bond_order_str_1 + atom_symbol_1
            else: 
                torsion_string = atom_symbol_4 + bond_order_str_3 + atom_symbol_3 + bond_order_str_2 + atom_symbol_2 + bond_order_str_1 + atom_symbol_1
        else:
            torsion_string = atom_symbol_4 + bond_order_str_3 + atom_symbol_3 + bond_order_str_2 + atom_symbol_2 + bond_order_str_1 + atom_symbol_1
            
        return torsion_string
    
    
    def get_torsion_value(self,
                        conf: Conformer,
                        begin_atom_idx: int,
                        second_atom_idx: int,
                        third_atom_idx: int,
                        end_atom_idx: int,
                        ):
        return GetDihedralDeg(conf, begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx)