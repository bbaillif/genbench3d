import MDAnalysis as mda
import logging

from rdkit import Chem
from rdkit.Chem import (AllChem, 
                        Mol)

# We put native ligand as input because we work with existing complex
# This code would need to be adapted if working with new pocket that don't have
# known ligand

class ComplexMinimizer():
    
    def __init__(self,
                 clean_mda_prot: mda.Universe,
                 native_ligand: Mol,
                 distance_from_ligand: float = 5, # Angstrom
                 ) -> None:
        self.clean_mda_prot = clean_mda_prot
        self.native_ligand = native_ligand
        self.distance_from_ligand = distance_from_ligand
        
        self.pocket = self.extract_pocket()
    
    def extract_pocket(self,
                       ligand_resname: str = 'UNL'):
        ligand = mda.Universe(self.native_ligand)
        ligand.add_TopologyAttr('resname', [ligand_resname])
        
        complx = mda.Merge(self.clean_mda_prot.atoms, ligand.atoms)

        # import pdb;pdb.set_trace()

        selection = f'(protein) and (around {self.distance_from_ligand} resname {ligand_resname}) and (not type H)'   
        atom_group: mda.AtomGroup = complx.select_atoms(selection)
        # atom_group.write('test_pocket.pdb')
        if len(atom_group) > 20:
            segids = {}
            for residue in atom_group.residues:
                segid = residue.segid
                resid = residue.resid
                if segid in segids:
                    segids[segid].append(resid)
                else:
                    segids[segid] = [resid]
            selections = []
            for segid, resids in segids.items():
                resids_str = ' '.join([str(resid) for resid in set(resids)])
                selections.append(f'((resid {resids_str}) and (segid {segid}))')
            pocket_selection = ' or '.join(selections)
            protein_pocket: mda.AtomGroup = self.clean_mda_prot.select_atoms(pocket_selection)
            self.pocket_mol = protein_pocket.atoms.convert_to("RDKIT")
        else:
            logging.warning('Pocket quite small')
        
    
    def minimize_ligand(self,
                        ligand_mol: Mol,
                        minimized_ligand_filepath: str = None,
                        n_steps: int = 1000,
                        distance_constraint: float = 1.0, # A
                        ignore_pocket: bool = False,
                        ):
        
        ligand = Chem.AddHs(ligand_mol, addCoords=True)
        if not ignore_pocket:
            complx = Chem.CombineMols(self.pocket_mol, ligand)
        else:
            complx = ligand
            
        Chem.SanitizeMol(complx)
            
        try:
            mol_properties = AllChem.MMFFGetMoleculeProperties(complx, 
                                                                mmffVariant='MMFF94s')
            mmff = AllChem.MMFFGetMoleculeForceField(complx, 
                                                    mol_properties,
                                                    confId=0,
                                                    nonBondedThresh=10.0,
                                                    ignoreInterfragInteractions=False)
            mmff.Initialize()
            
            for idx in range(self.pocket_mol.GetNumAtoms(), complx.GetNumAtoms()):
                atom = complx.GetAtomWithIdx(idx)
                if atom.GetSymbol() != 'H':
                    mmff.MMFFAddPositionConstraint(idx, distance_constraint, 999.0)

            # get the initial energy
            E_init = mmff.CalcEnergy()

        except Exception as e:
            logging.warning(f'MMFF minimization exception: {e}')
            return None
            
        else:
            
            if not ignore_pocket :
                for i in range(self.pocket_mol.GetNumAtoms()):
                    mmff.AddFixedPoint(i)
                
            with Chem.SDWriter('test_complex_before.sdf') as w:
                w.write(complx)
                
            results = mmff.Minimize(maxIts=n_steps)
            not_converged = results

            with Chem.SDWriter('test_complex_after.sdf') as w:
                w.write(complx)

            E_final = mmff.CalcEnergy()
            if not_converged:
                print('Not converged')
                
            # print(E_final - E_init)
                
            # Chem.MolToPDBFile(complx, 'test_opti_complx.pdb')
            minimized_frags = Chem.GetMolFrags(complx, asMols=True)
            minimized_ligand = minimized_frags[-1]
            if minimized_ligand_filepath is not None:
                with Chem.SDWriter(minimized_ligand_filepath) as writer:
                    writer.write(minimized_ligand)
                    
            # import pdb;pdb.set_trace()
                    
            return minimized_ligand #, E_init - E_final
    
    
    # Inspired from LiGAN code by Ragoza et al.
    
    # def uff_minimize_ligand(self,
    #                        ligand_mol: Mol,
    #                        pocket_filepath: str = None,
    #                        minimized_ligand_filepath: str = None,
    #                        n_steps: int = 1000):
    #     ligand = Chem.AddHs(ligand_mol, addCoords=True)
        
    #     if pocket_filepath is not None:
    #         if not os.path.exists(pocket_filepath):
    #             pdb_filepath = pocket_filepath.replace('_pocket', '_protein_only')
    #             self.extract_pocket(ligand_mol, pdb_filepath, pocket_filepath)
    #         pocket = mda.Universe(pocket_filepath)
    #         # pocket_selection = pocket.select_atoms('protein')
    #         pocket = pocket.atoms.convert_to("RDKIT")
    #         complx = Chem.CombineMols(pocket, ligand)
    #     else:
    #         complx = ligand
            
    #     Chem.SanitizeMol(complx)
            
    #     try:
    #         # initialize force field
    #         uff = AllChem.UFFGetMoleculeForceField(
    #             complx, confId=0, ignoreInterfragInteractions=False
    #         )
    #         uff.Initialize()
            
    #         for idx in range(pocket.GetNumAtoms(), complx.GetNumAtoms()):
    #             atom = complx.GetAtomWithIdx(idx)
    #             if atom.GetSymbol() != 'H':
    #                 uff.UFFAddPositionConstraint(idx, 0.5, 9999.0)
                
    #         # for bond in complx.GetBonds():
    #         #     id1 = bond.GetBeginAtomIdx()
    #         #     id2 = bond.GetEndAtomIdx()
    #         #     uff.UFFAddDistanceConstraint(id1, id2, True, -0.5, 0.5, 99999.0)

    #         # get the initial energy
    #         E_init = uff.CalcEnergy()

    #     except Exception as e:
    #         if 'getNumImplicitHs' in str(e):
    #             print('No implicit valence')
    #         if 'bad params pointer' in str(e):
    #             print('Invalid atom type')
    #         print('UFF1 exception: ', e)
    #         import pdb;pdb.set_trace()
            
    #     if pocket_filepath is not None:
    #         for i in range(pocket.GetNumAtoms()):
    #             uff.AddFixedPoint(i)
            
    #     with Chem.SDWriter('test_complex_before.sdf') as w:
    #         w.write(complx)
            
    #     results = uff.Minimize(maxIts=n_steps)
    #     not_converged = results

    #     with Chem.SDWriter('test_complex_after.sdf') as w:
    #         w.write(complx)

    #     # import pdb;pdb.set_trace()

    #     # get the final energy
    #     E_final = uff.CalcEnergy()
    #     if not_converged:
    #         print('Not converged')
            
    #     # print(E_final - E_init)
            
    #     # Chem.MolToPDBFile(complx, 'test_opti_complx.pdb')
    #     minimized_frags = Chem.GetMolFrags(complx, asMols=True)
    #     minimized_ligand = minimized_frags[-1]
    #     if minimized_ligand_filepath is not None:
    #         with Chem.SDWriter(minimized_ligand_filepath) as writer:
    #             writer.write(minimized_ligand)
                
    #     return minimized_ligand, E_init - E_final