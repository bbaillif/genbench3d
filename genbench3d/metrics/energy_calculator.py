import torch
import os
import logging
import numpy as np

from abc import abstractmethod
from .conf_search import padding_coords, padding_species, ensemble_opt, EnForce_ANI
from .spe import Calculator, ev2hatree
from rdkit import Chem
from rdkit.Chem import Mol, AllChem
from typing import Tuple, Union
from ase import Atoms
from ase.units import mol as mol_unit, kcal, eV
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
# from xtb.ase.calculator import XTB
# from ase.io import read
# from ase.optimize import BFGS

class EnergyCalculator():
    
    @abstractmethod
    def compute_strain_energy(self,
                              mol: Mol):
        raise NotImplementedError()


class FFEnergyCalculator(EnergyCalculator):
    
    def __init__(self,
                 force_field_name: str = 'MMFF94s'):
        assert force_field_name in ['MMFF94s', 'MMFF94', 'UFF'], \
            'Please select a valid force field name among [MMFF94s, MMFF94, UFF]'
        self.force_field_name = force_field_name
        
    def compute_strain_energy(self,
                              mol: Mol,
                              conf_id: int = 0,
                              return_mol: bool = False,
                              n_steps: int = 1000) -> Union[Tuple[float, Mol],
                                                                 float]:
        new_mol = Mol(mol)
        new_mol = Chem.AddHs(new_mol, addCoords=True)
        if self.force_field_name == 'MMFF94s':
            mol_properties = AllChem.MMFFGetMoleculeProperties(new_mol, 
                                                                mmffVariant='MMFF94s')
            force_field = AllChem.MMFFGetMoleculeForceField(new_mol, 
                                                            mol_properties,
                                                            confId=conf_id)
        elif self.force_field_name == 'MMFF94':
            mol_properties = AllChem.MMFFGetMoleculeProperties(new_mol)
            force_field = AllChem.MMFFGetMoleculeForceField(new_mol, 
                                                            mol_properties,
                                                            confId=conf_id)
        elif self.force_field_name == 'UFF':
            force_field = AllChem.UFFGetMoleculeForceField(new_mol, confId=conf_id)
            
        try:
            start_energy = force_field.CalcEnergy()
            not_converged = force_field.Minimize(maxIts=n_steps)
            if not_converged:
                print('Not converged')
            final_energy = force_field.CalcEnergy()
            strain_energy = start_energy - final_energy
            assert strain_energy > 0
        except Exception as e:
            logging.warning('Force field error')
            strain_energy = np.nan
            
        if return_mol:
            return strain_energy, new_mol
        else:
            return strain_energy
    
class TorchANIEnergyCalculator(EnergyCalculator):
    
    def __init__(self,
                 aimnet_filename: str = 'aimnet2nqed_pc14iall_b97m_sae.jpt',
                 root: str = '/home/bb596/threeDGenMolBenchmark',
                 device: str = 'cuda',
                 batchsize_atoms: int = 1024,
                 ) -> None:
        self.root = root
        self.aimnet_filename = aimnet_filename
        self.device = device
        self.batchsize_atoms = batchsize_atoms
        self.aimnet_path = os.path.join(root, aimnet_filename)
        model = torch.jit.load(self.aimnet_path, 
                                  map_location=device) # Return a ScriptModule object
        for p in model.parameters():  
            p.requires_grad_(False)
        self.model = EnForce_ANI(
                    ani=model,
                        name="AIMNET",
                        batchsize_atoms=batchsize_atoms,)
    
    def compute_strain_energy(self, 
                              mol: Mol,
                              opt_steps: int = 5000,
                            opt_tol: float = 0.003,
                            patience: int = 1000,
                            ) -> Tuple[float, Mol]:
        try:
            new_mol = Mol(mol)
            new_mol = Chem.AddHs(new_mol, addCoords=True)
            
            device = torch.device(f"cuda")
            config = {"opt_steps": opt_steps, 
                    "opttol": opt_tol, 
                    "patience": patience, 
                    "batchsize_atoms": self.batchsize_atoms}
            
            mol_charge = Chem.GetFormalCharge(new_mol)
            mol_numbers = [atom.GetAtomicNum() for atom in new_mol.GetAtoms()]
            coord = []
            charges = []
            numbers = []
            for conf in new_mol.GetConformers():
                coord.append(conf.GetPositions().tolist())
                charges.append(mol_charge)
                numbers.append(mol_numbers)
            
            coord_padded = padding_coords(coord, 0)
            numbers_padded = padding_species(numbers, 0)

            start_energy = self.calc_energy(new_mol)

            with torch.jit.optimized_execution(False):
                optdict = ensemble_opt(self.model, 
                                    coord_padded, 
                                    numbers_padded, 
                                    charges,
                                    config,
                                    device)  #Magic step

            energies = optdict['energy']
            fmax = optdict['fmax']
            convergence_mask = list(map(lambda x: (x <= config['opttol']), fmax))
            
            coord = optdict['coord'][0]
            
            # not_converged = MMFFOptimizeMolecule(mol)
            
            conf = new_mol.GetConformer()
            for i, position in enumerate(coord):
                conf.SetAtomPosition(i, position)
            final_energy = self.calc_energy(new_mol)
            
            with Chem.SDWriter('test_opt_mol.sdf') as writer:
                writer.write(new_mol)
            
            # import pdb; pdb.set_trace()
            
            # for i, conf in enumerate(mol.GetConformers()):
            #     final_energy = energies[i]
            #     converged = convergence_mask[i]
            #     if not converged:
            #         print('Not converged')
                
            #     coord = optdict['coord'][i]
            #     for i, position in enumerate(coord):
            #         conf.SetAtomPosition(i, position)     
            strain_energy = start_energy - final_energy
            
            return strain_energy
        
        except KeyboardInterrupt:
            exit()
        
        except Exception as e:
            print(e)
            import pdb;pdb.set_trace()
                
            
    def calc_energy(self,
                    mol: Mol) :

        species2numbers = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'Si':14, 'P':15,
                        'S':16, 'Cl':17, 'As':33, 'Se':34, 'Br':35, 'I':53, 'B':5}
        numbers2species = dict([(val, key) for (key, val) in species2numbers.items()])
            
        coord = mol.GetConformer().GetPositions()
        charge = Chem.GetFormalCharge(mol)
        charge = torch.tensor(charge, dtype=torch.float, device=self.device)
        species = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atoms = Atoms(species, coord)
        calculator = Calculator(self.model, charge)      
        atoms.set_calculator(calculator)

        e_ev = atoms.get_potential_energy()
        e_kcal_mol = e_ev * 23.060541945329334
        return e_kcal_mol
        
        
# class XTBEnergyCalculator(EnergyCalculator):
    
#     def compute_strain_energy(self, 
#                               mol: Mol):
#         mol = Chem.AddHs(mol, addCoords=True)
#         filepath = 'test_mol_xtb.sdf'
#         with Chem.SDWriter(filepath) as writer:
#             writer.write(mol)
#         atoms = read(filepath)
#         atoms.calc = XTB(method='GFN2-xTB')
#         start_energy = atoms.get_potential_energy()
#         dyn = BFGS(atoms, trajectory='test_xtb.traj')
#         dyn.run()
#         final_energy = atoms.get_potential_energy()
#         strain_energy = start_energy - final_energy
#         atoms.write('test_opti_xtb.xyz')
#         return strain_energy