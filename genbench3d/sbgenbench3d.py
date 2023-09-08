import os
import numpy as np
import MDAnalysis as mda
import prolif as plf
import logging

from typing import List
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Mol
from docking.vina_scorer import VinaScorer
from espsim import GetEspSim
from .genbench3d import GenBench3D
# from ccdc.docking import Docker
# from ccdc.io import MoleculeReader, Entry

class SBGenBench3D(GenBench3D):
    
    def __init__(self, 
                 original_structure_path: str,
                 clean_mda_prot: mda.Universe,
                 native_ligand: Mol,
                 root: str = 'genbench3d_results/', 
                 training_mols: List = None, 
                 show_plots: bool = True, 
                 tfd_threshold: float = 0.2) -> None:
        assert os.path.exists(original_structure_path)
        super().__init__(root, 
                         training_mols, 
                         show_plots, 
                         tfd_threshold)
        
        self.original_structure_path = original_structure_path
        self.clean_mda_prot = clean_mda_prot
        self.prot = plf.Molecule.from_mda(clean_mda_prot)
        self.native_ligand = Chem.AddHs(native_ligand, addCoords=True)
        
        
    def get_metrics_for_mol_list(self, 
                                 mols: List, 
                                 n_total_mols: int = None):
        self.metrics = super().get_metrics_for_mol_list(mols, n_total_mols)
        
        self.absolute_vina_scores = self.get_vina_score(ligands=mols,
                                                   relative=False)
        self.metrics['Median absolute Vina score'] = np.median(self.absolute_vina_scores)
        
        self.relative_vina_scores = self.get_vina_score(ligands=mols)
        median_vina_relative = np.median(self.relative_vina_scores)
        self.metrics['Median Vina score relative to test ligand'] = median_vina_relative

        self.ifp_sims = self.get_ifp_sims(ligands=mols)
        median_ifp_sim = np.median(self.ifp_sims)
        self.metrics['Median IFP similarity to test ligand'] = median_ifp_sim
        
        self.espsims = self.get_espsims(ligands=mols)
        median_espsim = np.median(self.espsims)
        self.metrics['Median ESPSIM to test ligand'] = median_espsim
        
        return self.metrics

        
    def get_vina_score(self,
                        ligands: List[Mol],
                        relative: bool=True):
        self.vina_scores = []
        receptor_pdbqt_filepath = self.original_structure_path.replace('.pdb', 
                                                                       '.pdbqt')
        vina_scorer = VinaScorer(receptor_pdbqt_filepath)
        vina_scorer.write_box_from_ligand(ligand=self.native_ligand)
        vina_scorer.set_box()
        
        if relative:
            native_energies = vina_scorer.score_mol(self.native_ligand)
            native_score = native_energies[0]
        
        for ligand in ligands:
            try:
                energies = vina_scorer.score_mol(ligand)
                score = energies[0]
                if relative:
                    score = score - native_score

            except Exception as e:
                logging.warning(f'Vina scoring error: {e}')
                score = np.nan
                
            self.vina_scores.append(score)
        return list(self.vina_scores)
    
    
    # def gold_plp_score_ligands(self,
    #                         ligands: List[Mol],
    #                         native_ligand_filepath: str):
    #     self.gold_plp_scores = []
    #     docker = Docker()
    #     settings = docker.settings
    #     settings.output_directory = 'docking_results'
    #     settings.fitness_function = None
    #     settings.rescore_function = 'plp'
        
    #     receptor_filepath = self.original_structure_path.replace('.pdb', '_protein_only_clean.pdb')
    #     settings.add_protein_file(receptor_filepath)

    #     protein = settings.proteins[0]
    #     native_ligand = MoleculeReader(native_ligand_filepath)[0]
    #     settings.binding_site = settings.BindingSiteFromLigand(protein=protein, ligand=native_ligand, distance=10.)
        
    #     for i, gen_mol in enumerate(ligands):
    #         ligand_filename = f'docking_results/test_lig_{i}.sdf'
        
    #         with Chem.SDWriter(ligand_filename) as writer:
    #             writer.write(gen_mol)
    #         ligand = MoleculeReader(ligand_filename)[0]
    #         ligand_entry = Entry.from_molecule(ligand)

    #         ligand_preparation = Docker.LigandPreparation()
    #         ligand_preped = ligand_preparation.prepare(ligand_entry)
    #         ligand_mol2_filename = ligand_filename.replace('sdf', 'mol2')
    #         mol2_string = ligand_preped.to_string(format='mol2')
    #         with open(ligand_mol2_filename, 'w') as writer :
    #             writer.write(mol2_string)
    #         settings.add_ligand_file(ligand_mol2_filename, 1)

    #     docker.dock()
        
    #     scores = []
    #     for docked_ligand in docker.results.ligands:
    #         score = docked_ligand.attributes['Gold.PLP.Fitness']
    #         scores.append(score)
        
    #     self.gold_plp_scores.extend(scores)
    
    
    def get_ifp_sims(self,
                    ligands: List[Mol]):
        self.ifp_sims = []
        try:
            ligands = [Chem.AddHs(ligand, addCoords=True) for ligand in ligands]
            
            fp = plf.Fingerprint()
            lig_list = [plf.Molecule.from_rdkit(self.native_ligand)] \
                + [plf.Molecule.from_rdkit(ligand) for ligand in ligands]
            fp.run_from_iterable(lig_list, self.prot)
            df = fp.to_dataframe()
            
            bvs = plf.to_bitvectors(df)
            for i, bv in enumerate(bvs[1:]):
                sim = DataStructs.TanimotoSimilarity(bvs[0], bv)
                self.ifp_sims.append(sim)
        except Exception as e:
            logging.warning(f'IFP computation error: {e}')
            
        return list(self.ifp_sims)
    
    
    def get_espsims(self,
                    ligands: List[Mol]):
        self.esp_sims = []
        for ligand in ligands:
            try:
                ligand = Chem.AddHs(ligand, addCoords=True)
                esp_sim = GetEspSim(ligand, self.native_ligand, renormalize=True)
            except Exception as e:
                logging.warning(f'ESPSIM computation error: {e}')
                esp_sim = np.nan
            self.esp_sims.append(esp_sim)
        return list(self.esp_sims)