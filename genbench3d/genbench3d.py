import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from typing import Any, Dict, List
from rdkit import Chem
from rdkit.Chem import Mol


from .conf_ensemble import GeneratedCEL, ConfEnsembleLibrary
from .data.generated_sample_reader import ASEDBReader
from tqdm import tqdm

from .metrics import (Metric,
                      TrainingMetric,
                      Validity2D,
                      Uniqueness2D,
                      Diversity2D,
                      Novelty2D,
                      MaxTrainSim,
                      MolProperty,
                      RingProportion,
                      Validity3D,
                      Uniqueness3D,
                      Diversity3D,
                      Novelty3D,
                      StrainEnergy,
                      )

# from .benchmark_results import BenchmarkResults

from genbench3d.params import DEFAULT_TFD_THRESHOLD

class GenBench3D():
    
    def __init__(self,
                 root: str = 'genbench3d_results/',
                 training_mols: List[Mol] = None,
                 show_plots: bool = False,
                 tfd_threshold: float = DEFAULT_TFD_THRESHOLD,
                 ) -> None:
        
        self.root = root
        self.show_plots = show_plots
        self.tfd_threshold = tfd_threshold
        
        if training_mols is not None:
            self.training_cel = ConfEnsembleLibrary.from_mol_list(training_mols)
        else:
            self.training_cel = None
        
        
    def initialize(self):
        
        self.graph_metrics: List[Metric] = [Validity2D(),
                                            Uniqueness2D(),
                                            Diversity2D(),
                                            RingProportion()]
        self.mol_properties: List[str] = ['MW', 'logP', 'SAScore', 'QED']
        for property_name in self.mol_properties:
            self.graph_metrics.append(MolProperty(name=property_name))
        
        self.training_graph_metrics: List[TrainingMetric] = [Novelty2D(),
                                                            MaxTrainSim()]
        
        self.validity3D = Validity3D()
        self.conf_metrics: List[Metric] = [
           self.validity3D,
                                           StrainEnergy()]
        self.valid_conf_metrics: List[Metric] = [Uniqueness3D(self.tfd_threshold),
                                                 Diversity3D()]
        self.training_valid_conf_metrics: List[TrainingMetric] = [Novelty3D(self.tfd_threshold)]
        
        self.results = {}
   
    
    def get_results_for_cel(self,
                            cel: GeneratedCEL,
                            ) -> Dict[str, Any]:
        
        self.initialize()

        for metric in self.graph_metrics:
            metric_name = metric.name
            logging.info(f'Computing {metric_name}')
            self.results[metric_name] = metric.get(cel)
        
        if self.training_cel is not None:
            for metric in self.training_graph_metrics:
                metric_name = metric.name
                logging.info(f'Computing {metric_name}')
                self.results[metric_name] = metric.get(cel, self.training_cel)
        
        for metric in self.conf_metrics:
            metric_name = metric.name
            logging.info(f'Computing {metric_name}')
            self.results[metric_name] = metric.get(cel)
            
        logging.info('Compute valid CEL for further analysis')
        valid_conf_ids = self.validity3D.valid_conf_ids
        valid_cel = GeneratedCEL.get_cel_subset(cel=cel,
                                                subset_conf_ids=valid_conf_ids)
        for metric in self.valid_conf_metrics:
            metric_name = metric.name
            logging.info(f'Computing {metric_name}')
            self.results[metric_name] = metric.get(cel=valid_cel)
            
        if self.training_cel is not None:
            for metric in self.training_valid_conf_metrics:
                metric_name = metric.name
                logging.info(f'Computing {metric_name}')
                self.results[metric_name] = metric.get(cel=valid_cel, 
                                                       training_cel=self.training_cel)
                   
        median_n_inv_bonds, median_n_inv_angles = self.validity3D.get_invalid_bonds_angles()
        self.results['Number of invalid bonds'] = median_n_inv_bonds
        self.results['Number of invalid angles'] = median_n_inv_angles
               
        return self.results
    
    
    def get_results_for_ase_db(self, 
                                filepath: str,
                                cel_name: str = 'generated_molecules',
                                n_total_mols: int = None,
                                ) -> Dict[str, Any]:
        reader = ASEDBReader()
        cel, self.n_total_mols = reader.read(filepath=filepath, 
                                            cel_name=cel_name)
        if n_total_mols: # User defined number of generated molecules
            self.n_total_mols = n_total_mols
        
        return self.get_results_for_cel(cel)
    
    
    def get_results_for_sdf_dir(self, 
                                dirpath: str,
                                cel_name: str = 'generated_molecules',
                                n_total_mols: int = None,
                                ) -> Dict[str, Any]:
        mols = []
        for filename in os.listdir(dirpath):
            filepath = os.path.join(dirpath, filename)
            if filepath.endswith('.sdf') and not filename.startswith('traj'):
                with Chem.SDMolSupplier(filepath) as suppl:
                    mols.extend([mol for mol in suppl])
        self.n_total_mols = len(mols)
        if n_total_mols is not None: # User defined number of generated molecules
            self.n_total_mols = n_total_mols
        
        new_mols = []
        none_mol_counter = 0
        for mol in mols:
            if mol is None or mol.GetNumHeavyAtoms() == 0:
                none_mol_counter += 1
            else:
                new_mols.append(mol)
        mols = new_mols
        self.none_mol_counter = none_mol_counter
        
        cel = ConfEnsembleLibrary.from_mol_list(mol_list=mols, 
                                                cel_name=cel_name)
        
        return self.get_results_for_cel(cel)
    
    
    def get_results_for_mol_list(self,
                                  mols: List[Mol],
                                  n_total_mols: int = None,
                                  ):
        cel = GeneratedCEL.from_mol_list(mols, 
                                         n_total_mols)
        return self.get_results_for_cel(cel)
    
    
    # def get_novelty_3D_for_cel(self, 
    #                            cel: ConfEnsembleLibrary,
    #                             valid_only: bool = True,
    #                            ) -> float:
    #     print('Computing Novelty3D')
    #     self.n_novel = 0
    #     if valid_only:
    #         conf_set = self.valid_confs
    #     else:
    #         conf_set = {name: [conf.GetId() for conf in ce.mol.GetConformers()]
    #                     for name, ce in cel.items()}
    #     self.novelties = defaultdict(list)
    #     for name, conf_ids in tqdm(conf_set.items()):
    #         if name not in self.training_cel.keys():
    #             self.n_novel += len(conf_ids)
    #         else:
    #             ce = cel[name]
    #             mol = Mol(ce.mol)
    #             confs = [conf for conf in mol.GetConformers()]
    #             if not valid_only: # remove invalid conformers
    #                 for conf in confs:
    #                     conf_id = conf.GetId()
    #                     if conf_id not in conf_ids:
    #                         mol.RemoveConformer(conf_id)
                        
    #             training_ce = self.training_cel[name]
    #             training_mol = Mol(training_ce.mol)
                    
    #             for conf1 in mol.GetConformers():
                    
    #                 tfds = []
    #                 is_novel = False
    #                 for conf2 in training_mol.GetConformers():
    #                     tfd = GetTFDBetweenMolecules(mol, 
    #                                                  training_mol, 
    #                                                  confId1=conf1.GetId(), 
    #                                                  confId2=conf2.GetId())
    #                     tfds.append(tfd)
    #                     is_novel = tfd > self.tfd_threshold
    #                     if is_novel: 
    #                         break
                    
    #                 self.novelties[name].append(is_novel)
    #                 if is_novel:
    #                     self.n_novel += 1
        
    #     if valid_only:
    #         if self.n_valid > 0:
    #             novelty_3d = self.n_novel / self.n_valid
    #         else:
    #             novelty_3d = 0
    #     else:
    #         novelty_3d = self.n_novel / self.n_confs
        
    #     return novelty_3d
                        
    
    # def get_diversity_3D_for_cel(self,
    #                              cel: ConfEnsembleLibrary,
    #                              valid_only: bool = True,
    #                              ) -> float:
    #     print('Computing Diversity3D')
    #     icds = []
    #     if valid_only:
    #         conf_set = self.valid_confs
    #     else:
    #         conf_set = {name: [conf.GetId() for conf in ce.mol.GetConformers()]
    #                     for name, ce in cel.items()}
    #     self.icds = defaultdict(list)
    #     for name, conf_ids in tqdm(conf_set.items()):
            
    #         try:
    #             ce = cel[name]

    #             # if len(conf_ids) == 1:
    #             #     icd = 0
    #             # elif len(conf_ids) > 1:
    #             if len(conf_ids) > 1:
    #                 mol = ce.mol
    #                 mol = Mol(mol)
    #                 confs = [conf for conf in mol.GetConformers()]
                    
    #                 for conf in confs:
    #                     conf_id = conf.GetId()
    #                     if conf_id not in conf_ids:
    #                         # import pdb; pdb.set_trace()
    #                         mol.RemoveConformer(conf_id)
                            
    #                 if name in self.tfd_matrices:
    #                     tfd_matrix = self.tfd_matrices[name]
    #                 else:
    #                     print('Check TFD matrix storage')
    #                     tfd_matrix = GetTFDMatrix(mol)
    #                 icd = np.mean(tfd_matrix)
                    
    #                 icds.append(icd)
    #                 self.icds[name].append(icd)
    #         except Exception as e:
    #             print('Diversity 3D exception: ', e)
                
    #     if len(icds) > 0:
    #         diversity_3D = np.mean([icd 
    #                     for icd in icds 
    #                     if icd != 0])
    #     else:
    #         diversity_3D = 0
    #     return diversity_3D
        
        
    # def get_max_sim_training(self,
    #                          cel: ConfEnsembleLibrary) -> List[float]:
    #     print('Computing MaxSimTraining')
    #     max_sims = []
    #     self.max_sims = {}
    #     self.training_ecfps = [GetMorganFingerprintAsBitVect(ce.mol, 
    #                                                          3, 
    #                                                          useChirality=True)
    #                             for name, ce in self.training_cel.library.items()]
    #     for name, ce in tqdm(cel.items()):
    #         if name in self.novel_names:
    #             ecfp = GetMorganFingerprintAsBitVect(ce.mol, 3, useChirality=True)
    #             sims = BulkTanimotoSimilarity(ecfp, self.training_ecfps)
    #             max_sim = np.max(sims)
    #             # check if 2 fps are the same even though smiles is not
    #             if max_sim == 1: 
    #                 print(name)
    #                 print(np.argmax(sims))
    #             max_sims.append(max_sim)
    #             self.max_sims[name] = max_sim
                
    #     return max_sims
        
        
    # def get_invalid_bonds_angles(self):
    #     n_invalid_bonds_l = []
    #     n_invalid_angles_l = []
    #     invalidity_dfs = []
    #     for name, invalidity_list in self.validity3D.invalidities.items():
    #         # columns: name, conf_id, geometry_type, string, value
    #         invalidity_df = pd.DataFrame(invalidity_list) 
    #         if invalidity_df.shape[0] > 0:
    #             invalid_numbers_df = invalidity_df.pivot_table(values='value', 
    #                                                             index=['name', 'conf_id'],
    #                                                             columns=['geometry_type'],
    #                                                             aggfunc='count')
    #             if 'bond' in invalid_numbers_df.columns:
    #                 n_invalid_bonds_l.extend(invalid_numbers_df['bond'].values)
    #             if 'angle' in invalid_numbers_df.columns: 
    #                 n_invalid_angles_l.extend(invalid_numbers_df['angle'].values)
                
    #             invalidity_dfs.append(invalid_numbers_df)
            
    #     if len(invalidity_dfs) > 0:
    #         self.invalidity_df = pd.concat(invalidity_dfs)
                    
    #         if self.show_plots:
    #             sns.histplot(x=n_invalid_bonds_l)
    #             plt.xlabel('Number of invalid bonds')
    #             plt.show()
            
    #             sns.histplot(x=n_invalid_angles_l)
    #             plt.xlabel('Number of invalid angles')
    #             plt.show()
        
    #     return n_invalid_bonds_l, n_invalid_angles_l
                
                
    # def get_mol_property_distribution(self, 
    #                                    cel: ConfEnsembleLibrary, 
    #                                    property_name : str):
    #     logging.info('Evaluating property: ', property_name)
        
    #     self.properties = {}
        
    #     assert property_name in ['MW', 'logP', 'SAScore', 'QED']
    #     if property_name == 'MW':
    #         property_func = self.mol_properties_calculator.get_mw
    #     elif property_name == 'logP':
    #         property_func = self.mol_properties_calculator.get_logp
    #     elif property_name == 'SAScore':
    #         property_func = self.mol_properties_calculator.get_sascore
    #     elif property_name == 'QED':
    #         property_func = self.mol_properties_calculator.get_qed
    #     else:
    #         raise Exception('Property function not coded')
        
    #     property_list = []
    #     for name, ce in tqdm(cel.items()) :
    #         mol = ce.get_mol()
    #         property_value = property_func(mol)
    #         property_list.append(property_value)
    #         self.properties[name] = property_value
            
    #     if self.show_plots:
    #         series = pd.Series(property_list, name=property_name)
    #         df = pd.DataFrame(series)
    #         sns.histplot(data=df, x=property_name)
    #         plt.show()
        
    #     return property_list
                
                
    # def get_strain_energies_for_cel(self,
    #                                 cel: ConfEnsembleLibrary):
    #     self.strain_energies = {}
    #     strain_energies = []
    #     for name, ce in tqdm(cel.items()) :
    #         mol = ce.mol
    #         mol_strain_energies = []
    #         for conf in mol.GetConformers():
    #             conf_id = conf.GetId()
    #             strain_energy = self.energy_calculator.compute_strain_energy(mol, 
    #                                                                          conf_id)
    #             mol_strain_energies.append(strain_energy)
    #         self.strain_energies[name] = mol_strain_energies
    #         strain_energies.extend(mol_strain_energies)
            
    #     if self.show_plots:
    #         colname = 'Strain energy'
    #         series = pd.Series(strain_energies, name=colname)
    #         df = pd.DataFrame(series)
    #         sns.histplot(data=df, x=colname)
    #         plt.show()
            
    #     return strain_energies
        
    