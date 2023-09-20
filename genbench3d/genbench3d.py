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
    