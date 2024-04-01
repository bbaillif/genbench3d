import os
import logging

from typing import Any, Dict, List
from rdkit import Chem
from rdkit.Chem import Mol

from .conf_ensemble import GeneratedCEL, ConfEnsembleLibrary
from .data.generated_sample_reader import ASEDBReader
from tqdm import tqdm
from genbench3d.geometry import ReferenceGeometry
from genbench3d.data.source import CrossDocked, CSDDrug

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
                 show_plots: bool = False,
                 tfd_threshold: float = DEFAULT_TFD_THRESHOLD,
                 ) -> None:
        
        self.root = root
        self.show_plots = show_plots
        self.tfd_threshold = tfd_threshold
        
        self.training_cel = None
        
        
    def set_training_mols(self,
                          training_mols: List[Mol]) -> None:
        self.training_cel = ConfEnsembleLibrary.from_mol_list(training_mols)
        
    
    def set_training_cel(self,
                         training_cel: ConfEnsembleLibrary):
        self.training_cel = training_cel
    
        
    def initialize(self) -> None:
        
        self.graph_metrics: List[Metric] = [Validity2D(),
                                            Uniqueness2D(),
                                            Diversity2D(),
                                            RingProportion()]
        self.mol_properties: List[str] = ['MW', 'logP', 'SAScore', 'QED']
        for property_name in self.mol_properties:
            self.graph_metrics.append(MolProperty(name=property_name))
        
        self.training_graph_metrics: List[TrainingMetric] = [Novelty2D(),
                                                            MaxTrainSim()]
        
        self.csd_drug = CSDDrug()
        self.csd_geometry = ReferenceGeometry(source=self.csd_drug)
        self.validity3D_csd = Validity3D(name='Validity3D CSD',
                                         reference_geometry=self.csd_geometry)
        
        self.conf_metrics: List[Metric] = [self.validity3D_csd]
        
        # self.crossdocked = CrossDocked()
        # self.crossdocked_geometry = ReferenceGeometry(source=self.crossdocked)
        # self.validity3D_crossdocked = Validity3D(name='Validity3D CrossDocked',
        #                                          reference_geometry=self.crossdocked_geometry)
        # self.conf_metrics.append(self.validity3D_crossdocked)
        
        self.conf_metrics.append(StrainEnergy())

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
                self.results[metric_name] = metric.get(cel=cel, 
                                                       training_cel=self.training_cel)
        
        for metric in self.conf_metrics:
            metric_name = metric.name
            logging.info(f'Computing {metric_name}')
            self.results[metric_name] = metric.get(cel)
            
        self.results['Number of tested confs'] = cel.n_total_confs
            
        logging.info('Compute valid CEL for further analysis')
        valid_conf_ids = self.validity3D_csd.valid_conf_ids
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
                   
        validity3Ds = [metric for metric in self.conf_metrics if isinstance(metric, Validity3D)]
        for validity3D in validity3Ds:
                   
            self.results[f'Number of valid 3D confs ({validity3D.name})'] = validity3D.n_valid_confs
                   
            n_inv_geom = validity3D.get_invalid_geometries()
            for geometry, values in n_inv_geom.items():
                self.results[f'Number of invalid {geometry}s ({validity3D.name})'] = values
                
            new_patterns = validity3D.get_new_patterns()
            self.results[f'Number of new patterns ({validity3D.name})'] = len(new_patterns)
            self.results[f'Number of new unique patterns ({validity3D.name})'] = len(set(new_patterns))
                
            geometry_combos = [['bond'], ['angle'], ['torsion'],
                               ['bond', 'angle'],
                               ['bond', 'angle', 'torsion']]
            
            for combo in geometry_combos:
                combo_str = '+'.join(combo)
                min_q_values = validity3D.get_min_q_values_for_geometries(combo)
                self.results[f'Min {combo_str} q-value ({validity3D.name})'] = min_q_values
                
                geo_mean_q_values = validity3D.get_geo_mean_q_values_for_geometries(combo)
                self.results[f'Geometric mean {combo_str} q-value ({validity3D.name})'] = geo_mean_q_values
                
            # bond_min_q_values = validity3D.get_min_q_values_for_geometry('bond')
            # self.results[f'Min bond q-value ({validity3D.name})'] = bond_min_q_values
            # angle_min_q_values = validity3D.get_min_q_values_for_geometry('angle')
            # self.results[f'Min angle q-value ({validity3D.name})'] = angle_min_q_values
            # torsion_min_q_values = validity3D.get_min_q_values_for_geometry('torsion')
            # self.results[f'Min torsion q-value ({validity3D.name})'] = torsion_min_q_values
            # ba_min_q_values = validity3D.get_min_q_values_for_geometries(['bond', 'angle'])
            # self.results[f'Min bond+angle q-value ({validity3D.name})'] = ba_min_q_values
            # bat_min_q_values = validity3D.get_min_q_values_for_geometries(['bond', 'angle', 'torsion'])
            # self.results[f'Min bond+angle+torsion q-value ({validity3D.name})'] = bat_min_q_values
            
            # bond_geo_mean_q_values = validity3D.get_geo_mean_q_values_for_geometry('bond')
            # self.results[f'Geometric mean bond q-value ({validity3D.name})'] = bond_geo_mean_q_values
            # angle_geo_mean_q_values = validity3D.get_geo_mean_q_values_for_geometry('angle')
            # self.results[f'Geometric mean angle q-value ({validity3D.name})'] = angle_geo_mean_q_values
            # torsion_geo_mean_q_values = validity3D.get_geo_mean_q_values_for_geometry('torsion')
            # self.results[f'Geometric mean torsion q-value ({validity3D.name})'] = torsion_geo_mean_q_values
            # ba_geo_mean_q_values = validity3D.get_geo_mean_q_values_for_geometries(['bond', 'angle'])
            # self.results[f'Geometric mean bond+angle q-value ({validity3D.name})'] = ba_geo_mean_q_values
            # bat_geo_mean_q_values = validity3D.get_geo_mean_q_values_for_geometries(['bond', 'angle', 'torsion'])
            # self.results[f'Geometric mean bond+angle+torsion q-value ({validity3D.name})'] = bat_geo_mean_q_values
            
            
               
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
                                  names: list[str] = None,
                                  ):
        cel = GeneratedCEL.from_mol_list(mol_list=mols, 
                                         n_total_mols=n_total_mols,
                                         names=names)
        return self.get_results_for_cel(cel)
    