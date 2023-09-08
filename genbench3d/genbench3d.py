import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Any, Dict, List
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Mol
from conf_ensemble import ConfEnsembleLibrary
from data.generated_sample_reader import ASEDBReader
from tqdm import tqdm
from rdkit.Chem.TorsionFingerprints import GetTFDMatrix, GetTFDBetweenMolecules
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity

from .metrics.validity3d import Validity3D
from .metrics.energy_calculator import FFEnergyCalculator
from .metrics.mol_properties import MolPropertiesCalculator

class GenBench3D():
    
    def __init__(self,
                 root: str = 'genbench3d_results/',
                 training_mols: List[Mol] = None,
                 show_plots: bool = True,
                 tfd_threshold: float = 0.2,
                 ) -> None:
        
        self.root = root
        self.show_plots = show_plots
        self.tfd_threshold = tfd_threshold
        
        self.validity3D = Validity3D()
        self.energy_calculator = FFEnergyCalculator()
        self.mol_properties_calculator = MolPropertiesCalculator()
        
        if training_mols is not None:
            self.training_cel = ConfEnsembleLibrary.from_mol_list(training_mols)
        else:
            self.training_cel = None
   
    
    def get_metrics_for_cel(self,
                            cel: ConfEnsembleLibrary,
                            ) -> Dict[str, Any]:
        
        self.cel = cel
        self.metrics = {}
        n_parsed_mols = len(cel.library)
        n_total_confs = sum([ce.mol.GetNumConformers()
                             for name, ce in cel.library.items()])
        validity2D = n_total_confs / self.n_total_mols
        self.metrics['Validity2D'] = validity2D
        
        validity3D = self.get_validity_3D_for_cel(cel)
        self.metrics['Validity3D'] = validity3D
        
        uniqueness2D = n_parsed_mols / n_total_confs
        self.metrics['Uniqueness2D'] = uniqueness2D
        
        uniqueness3D = self.get_uniqueness_3D_for_cel(cel)
        self.metrics['Uniqueness3D'] = uniqueness3D
        
        if self.training_cel is not None:
        
            self.novel_names = [name 
                                for name in cel.library
                                if name not in self.training_cel.library ]
            self.n_novel = len(self.novel_names)
            novelty2D = self.n_novel / n_parsed_mols
            self.metrics['Novelty2D'] = novelty2D
            
            max_training_sims = self.get_max_sim_training(cel)
            self.metrics['Max training similarity'] = np.median(max_training_sims)
            
            novelty3D = self.get_novelty_3D_for_cel(cel)
            self.metrics['Novelty3D'] = novelty3D
            
        diversity2D = self.get_diversity_2D_for_cel(cel)
        self.metrics['Diversity2D'] = diversity2D
        
        diversity3D = self.get_diversity_3D_for_cel(cel)
        self.metrics['Diversity3D'] = diversity3D
                   
        n_invalid_bonds, n_invalid_angles = self.get_invalid_bonds_angles()
        n_invalid_bonds = [v for v in n_invalid_bonds if not np.isnan(v)]
        n_invalid_angles = [v for v in n_invalid_angles if not np.isnan(v)]
        self.metrics['mean_number_invalid_bonds'] = np.mean(n_invalid_bonds)
        self.metrics['mean_number_invalid_angles'] = np.mean(n_invalid_angles)
                   
        properties = ['MW', 'logP', 'SAScore', 'QED']
        median_prop_dict = {prop: self.get_mol_property_distribution(cel, prop) 
                            for prop in properties}
        
        self.properties_df = pd.DataFrame(median_prop_dict)
        
        for prop in properties:
            try:
                self.metrics[f'median_{prop}'] = self.properties_df[prop].median()
            except:
                import pdb;pdb.set_trace()
                
        strain_energies = self.get_strain_energies_for_cel(cel)
        self.metrics['median_strain_energy'] = np.median(strain_energies)
               
        return self.metrics
    
    
    def get_metrics_for_ase_db(self, 
                                filepath: str,
                                cel_name: str = 'generated_molecules',
                                n_total_mols: int = None,
                                ) -> Dict[str, Any]:
        reader = ASEDBReader()
        cel, self.n_total_mols = reader.read(filepath=filepath, 
                                            cel_name=cel_name)
        if n_total_mols: # User defined number of generated molecules
            self.n_total_mols = n_total_mols
        
        return self.get_metrics_for_cel(cel)
    
    
    def get_metrics_for_sdf_dir(self, 
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
        
        return self.get_metrics_for_cel(cel)
    
    
    def get_metrics_for_mol_list(self,
                                  mols: List[Mol],
                                  n_total_mols: int = None,
                                  ):
        self.cel = ConfEnsembleLibrary.from_mol_list(mols)
        self.n_total_mols = len(mols)
        if n_total_mols is not None: # User defined number of generated molecules
            self.n_total_mols = n_total_mols
        
        return self.get_metrics_for_cel(self.cel)
    
    
    def get_validity_3D_for_cel(self, 
                                cel: ConfEnsembleLibrary
                                ) -> float:
        print('Computing Validity3D')
        self.n_confs = 0
        self.n_valid = 0
        self.valid_confs = {} # {name: [valid_idx1, valid_idx2...]}
        for name, ce in tqdm(cel.library.items()):
            mol = ce.mol
            self.n_confs += mol.GetNumConformers()
            valid_conf_ids = self.validity3D.get_valid_conf_ids_for_mol(mol, 
                                                                         name)
            if len(valid_conf_ids) > 0:
                self.valid_confs[name] = valid_conf_ids
                self.n_valid += len(valid_conf_ids)
        validity_3D = self.n_valid / self.n_confs
        
        return validity_3D
                
    
    def get_uniqueness_3D_for_cel(self, 
                                  cel: ConfEnsembleLibrary,
                                  valid_only=True,
                                  ) -> float:
        print('Computing Uniqueness3D')
        self.n_unique = 0
        self.unique_mol = {}
        self.n_clusters = {}
        self.tfd_matrices = {}
        if valid_only:
            conf_set = self.valid_confs
        else:
            conf_set = {name: [conf.GetId() for conf in ce.mol.GetConformers()]
                        for name, ce in cel.library.items()}
        for name, conf_ids in tqdm(conf_set.items()):
            
            ce = cel.library[name]

            if len(conf_ids) == 1:
                self.n_unique += 1
            elif len(conf_ids) > 1:
                mol = ce.mol
                mol = Mol(mol)
                confs = [conf for conf in mol.GetConformers()]
                if not valid_only: # remove invalid conformers 
                    for conf in confs:
                        conf_id = conf.GetId()
                        if conf_id not in conf_ids:
                            mol.RemoveConformer(conf_id)
                        
                try:
                    tfd_matrix = GetTFDMatrix(mol)
                    n_confs = mol.GetNumConformers()
                    tfd_matrix = self.get_full_matrix_from_tril(tfd_matrix, 
                                                                n=n_confs)
                    tfd_matrix = squareform(tfd_matrix) # to get to triu
                    
                    self.tfd_matrices[name] = tfd_matrix

                    Z = linkage(tfd_matrix)

                    max_value = self.tfd_threshold
                    T = fcluster(Z, 
                                t=max_value, 
                                criterion='distance')
                    n_clusters = max(T)
                    self.n_unique += n_clusters
                    self.n_clusters[mol] = n_clusters
                    
                    if n_clusters == 1 and mol.GetNumConformers() > 1:
                        self.unique_mol[name] = ce.mol
                except Exception as e:
                    print('Uniqueness 3D exception:', e)
                    
            else:
                import pdb;pdb.set_trace()
                raise RuntimeError()
                
        if valid_only:
            if self.n_valid > 0:
                uniqueness_3d = self.n_unique / self.n_valid
            else:
                uniqueness_3d = 0
        else:
            uniqueness_3d = self.n_unique / self.n_confs
        return uniqueness_3d
    
    
    def get_novelty_3D_for_cel(self, 
                               cel: ConfEnsembleLibrary,
                                valid_only: bool = True,
                               ) -> float:
        print('Computing Novelty3D')
        self.n_novel = 0
        if valid_only:
            conf_set = self.valid_confs
        else:
            conf_set = {name: [conf.GetId() for conf in ce.mol.GetConformers()]
                        for name, ce in cel.library.items()}
        self.novelties = defaultdict(list)
        for name, conf_ids in tqdm(conf_set.items()):
            if name not in self.training_cel.library:
                self.n_novel += len(conf_ids)
            else:
                ce = cel.library[name]
                mol = Mol(ce.mol)
                confs = [conf for conf in mol.GetConformers()]
                if not valid_only: # remove invalid conformers
                    for conf in confs:
                        conf_id = conf.GetId()
                        if conf_id not in conf_ids:
                            mol.RemoveConformer(conf_id)
                        
                training_ce = self.training_cel.library[name]
                training_mol = Mol(training_ce.mol)
                    
                for conf1 in mol.GetConformers():
                    
                    tfds = []
                    is_novel = False
                    for conf2 in training_mol.GetConformers():
                        tfd = GetTFDBetweenMolecules(mol, 
                                                     training_mol, 
                                                     confId1=conf1.GetId(), 
                                                     confId2=conf2.GetId())
                        tfds.append(tfd)
                        is_novel = tfd > self.tfd_threshold
                        if is_novel: 
                            break
                    
                    self.novelties[name].append(is_novel)
                    if is_novel:
                        self.n_novel += 1
        
        if valid_only:
            if self.n_valid > 0:
                novelty_3d = self.n_novel / self.n_valid
            else:
                novelty_3d = 0
        else:
            novelty_3d = self.n_novel / self.n_confs
        
        return novelty_3d
                        
          
    def get_diversity_2D_for_cel(self, 
                                 cel: ConfEnsembleLibrary
                                 ) -> float:
        print('Computing Diversity2D')
        mols = []
        for name, ce in cel.library.items():
            mols.append(ce.mol)
            
        dists = []
        self.ecfps = [GetMorganFingerprintAsBitVect(mol, 3, useChirality=True)
                 for mol in mols]
        n_mols = len(mols)
        for i in tqdm(range(n_mols)):
            fp1 = self.ecfps[i]
            for j in range(i + 1, n_mols):
                fp2 = self.ecfps[j]
                sim = TanimotoSimilarity(fp1, fp2)
                dist = 1 - sim
                dists.append(dist)
                
        self.dists = dists
        return np.mean(dists)
    
    
    def get_diversity_3D_for_cel(self,
                                 cel: ConfEnsembleLibrary,
                                 valid_only: bool = True,
                                 ) -> float:
        print('Computing Diversity3D')
        icds = []
        if valid_only:
            conf_set = self.valid_confs
        else:
            conf_set = {name: [conf.GetId() for conf in ce.mol.GetConformers()]
                        for name, ce in cel.library.items()}
        self.icds = defaultdict(list)
        for name, conf_ids in tqdm(conf_set.items()):
            
            try:
                ce = cel.library[name]

                # if len(conf_ids) == 1:
                #     icd = 0
                # elif len(conf_ids) > 1:
                if len(conf_ids) > 1:
                    mol = ce.mol
                    mol = Mol(mol)
                    confs = [conf for conf in mol.GetConformers()]
                    
                    for conf in confs:
                        conf_id = conf.GetId()
                        if conf_id not in conf_ids:
                            # import pdb; pdb.set_trace()
                            mol.RemoveConformer(conf_id)
                            
                    if name in self.tfd_matrices:
                        tfd_matrix = self.tfd_matrices[name]
                    else:
                        print('Check TFD matrix storage')
                        tfd_matrix = GetTFDMatrix(mol)
                    icd = np.mean(tfd_matrix)
                    
                    icds.append(icd)
                    self.icds[name].append(icd)
            except Exception as e:
                print('Diversity 3D exception: ', e)
                
        if len(icds) > 0:
            diversity_3D = np.mean([icd 
                        for icd in icds 
                        if icd != 0])
        else:
            diversity_3D = 0
        return diversity_3D
        
        
    def get_max_sim_training(self,
                             cel: ConfEnsembleLibrary) -> List[float]:
        print('Computing MaxSimTraining')
        max_sims = []
        self.max_sims = {}
        self.training_ecfps = [GetMorganFingerprintAsBitVect(ce.mol, 
                                                             3, 
                                                             useChirality=True)
                                for name, ce in self.training_cel.library.items()]
        for name, ce in tqdm(cel.library.items()):
            if name in self.novel_names:
                ecfp = GetMorganFingerprintAsBitVect(ce.mol, 3, useChirality=True)
                sims = BulkTanimotoSimilarity(ecfp, self.training_ecfps)
                max_sim = np.max(sims)
                # check if 2 fps are the same even though smiles is not
                if max_sim == 1: 
                    print(name)
                    print(np.argmax(sims))
                max_sims.append(max_sim)
                self.max_sims[name] = max_sim
                
        return max_sims
        
        
    def get_invalid_bonds_angles(self):
        n_invalid_bonds_l = []
        n_invalid_angles_l = []
        invalidity_dfs = []
        for name, invalidity_list in self.validity3D.invalidities.items():
            # columns: name, conf_id, geometry_type, string, value
            invalidity_df = pd.DataFrame(invalidity_list) 
            if invalidity_df.shape[0] > 0:
                invalid_numbers_df = invalidity_df.pivot_table(values='value', 
                                                                index=['name', 'conf_id'],
                                                                columns=['geometry_type'],
                                                                aggfunc='count')
                if 'bond' in invalid_numbers_df.columns:
                    n_invalid_bonds_l.extend(invalid_numbers_df['bond'].values)
                if 'angle' in invalid_numbers_df.columns: 
                    n_invalid_angles_l.extend(invalid_numbers_df['angle'].values)
                
                invalidity_dfs.append(invalid_numbers_df)
            
        if len(invalidity_dfs) > 0:
            self.invalidity_df = pd.concat(invalidity_dfs)
                    
            if self.show_plots:
                sns.histplot(x=n_invalid_bonds_l)
                plt.xlabel('Number of invalid bonds')
                plt.show()
            
                sns.histplot(x=n_invalid_angles_l)
                plt.xlabel('Number of invalid angles')
                plt.show()
        
        return n_invalid_bonds_l, n_invalid_angles_l
                
                
    def get_mol_property_distribution(self, 
                                       cel: ConfEnsembleLibrary, 
                                       property_name : str):
        print('Evaluating property: ', property_name)
        
        self.properties = {}
        
        assert property_name in ['MW', 'logP', 'SAScore', 'QED']
        if property_name == 'MW':
            property_func = self.mol_properties_calculator.get_mw
        elif property_name == 'logP':
            property_func = self.mol_properties_calculator.get_logp
        elif property_name == 'SAScore':
            property_func = self.mol_properties_calculator.get_sascore
        elif property_name == 'QED':
            property_func = self.mol_properties_calculator.get_qed
        else:
            raise Exception('Property function not coded')
        
        property_list = []
        for name, ce in tqdm(cel.library.items()) :
            mol = ce.mol
            property_value = property_func(mol)
            property_list.append(property_value)
            self.properties[name] = property_value
            
        if self.show_plots:
            series = pd.Series(property_list, name=property_name)
            df = pd.DataFrame(series)
            sns.histplot(data=df, x=property_name)
            plt.show()
        
        return property_list
                
                
    def get_strain_energies_for_cel(self,
                                    cel: ConfEnsembleLibrary):
        self.strain_energies = {}
        strain_energies = []
        for name, ce in tqdm(cel.library.items()) :
            mol = ce.mol
            mol_strain_energies = []
            for conf in mol.GetConformers():
                conf_id = conf.GetId()
                strain_energy = self.energy_calculator.compute_strain_energy(mol, 
                                                                             conf_id)
                mol_strain_energies.append(strain_energy)
            self.strain_energies[name] = mol_strain_energies
            strain_energies.extend(mol_strain_energies)
            
        if self.show_plots:
            colname = 'Strain energy'
            series = pd.Series(strain_energies, name=colname)
            df = pd.DataFrame(series)
            sns.histplot(data=df, x=colname)
            plt.show()
            
        return strain_energies
        
    
    @staticmethod
    def get_full_matrix_from_tril(tril_matrix, n):
        # let the desired full distance matrix between 4 samples be
        # [a b c d
        #  e f g h
        #  i j k l
        #  m n o p]
        # where the diagonal a = f = l = p = 0
        # having b = e, c = i, d = m, g = j, h = n, l = o by symmetry
        # scipy squareform works on triu [b c d g h l]
        # while we want to work from tril [e i j m n o]
        
        matrix = np.zeros((n, n))
        i=1
        j=0
        for v in tril_matrix:
            matrix[i, j] = matrix[j, i] = v
            j = j + 1
            if j == i:
                i = i + 1
                j = 0
        return matrix