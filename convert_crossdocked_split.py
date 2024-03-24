import os
import torch
import pickle
from genbench3d.params import CROSSDOCKED_SPLITS_PT_PATH, CROSSDOCKED_SPLITS_P_PATH

splits = torch.load(CROSSDOCKED_SPLITS_PT_PATH)
with open(CROSSDOCKED_SPLITS_P_PATH, 'wb') as f:
    pickle.dump(splits, f)

# if not os.path.exists(train_ligand_path):
#     splits = torch.load(split_path)

#     train_ligands = []
#     train_data = list(splits['train'])
#     for duo in tqdm(train_data):
#         pocket_path, ligand_path = duo
#         ligand_path = os.path.join(cross_docked_data_path, ligand_path)
#         ligand = [mol for mol in Chem.SDMolSupplier(str(ligand_path))][0]
#         train_ligands.append(ligand)
#     with Chem.SDWriter(train_ligand_path) as writer:
#         for ligand in train_ligands:
#             writer.write(ligand)
# else:
#     train_ligands = [mol for mol in Chem.SDMolSupplier(train_ligand_path)]