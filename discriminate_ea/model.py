import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

### Main Model, DiscriminatEA
class DiscriminatEA(nn.Module):
    def __init__(self, ent_name_emb, ent_dw_emb, 
                ent_types, use_name=True, use_structure=True, emb_size=64, 
                structure_size=8, device="cuda"):

        super(DiscriminatEA, self).__init__()

        self.device = device
        self.use_name = use_name
        self.use_structure = use_structure

        self.emb_size = emb_size
        self.struct_size = structure_size
        if ent_types is not None:
            self.ent_types = ent_types.to(self.device)
            self.num_entity_types = len(torch.unique(self.ent_types))
        else:
            self.ent_types = None
            self.num_entity_types = 0

        linear_size_1 = 0
        
        # Initialize name embeddings and layers
        if self.use_name:
            linear_size_1 += self.emb_size
            self.ent_name_emb = torch.tensor(ent_name_emb).to(self.device).float()
            self.fc_name_0 = nn.Linear(self.ent_name_emb.shape[-1], emb_size)
            self.fc_name = nn.Linear(emb_size, emb_size)
        else:
            self.ent_name_emb = None
            self.fc_name_0 = None
            self.fc_name = None
        
        if self.use_structure:
            linear_size_1 += self.struct_size
            self.ent_dw_emb = torch.tensor(ent_dw_emb).to(self.device).float()
            self.fc_dw_0 = nn.Linear(self.ent_dw_emb.shape[-1], emb_size)
            self.fc_dw = nn.Linear(emb_size, self.struct_size)
        
        self.fc_final = nn.Linear(linear_size_1, emb_size)

        self.dropout = nn.Dropout(p=0.3)
        self.activation = nn.ReLU()

    def forward(self):
        features = []

        if self.use_name:
            ent_name_feature = self.fc_name(self.fc_name_0(self.dropout(self.ent_name_emb)))
            features.append(ent_name_feature)
        
        if self.use_structure:
            ent_dw_feature = self.fc_dw(self.fc_dw_0(self.dropout(self.ent_dw_emb)))
            features.append(ent_dw_feature)

        output_feature = torch.cat(features, dim=1)
        output_feature = self.fc_final(output_feature)
        
        return output_feature