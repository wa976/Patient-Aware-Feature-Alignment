
import torch
from BEATs.BEATs import BEATs, BEATsConfig
import torch.nn as nn
from copy import deepcopy


class BEATsTransferLearningModel(nn.Module):
    def __init__(
        self,
        num_target_classes: int = 4,
        model_path: str = "./pretrained_models/BEATs_iter3_plus_AS2M.pt",
        ft_entire_network: bool = True,
        spec_transform = None,
    ):

        super().__init__()
        self.num_target_classes = num_target_classes

        # Initialise BEATs model
        self.checkpoint = torch.load(model_path)
        self.cfg = BEATsConfig(
            {
                **self.checkpoint["cfg"],
                "predictor_class": self.num_target_classes,
                "finetuned_model": False,
                "spec_transform": spec_transform
            }
        )

        
        self._build_model()
        
        self.final_feat_dim = self.cfg.encoder_embed_dim

    
    def _build_model(self):
        self.beats = BEATs(self.cfg)
        self.beats.load_state_dict(self.checkpoint["model"])
        

    def forward(self, x, padding_mask=None,training=False):
        """Forward pass. Return x"""


        if padding_mask != None:
            x, _  = self.beats.extract_features(x, padding_mask)
        else:
            x, _  = self.beats.extract_features(x)
                    

        return x






