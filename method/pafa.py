import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=None, output_dim=None, attention=False, proj_type='end2end',norm_type='bn'):
        super().__init__()
        self.attention = attention
        self.proj_type = proj_type
        
        # Set default output_dim if None
        if hidden_dim is None:
            hidden_dim = input_dim
            
        if attention:
            self.temporal_attn = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Softmax(dim=1)
            )
        
        # MLP projection with optional hidden layer
        if hidden_dim is not None:
            if norm_type == 'bn':
                self.projection = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
            elif norm_type == 'ln':
                self.projection = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
        else:
            if norm_type == 'bn':
                self.projection = nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.BatchNorm1d(output_dim),
                    nn.ReLU(),
                    nn.Linear(output_dim, output_dim)
                )
            elif norm_type == 'ln':
                self.projection = nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.ReLU(),
                    nn.Linear(output_dim, output_dim)
                )
        
    def forward(self, x):
        # Handle gradient options first
        if self.proj_type == 'feat_fixed':
            x = x.detach()
            
        if self.attention:
            # Temporal attention weights
            attn_weights = self.temporal_attn(x)
            x = torch.sum(x * attn_weights, dim=1)
        
        # Project features
        x = self.projection(x)
        
        # Handle proj_fixed option
        if self.proj_type == 'proj_fixed':
            x = x.detach()
            
        return x
    
    
    


class PAFALoss(nn.Module):
    
    def __init__(self, eps=1e-6):
        super(PAFALoss, self).__init__()
        self.eps = eps

    def forward(self, features, patient_ids=None, lambda_pcsl=0.1, lambda_gpal=0.1):
        
        loss = self.compute_patient_loss(features, patient_ids, lambda_pcsl, lambda_gpal)
        return loss

    def compute_patient_loss(self, features, patient_ids, lambda_pcsl=0.1, lambda_gpal=0.1):

        device = features.device
        unique_ids = torch.unique(patient_ids)
        if unique_ids.numel() <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        within_variance = 0.0
        centroids = []
        
        # For each patient, calculate the centroid and within-patient variance
        for pid in unique_ids:
            mask = (patient_ids == pid)
            features_pid = features[mask]
            centroid = features_pid.mean(dim=0)
            centroids.append(centroid)
            within_variance += torch.sum((features_pid - centroid) ** 2)
        
        centroids = torch.stack(centroids, dim=0)  # [num_patients, feature_dim]

        # Calculate the total squared distance between centroids of different patients (between_distance)
        between_distance = 0.0
        num_patients = centroids.shape[0]
        for i in range(num_patients):
            for j in range(i + 1, num_patients):
                between_distance += torch.norm(centroids[i] - centroids[j]) ** 2

        # PCSL Loss: within_variance / (between_distance + eps)
        loss_pcsl = within_variance / (between_distance + self.eps)

        # GPAL Loss: Encourage each patient's centroid to be close to the global centroid
        global_centroid = centroids.mean(dim=0)
        loss_gpal = torch.mean(torch.norm(centroids - global_centroid, dim=1) ** 2)

        return lambda_pcsl * loss_pcsl + lambda_gpal * loss_gpal