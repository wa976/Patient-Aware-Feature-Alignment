import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=None, output_dim=None, model='beats', proj_type='end2end',norm_type='bn'):
        super().__init__()
        self.model = model
        self.proj_type = proj_type
        
        # Set default output_dim if None
        if hidden_dim is None:
            hidden_dim = input_dim
            
        if model == 'beats':
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
            
        if self.model == 'beats':
            # Temporal attention weights
            attn_weights = self.temporal_attn(x)
            x = torch.sum(x * attn_weights, dim=1)
        
        # Project features
        x = self.projection(x)
        
        # Handle proj_fixed option
        if self.proj_type == 'proj_fixed':
            x = x.detach()
            
        return x




class PatientVarianceInvarianceLoss(nn.Module):
    """
    VarianceInvarianceLoss는 입력 feature에 대해 두 종류의 정규화 loss를 계산합니다.
    
    1. Patient-level loss:
       - Patient Variance Ratio Loss (PVR Loss): 동일 환자 내 feature들의 분산과 환자 간 centroid 간 거리 비율.
       - Patient Invariance Loss (PI Loss): 환자별 centroid들이 global centroid 주위에 모이도록 함.
       
    2. Class-level loss (옵션):
       - Class Variance Ratio Loss (CVR Loss): 동일 클래스 내 feature들의 분산과 클래스 간 centroid 간 거리 비율.
       - Class Invariance Loss (CI Loss): 클래스별 centroid들이 global centroid 주위에 모이도록 함.
       
    인자로 use_patient와 use_class를 선택하여 환자 정보와 클래스 정보를 각각 고려할 수 있습니다.
    각 항의 가중치는 lambda_pvr, lambda_pi, lambda_cvr, lambda_ci로 조절할 수 있습니다.
    """
    def __init__(self, eps=1e-6):
        super(PatientVarianceInvarianceLoss, self).__init__()
        self.eps = eps


    def forward(self, features, patient_ids=None, lambda_pvr=0.1, lambda_pi=0.1):
        """
        Args:
            features: Tensor, shape [batch_size, feature_dim]
            patient_ids: Tensor, shape [batch_size], 환자 ID (use_patient=True일 경우 필요)
            class_labels: Tensor, shape [batch_size], 클래스 label (use_class=True일 경우 필요)
        Returns:
            total_loss: 두 항의 가중합으로 구성된 scalar loss.
        """

    
        loss = self.compute_patient_loss(features, patient_ids,lambda_pvr, lambda_pi)

  
        return loss

    def compute_patient_loss(self, features, patient_ids,lambda_pvr=0.1, lambda_pi=0.1):
        """
        Patient-level loss: PVR Loss + PI Loss
        """
        device = features.device
        unique_ids = torch.unique(patient_ids)
        if unique_ids.numel() <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        within_variance = 0.0
        centroids = []
        # 각 환자별로 centroid와 within variance 계산
        for pid in unique_ids:
            mask = (patient_ids == pid)
            features_pid = features[mask]
            centroid = features_pid.mean(dim=0)
            centroids.append(centroid)
            within_variance += torch.sum((features_pid - centroid) ** 2)
        centroids = torch.stack(centroids, dim=0)  # [num_patients, feature_dim]

        # 환자 간 centroid 간의 거리 제곱의 총합 (between_distance)
        between_distance = 0.0
        num_patients = centroids.shape[0]
        for i in range(num_patients):
            for j in range(i + 1, num_patients):
                between_distance += torch.norm(centroids[i] - centroids[j]) ** 2

        # PVR Loss: within variance / (between distance + eps)
        loss_pvr = within_variance / (between_distance + self.eps)

        # Patient Invariance Loss (PI Loss): 각 환자 centroid가 global centroid에 가까워지도록 함
        global_centroid = centroids.mean(dim=0)
        loss_pi = torch.mean(torch.norm(centroids - global_centroid, dim=1) ** 2)


        return lambda_pvr * loss_pvr + lambda_pi * loss_pi
