import torch
import numpy as np
from collections import defaultdict
import csv

def get_patient_centroids(data_loader, model, projector, device='cuda', save=False):
    """
    data_loader: (train_loader or val_loader)
    model: 학습된 모델 (예: beats, ast 등)
    projector: PAFA인 경우 ProjectionHead, CE-only인 경우 nn.Identity()일 수 있음
    device: cuda or cpu
    save: 환자별 샘플 수를 CSV로 저장할지 여부
    ---
    return:
        patient_centroids: {patient_id: (centroid (np.ndarray), count(int))}
            예) {
               0: (array([...]), 12),
               1: (array([...]), 5),
               ...
            }
    """

    model.eval()
    projector.eval()

    patient_features = defaultdict(list)

    with torch.no_grad():
        for images, labels in data_loader:
            # labels = (class_label, device_label, patient_label)
            patient_ids = labels[2]
            images = images.to(device, non_blocking=True)

            # model forward
            #   beats인 경우 -> model(images, training=False)
            #   ast 등 다른 모델인 경우 -> model(images, args=args, training=False)
            features = model(images, training=False)  
            # "pafa"면 실제 projection을 거치고, 
            # "ce-only"면 projector가 nn.Identity()일 수도 있음
            proj_feats = projector(features)  # shape (B, feat_dim)

            for i, pid in enumerate(patient_ids):
                # pid가 tensor라면 .item()으로 변환
                pid_val = pid.item() if isinstance(pid, torch.Tensor) else pid
                patient_features[pid_val].append(proj_feats[i].cpu().numpy())

    # 각 환자별로 평균 feature(centroid)와 샘플 수 count 저장
    patient_centroids = {}
    for pid_val, feat_list in patient_features.items():
        feat_array = np.array(feat_list)  # (N, feat_dim)
        centroid = feat_array.mean(axis=0)
        count = feat_array.shape[0]
        patient_centroids[pid_val] = (centroid, count)
    
    # 환자별 샘플 수 저장 (선택적으로 CSV 저장)
    if save:
        save_patient_sample_counts_csv(patient_centroids, 'patient_sample_counts.csv')

    return patient_centroids


def sort_patients_by_count(patient_centroids):
    """
    patient_centroids: {pid: (centroid, count)}
    ---
    return:
        sorted_pids: 환자 샘플 수(count) 기준 오름차순으로 정렬된 patient_id 리스트
    """
    # 환자별 count 기준 오름차순 정렬
    sorted_by_count = sorted(patient_centroids.items(), key=lambda x: x[1][1])
    # 환자 ID 순만 리스트로 뽑아 반환
    sorted_pids = [item[0] for item in sorted_by_count]
    return sorted_pids


def find_closest_pairs(train_centroids, test_centroids, train_pids_of_interest, top_k=1):
    """
    train_centroids: {pid: (centroid, count)}
    test_centroids:  {pid: (centroid, count)}
    train_pids_of_interest: 분석하고자 하는 train 환자들의 ID 리스트 (복수)
    top_k: 가까운 순서대로 몇 개의 test 환자를 보고 싶은지

    동작:
        1) 입력받은 train_pids_of_interest 각각의 centroid를 모음
        2) 모은 centroid들을 평균내어 '평균 centroid'를 구함
        3) 이 '평균 centroid'와 test_centroids 내 모든 환자의 centroid 간 거리를 계산
        4) distance 기준 오름차순 정렬
        5) top_k개를 반환
    ---
    return:
        closest_list: [(test_pid, distance), ...] 형태로 distance가 작은 순서대로 top_k개
    """
    # 1) 관심있는 train 환자들의 centroid만 모아서 평균 계산
    centroids_of_interest = []
    for pid_tr in train_pids_of_interest:
        centroid_tr, _ = train_centroids[pid_tr]
        centroids_of_interest.append(centroid_tr)
    
    # 여러 환자의 centroid를 평균
    avg_centroid = np.mean(centroids_of_interest, axis=0)

    # 2) test_centroids와의 거리 계산
    dist_list = []
    for pid_te, (centroid_te, _) in test_centroids.items():
        dist = np.linalg.norm(avg_centroid - centroid_te)  # Euclidean distance
        dist_list.append((pid_te, dist))

    # 3) distance 기준 오름차순 정렬 후 top_k 추출
    dist_list.sort(key=lambda x: x[1])
    closest_list = dist_list[:top_k]

    return closest_list



def save_patient_sample_counts_csv(patient_centroids, csv_filepath):
    """
    patient_centroids: {patient_id: (centroid, sample_count)}
    csv_filepath: 저장할 CSV 파일 경로 (예: "train_patient_counts.csv")
    
    각 환자의 sample count를 내림차순으로 정렬한 후, patient id와 sample_count 두 개 칼럼으로 CSV 파일을 저장한다.
    """
    # sample count 기준 내림차순 정렬
    sorted_patients = sorted(patient_centroids.items(), key=lambda x: x[1][1], reverse=True)
    
    with open(csv_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 헤더 작성
        writer.writerow(["patient_id", "sample_count"])
        # 각 환자별로 patient_id와 sample_count 작성
        for pid, (_, count) in sorted_patients:
            writer.writerow([pid, count])