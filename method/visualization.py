import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Add this line before importing pyplot
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def get_fixed_patient_ids(data_loader,mode):

    all_patient_ids = []
    for _, labels in data_loader:
        all_patient_ids.extend(labels[2].numpy())
    
    unique_patients, counts = np.unique(all_patient_ids, return_counts=True)
    
    sorted_indices = np.argsort(counts)
    

    if mode == 'train':
        selected_patients = np.array([110,112,218,222,166,157,120]) ## sub cluster
        # selected_patients = np.array([107,130,154,158,172,203]) ## high sample

    
   
    
    return selected_patients




def visualize_features(model, classifier, data_loader, args, projector, mode='train', selected_patients=None):

    model.eval()
    if mode == 'train' and args.method == 'pafa':
        projector.eval()
    
    if selected_patients is None:
        selected_patients = get_fixed_patient_ids(data_loader)
    
    # Lists to store data
    all_features = []
    all_patient_labels = []
    all_class_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.cuda()
            
            if args.model == 'beats':
                features = model(images, training=False)
                if mode == 'train' and args.method == 'pafa':
                    features = projector(features)
                else:
                    features = classifier(features)
                    features = features.mean(dim=1)
                
                all_features.append(features.cpu().numpy())
                all_patient_labels.extend(labels[2].numpy())
                all_class_labels.extend(labels[0].numpy())
    
    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)
    all_patient_labels = np.array(all_patient_labels)
    all_class_labels = np.array(all_class_labels)
    
    # Apply t-SNE to all data
    tsne = TSNE(n_components=2, random_state=42)
    all_features_tsne = tsne.fit_transform(all_features)
    
    # 클래스 레이블(숫자 -> 문자) 정의
    class_labels = {0: 'Normal', 1: 'Crackle', 2: 'Wheeze', 3: 'Both'}
    
    # 선택된 환자들에 대한 color palette 생성
    unique_selected = np.unique(selected_patients)
    n_selected = len(unique_selected)
    colors = plt.cm.tab10(np.linspace(0, 1, n_selected))  # n_selected개 색상
    patient_colors = {patient: color for patient, color in zip(unique_selected, colors)}
    
    # 클래스별로 다른 marker 사용
    markers = ['o', 's', '^', 'D']
    class_markers = {cls: markers[cls % len(markers)] for cls in range(args.n_cls)}
    
    # Plot 시작
    plt.figure(figsize=(10, 10))
    
    # 1) 선택되지 않은 환자들(others)은 연한 회색(lightgray)으로 표시
    mask_others = ~np.isin(all_patient_labels, unique_selected)
    for cls in range(args.n_cls):
        mask = mask_others & (all_class_labels == cls)
        if np.any(mask):
            plt.scatter(
                all_features_tsne[mask, 0], 
                all_features_tsne[mask, 1],
                c='lightgray',
                marker=class_markers[cls],
                s=130,
                alpha=0.2
            )
    
    # 2) 선택된 환자들 각각은 고유 색상 + 클래스 마커로 표시
    for patient in unique_selected:
        for cls in range(args.n_cls):
            mask = (all_patient_labels == patient) & (all_class_labels == cls)
            if np.any(mask):
                plt.scatter(
                    all_features_tsne[mask, 0], 
                    all_features_tsne[mask, 1],
                    c=[patient_colors[patient]],
                    marker=class_markers[cls],
                    s=130,
                    edgecolor='black',
                    alpha=0.8,
                )
    
    plt.title(f'T-SNE Visualization BEATs + PAFA', fontsize=26)
    
    plt.tick_params(axis='both', which='major', labelsize=20)

    
    # ---------- 범례(legend) 구성 ----------
    # (1) 클래스 범례: 마커로 구분
    class_legend_elements = []
    for cls, marker in class_markers.items():
        class_legend_elements.append(
            Line2D([0], [0],
                   marker=marker, color='black', 
                   label=class_labels[cls],  # 문자 레이블
                   linestyle='None',
                   markersize=16
                   )
        )
    
    # (2) 환자 범례: 색으로 구분
    #     + 'Others' (연한 회색)도 추가
    patient_legend_elements = []
    # - 선택된 각 환자
    for patient in unique_selected:
        patient_legend_elements.append(
            Patch(facecolor=patient_colors[patient],
                  edgecolor='black',
                  label=f'Patient {patient}'
                  )
        )
    # - Others
    patient_legend_elements.append(
        Patch(facecolor='lightgray',
              edgecolor='black',
              label='Others')
    )
    
   
    legend1 = plt.legend(handles=class_legend_elements,
                         loc='upper left',
                         prop={'size': 18})
    plt.gca().add_artist(legend1)  
    

    plt.legend(handles=patient_legend_elements,
               loc='lower left',
               prop={'size': 16})
    
    # ---------------------------------------
    
    # Save the plot
    save_name = f'tsne_{mode}_{selected_patients}'
    plt.savefig(f'{args.save_folder}/{save_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_features_by_class(model, classifier, data_loader, args, projector, mode='train'):

    model.eval()
    classifier.eval()
    if mode == 'train' and args.method == 'pafa':
        projector.eval()
    
    all_features = []
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.cuda()
            
            # 모델을 통해 feature 추출 (t-SNE용)
            feats = model(images, training=False)
            if mode == 'train' and args.method == 'pafa':
                features_embed = projector(feats)
            else:
                features_embed = classifier(feats)
                if features_embed.dim() == 3:
                    features_embed = features_embed.mean(dim=1)
            
            # 별도로 분류기를 이용하여 예측 클래스 산출
            logits = classifier(feats)
            if logits.dim() == 3:
                logits = logits.mean(dim=1)
            preds = logits.argmax(dim=-1).cpu().numpy()
            
            all_features.append(features_embed.cpu().numpy())
            all_true.extend(labels[0].numpy())
            all_pred.extend(preds)
    
    all_features = np.concatenate(all_features, axis=0)
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    
    # t-SNE 적용
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(all_features)
    
    # 클래스 레이블 정의 (필요시 수정)
    class_labels = {0: 'Normal', 1: 'Crackle', 2: 'Wheeze', 3: 'Both'}
    n_cls = args.n_cls
    
    # 클래스별 색상 팔레트 생성 (plt.cm.tab10 사용)
    colors = plt.cm.tab10(np.linspace(0, 1, n_cls))
    color_map = {i: colors[i] for i in range(n_cls)}
    
        # ------------------------- Plot 1: True Labels -------------------------
    plt.figure(figsize=(8, 8))
    for cls in range(n_cls):
        mask = all_true == cls
        if np.any(mask):
            plt.scatter(
                features_tsne[mask, 0],
                features_tsne[mask, 1],
                color=color_map[cls],
                marker='o',
                s=100,
                edgecolor='black',
                alpha=0.7,
                label=f'{class_labels[cls]}'
            )
    plt.title(f'T-SNE by True Class BEATs + PAFA', fontsize=26)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(prop={'size': 16},markerscale=1.5,loc='upper left')
    
    save_name_true = f'tsne_class_true_{mode}'
    plt.savefig(f'{args.save_folder}/{save_name_true}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ------------------------- Plot 2: Predicted Labels -------------------------
    plt.figure(figsize=(8, 8))
    for cls in range(n_cls):
        mask = all_pred == cls
        if np.any(mask):
            plt.scatter(
                features_tsne[mask, 0],
                features_tsne[mask, 1],
                facecolors='none',
                edgecolor=color_map[cls],
                marker='s',
                s=100,
                alpha=0.7,
                label=f'{class_labels[cls]}'
            )
    plt.title(f'T-SNE by Predicted Class BEATs + PAFA', fontsize=26)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(prop={'size': 18})
    
    save_name_pred = f'tsne_class_pred_{mode}'
    plt.savefig(f'{args.save_folder}/{save_name_pred}.png', dpi=300, bbox_inches='tight')
    plt.close()



def visualize_train_test(train_loader, test_loader, model, classifier, args, projector):

    
    selected_patients_train = get_fixed_patient_ids(train_loader,mode='train')
    
    print("Generating t-SNE visualization for training set ...")
    visualize_features(model, classifier, train_loader, args, projector, 
                       mode='train', 
                       selected_patients=selected_patients_train)


    print("Generating t-SNE visualization by class for training set ...")
    visualize_features_by_class(model, classifier, train_loader, args, projector, mode='train')
    

    