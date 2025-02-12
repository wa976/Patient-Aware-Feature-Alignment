import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Add this line before importing pyplot
import seaborn as sns

def get_fixed_patient_ids(data_loader, n_patients=10):
    """
    Get fixed patient IDs for consistent visualization across different runs
    """
    all_patient_ids = []
    for _, labels in data_loader:
        all_patient_ids.extend(labels[2].numpy())
    
    unique_patients = np.unique(all_patient_ids)
    # Sort to ensure deterministic behavior
    unique_patients.sort()
    
    # Use fixed seed and fixed starting index for reproducibility
    np.random.seed(42)
    if len(unique_patients) > n_patients:
        # Instead of random choice, use fixed indices
        selected_indices = np.arange(len(unique_patients))
        np.random.shuffle(selected_indices)
        # Save these indices to a file if they don't exist
        indices_file = 'selected_patient_indices.npy'
        if not os.path.exists(indices_file):
            np.save(indices_file, selected_indices[:n_patients])
        else:
            selected_indices = np.load(indices_file)
        selected_patients = unique_patients[selected_indices]
    else:
        selected_patients = unique_patients
        
    return selected_patients

def visualize_features(model, classifier, data_loader, args, projector, mode='train', n_patients=10, selected_patients=None):
    """
    Visualize features using t-SNE for selected patients, with other patients in grey
    """
    model.eval()
    if mode == 'train' and args.method == 'pafa':
        projector.eval()
    
    # If no patients are pre-selected, get the fixed set
    if selected_patients is None:
        selected_patients = get_fixed_patient_ids(data_loader, n_patients)
    
    # Ensure exactly n_patients are selected
    selected_patients = selected_patients[:n_patients]
    
    # Lists to store all data
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
    
    # Create color palette for selected patients
    colors = plt.cm.tab10(np.linspace(0, 1, n_patients))
    patient_colors = {patient: color for patient, color in zip(selected_patients, colors)}
    
    # Create markers for classes
    markers = ['o', 's', '^', 'D']
    class_markers = {cls: marker for cls, marker in zip(range(args.n_cls), markers)}
    
    # Create the plot
    plt.figure(figsize=(10, 10))
    
    # First plot all non-selected patients in grey
    mask_others = ~np.isin(all_patient_labels, selected_patients)
    for cls in range(args.n_cls):
        mask = mask_others & (all_class_labels == cls)
        if np.any(mask):
            plt.scatter(all_features_tsne[mask, 0], 
                      all_features_tsne[mask, 1],
                      c='lightgray',
                      marker=class_markers[cls],
                      s=100,
                      alpha=0.2)
    
    # Then plot selected patients with colors (on top)
    for patient in selected_patients:
        for cls in range(args.n_cls):
            mask = (all_patient_labels == patient) & (all_class_labels == cls)
            if np.any(mask):
                plt.scatter(all_features_tsne[mask, 0], 
                          all_features_tsne[mask, 1],
                          c=[patient_colors[patient]],
                          marker=class_markers[cls],
                          s=100,
                          alpha=0.8)
    
    title = f'T-SNE Visualization ({mode} set)'

    plt.title(title)
    
    # Add legend for classes only
    legend_elements = [plt.Line2D([0], [0], marker=marker, color='gray', 
                                label=f'Class {cls}', linestyle='None', 
                                markersize=10)
                      for cls, marker in class_markers.items()]
    plt.legend(handles=legend_elements, title='Classes')
    
    # Save the plot
    save_name = f'tsne_{mode}'
    save_name += f'_{n_patients}patients'
    plt.savefig(f'{args.save_folder}/{save_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_train_test(train_loader, test_loader, model, classifier, args, projector, n_patients=10):
    """
    Visualize both training and test features for fixed n_patients
    """
    # Get fixed patient IDs first
    selected_patients = get_fixed_patient_ids(train_loader, n_patients)
    
 
    print(f"Generating t-SNE visualization for training set with fixed {n_patients} patients...")
    visualize_features(model, classifier, train_loader, args, projector, 
                        mode='train', n_patients=n_patients,
                        selected_patients=selected_patients)

    
    print(f"Generating t-SNE visualization for test set with fixed {n_patients} patients...")
    visualize_features(model, classifier, test_loader, args, projector, 
                      mode='test', n_patients=n_patients,
                      selected_patients=selected_patients)