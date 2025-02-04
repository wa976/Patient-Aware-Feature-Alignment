from copy import deepcopy
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import math
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from util.icbhi_dataset import ICBHIDataset
from util.icbhi_util import get_score
from util.augmentation import SpecAugment
from util.misc import adjust_learning_rate, warmup_learning_rate, set_optimizer, update_moving_average
from util.misc import AverageMeter, accuracy, save_model, update_json
from models import get_backbone_class
from method.pvil import ProjectionHead,PatientVarianceInvarianceLoss
    
    
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

def visualize_features(model, classifier, data_loader, args, projector, mode='train', projection_type=None, n_patients=10, selected_patients=None):
    """
    Visualize features using t-SNE for selected patients, with other patients in grey
    """
    model.eval()
    if mode == 'train' and args.method == 'pvil':
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
                if mode == 'train' and args.method == 'pvil':
                    if projection_type == 'ipc':
                        features = projector[0](features)
                    else:  # cpc
                        features = projector[1](features)
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
    if projection_type:
        title += f' - {projection_type.upper()} Projection'
    plt.title(title)
    
    # Add legend for classes only
    legend_elements = [plt.Line2D([0], [0], marker=marker, color='gray', 
                                label=f'Class {cls}', linestyle='None', 
                                markersize=10)
                      for cls, marker in class_markers.items()]
    plt.legend(handles=legend_elements, title='Classes')
    
    # Save the plot
    save_name = f'tsne_{mode}'
    if projection_type:
        save_name += f'_{projection_type}'
    save_name += f'_{n_patients}patients'
    plt.savefig(f'{args.save_folder}/{save_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_train_test(train_loader, test_loader, model, classifier, args, projector, n_patients=10):
    """
    Visualize both training and test features for fixed n_patients
    """
    # Get fixed patient IDs first
    selected_patients = get_fixed_patient_ids(train_loader, n_patients)
    
    if args.method == 'pvil':
        print(f"Generating t-SNE visualization for training set (IPC) with fixed {n_patients} patients...")
        visualize_features(model, classifier, train_loader, args, projector, 
                         mode='train', projection_type='ipc', n_patients=n_patients,
                         selected_patients=selected_patients)
        
        print(f"Generating t-SNE visualization for training set (CPC) with fixed {n_patients} patients...")
        visualize_features(model, classifier, train_loader, args, projector, 
                         mode='train', projection_type='cpc', n_patients=n_patients,
                         selected_patients=selected_patients)
    
    print(f"Generating t-SNE visualization for test set with fixed {n_patients} patients...")
    visualize_features(model, classifier, test_loader, args, projector, 
                      mode='test', n_patients=n_patients,
                      selected_patients=selected_patients)
    
    

            
def parse_args():
    parser = argparse.ArgumentParser('argument for supervised training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    parser.add_argument('--eval', action='store_true',
                        help='only evaluation with pretrained encoder and classifier')
    parser.add_argument('--two_cls_eval', action='store_true',
                        help='evaluate with two classes')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')
    # dataset
    parser.add_argument('--dataset', type=str, default='icbhi')
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # icbhi dataset
    parser.add_argument('--class_split', type=str, default='lungsound',
                        help='lungsound: (normal, crackles, wheezes, both), diagnosis: (healthy, chronic diseases, non-chronic diseases)')
    parser.add_argument('--n_cls', type=int, default=0,
                        help='set k-way classification problem for class')
    parser.add_argument('--d_cls', type=int, default=0,
                        help='set k-way classification problem for device (meta)')
    parser.add_argument('--test_fold', type=str, default='official', choices=['official', '0', '1', '2', '3', '4'],
                        help='test fold to use official 60-40 split or 80-20 split from RespireNet')
    parser.add_argument('--sample_rate', type=int,  default=16000, 
                        help='sampling rate when load audio data, and it denotes the number of samples per one second')
    parser.add_argument('--desired_length', type=int,  default=8, 
                        help='fixed length size of individual cycle')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='the number of mel filter banks')
    parser.add_argument('--pad_types', type=str,  default='repeat', 
                        help='zero: zero-padding, repeat: padding with duplicated samples, aug: padding with augmented samples')
    parser.add_argument('--resz', type=float, default=1, 
                        help='resize the scale of mel-spectrogram')
    parser.add_argument('--raw_augment', type=int, default=0, 
                        help='control how many number of augmented raw audio samples')
    parser.add_argument('--specaug_policy', type=str, default='icbhi_ast_sup', 
                        help='policy (argument values) for SpecAugment')
    parser.add_argument('--specaug_mask', type=str, default='mean', 
                        help='specaug mask value', choices=['mean', 'zero'])
    parser.add_argument('--nospec', action='store_true')

    # model
    parser.add_argument('--model', type=str, default='beats')
    parser.add_argument('--pretrained', action='store_true')
    
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained encoder model')
    parser.add_argument('--from_sl_official', action='store_true',
                        help='load from supervised imagenet-pretrained model (official PyTorch)')
    parser.add_argument('--ma_update', action='store_true',
                        help='whether to use moving average update for model')
    parser.add_argument('--ma_beta', type=float, default=0,
                        help='moving average value')
    
    # for AST
    parser.add_argument('--audioset_pretrained', action='store_true',
                        help='load from imagenet- and audioset-pretrained model')
    parser.add_argument('--method', type=str, default='ce') 
    

    
    # PVIL
    parser.add_argument('--norm_type', type=str, default='bn',
                        help='normalization type', choices=['bn', 'ln'])
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='projection hidden dimension')
    parser.add_argument('--output_dim', type=int, default=128,
                        help='projection output dimension')
    parser.add_argument('--proj_type', type=str, default='end2end',
                        help='projection type', choices=['end2end', 'feat_fixed', 'proj_fixed'])
    parser.add_argument('--lambda_pvr', type=float, default=0.1,
                        help='lambda for patient variance ratio loss')
    parser.add_argument('--lambda_pi', type=float, default=0.1,
                        help='lambda for patient invariance loss')
    parser.add_argument('--w_ce', type=float, default=1.0,
                    help='weight for classification loss')
    parser.add_argument('--w_pvil', type=float, default=0.5,
                    help='weight for patient loss')
    
    
                        
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.model_name = '{}_{}_{}'.format(args.dataset, args.model, args.method)
    if args.tag:
        args.model_name += '_{}'.format(args.tag)

    
    args.save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

    args.d_cls = 2    
            
    if args.dataset == 'icbhi':
        if args.class_split == 'lungsound':
            if args.n_cls == 4:
                args.cls_list = ['normal', 'crackle', 'wheeze', 'both']
            elif args.n_cls == 2:
                args.cls_list = ['normal', 'abnormal']
            
            args.device_list = ['L', 'A', 'M', '3']
                
        elif args.class_split == 'diagnosis':
            if args.n_cls == 3:
                args.cls_list = ['healthy', 'chronic_diseases', 'non-chronic_diseases']
            elif args.n_cls == 2:
                args.cls_list = ['healthy', 'unhealthy']
            else:
                raise NotImplementedError
        
    else:
        raise NotImplementedError
    
    if args.n_cls == 0 and args.m_cls !=0:
        args.n_cls = args.m_cls
        args.cls_list = args.meta_cls_list

    return args

def set_loader(args):
    if args.dataset == 'icbhi':        
        args.h = int(args.desired_length * 100 - 2)
        args.w = 128
        
        
        if args.model == 'beats':
            train_transform = None
            val_transform = None
 
        else:
            train_transform = [transforms.ToTensor(),
                            transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
            val_transform = [transforms.ToTensor(),
                        transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        
            ##
            train_transform = transforms.Compose(train_transform)
            val_transform = transforms.Compose(val_transform)
        

        train_dataset = ICBHIDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
        val_dataset = ICBHIDataset(train_flag=False, transform=val_transform, args=args,  print_flag=True)
    
        
        
    else:
        raise NotImplemented    
    
    
   
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    
    
    return train_loader, val_loader, args


        
        
def set_model(args):
    kwargs = {}
    if args.model == 'ast':
        kwargs['input_fdim'] = int(args.h * args.resz)
        kwargs['input_tdim'] = int(args.w * args.resz)
        kwargs['label_dim'] = args.n_cls
        kwargs['imagenet_pretrain'] = args.from_sl_official
        kwargs['audioset_pretrain'] = args.audioset_pretrained
 
        
    elif args.model == 'beats':
        if args.nospec:
            kwargs['spec_transform'] = None
        else:
            kwargs['spec_transform'] = SpecAugment(args)

    
    model = get_backbone_class(args.model)(**kwargs)
    
    

    if args.model == 'beats' and args.method == 'pvil':
        classifier = nn.Linear(model.final_feat_dim, args.n_cls).cuda()
        projector = ProjectionHead(model.final_feat_dim, args.hidden_dim, args.output_dim, model='beats', norm_type=args.norm_type, proj_type=args.proj_type).cuda()
    elif args.model == 'ast' and args.method == 'pvil':
        classifier = nn.Linear(model.final_feat_dim, args.n_cls).cuda()
        projector_pvrl = ProjectionHead(model.final_feat_dim, args.hidden_dim, args.output_dim, model='ast', norm_type=args.norm_type, proj_type=args.proj_type).cuda()
        projector_pi= ProjectionHead(model.final_feat_dim, args.hidden_dim, args.output_dim, model='ast', norm_type=args.norm_type, proj_type=args.proj_type).cuda()
        projector = [projector_pvrl, projector_pi]
    elif args.model == 'cnn6' and args.method == 'pvil':
        classifier = nn.Linear(model.final_feat_dim, args.n_cls).cuda()
        projector = ProjectionHead(model.final_feat_dim, args.hidden_dim, args.output_dim, model='cnn6', norm_type=args.norm_type, proj_type=args.proj_type).cuda()
    else:
        classifier = nn.Linear(model.final_feat_dim, args.n_cls).cuda() if args.model not in ['ast'] else deepcopy(model.mlp_head).cuda()
        projector = nn.Identity()
    

               
    if args.model not in ['ast','beats'] and args.from_sl_official:
        model.load_sl_official_weights()
        print('pretrained model loaded from PyTorch ImageNet-pretrained')

    # load SSL pretrained checkpoint for linear evaluation
    if args.pretrained and args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = ckpt['model']

        # HOTFIX: always use dataparallel during SSL pretraining
        new_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "backbone." in k:
                k = k.replace("backbone.", "")
            if not 'mlp_head' in k: #del mlp_head
                new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        
        
        if ckpt.get('classifier', None) is not None:
            print("correct")

            classifier.load_state_dict(ckpt['classifier'], strict=True)

        print('pretrained model loaded from: {}'.format(args.pretrained_ckpt))
    
    criterion = nn.CrossEntropyLoss()
    pvil = PatientVarianceInvarianceLoss()
    
    if args.method == 'ce':
        criterion = [criterion.cuda()]
    elif args.method == 'pvil':
        criterion = [criterion.cuda(), pvil.cuda()]


    model.cuda()
    
   
   
    # optim_params = list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters())
    
    optim_params = list(model.parameters()) + list(classifier.parameters()) + list(projector[0].parameters()) + list(projector[1].parameters())
    optimizer = set_optimizer(args, optim_params)
    
    return model, classifier, projector, criterion, optimizer




def train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler=None):
   
    
    model.train()
    
    classifier.train()
    # projector.train()
    
    projector[0].train()
    projector[1].train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    

    end = time.time()
    
    for idx, (images, labels) in enumerate(train_loader):
        
        # data load
        data_time.update(time.time() - end)
         
        images = images.cuda(non_blocking=True)
        
    
        class_labels = labels[0].cuda(non_blocking=True)
        device_labels = labels[1].cuda(non_blocking=True)
        patient_labels = labels[2].cuda(non_blocking=True)
        
        
        bsz = class_labels.shape[0] 
        

        if args.ma_update:
            # store the previous iter checkpoint
            with torch.no_grad():
                ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict()), deepcopy(projector[0].state_dict()), deepcopy(projector[1].state_dict())]

                # ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict()), deepcopy(projector.state_dict())]
                lamb = None

        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast():
            if args.method == 'ce':
                
                if args.model == 'beats':
                    features  = model(images,training=True)
                
                
                    output = classifier(features)
        
        
                    output = output.mean(dim=1)
          
                    
                    loss = criterion[0](output, class_labels)

                else:
                    if args.nospec:
                        features = model(images, args=args,training=True)
                    else:
                        features = model(args.transforms(images), args=args,training=True)

                    output = classifier(features)
                    
                    loss = criterion[0](output, class_labels)
                    
            elif args.method == 'pvil':
                if args.model == 'beats':
                    features= model(images,training=True)
                    
                    
                    output_class = classifier(features)
                    
                    output_pvil = projector(features)

                  
                        
                    output = output_class.mean(dim=1)
                    
                    cls_loss = criterion[0](output, class_labels)
                    pvil_loss = criterion[1](output_pvil, patient_labels,class_labels,lambda_pvr=args.lambda_pvr, lambda_pi=args.lambda_pi)
                    loss = args.w_ce * cls_loss + args.w_pvil * pvil_loss 
                else:
                    if args.nospec:
                        features = model(images, args=args,training=True)
                    else:
                        features = model(args.transforms(images), args=args,training=True)
                        
                    
                    # projector_output = projector(features)
                    projector_output = projector[0](features)
                    
                    output = classifier(features)
                    
                    loss_class = criterion[0](output, class_labels)
                    loss_pvr = criterion[1](projector_output, patient_labels,lambda_pvr=args.lambda_pvr, lambda_pi=args.lambda_pi)
                    
                    loss = args.w_ce * loss_class + args.w_pvil * loss_pvr

         
            
            losses.update(loss.item(), bsz)
       
        [acc1], _ = accuracy(output[:bsz], class_labels, topk=(1,))
        top1.update(acc1[0], bsz)
        
        optimizer.zero_grad()
    
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.ma_update:
            with torch.no_grad():
                # exponential moving average update
                model = update_moving_average(args.ma_beta, model, ma_ckpt[0])
                classifier = update_moving_average(args.ma_beta, classifier, ma_ckpt[1])
                # projector = update_moving_average(args.ma_beta, projector, ma_ckpt[2])

                projector[0] = update_moving_average(args.ma_beta, projector[0], ma_ckpt[2])
                projector[1] = update_moving_average(args.ma_beta, projector[1], ma_ckpt[3])
        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg

    


    
def validate(val_loader, model, classifier, criterion, args, best_acc, best_model=None,projector=None):
    save_bool = False
    model.eval()

    classifier.eval()
    # projector.eval()
    
    projector[0].eval()
    projector[1].eval() 
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    hits, counts = [0.0] * args.n_cls, [0.0] * args.n_cls

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            
            class_labels = labels[0].cuda(non_blocking=True)
        
            labels = class_labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
                  
            with torch.cuda.amp.autocast():
                if args.model == 'beats':
                    features = model(images,training=False)
                    output = classifier(features)
                    output = output.mean(dim=1)
                else:
                    features = model(images, args=args, training=False)
                    output = classifier(features)
                loss = criterion[0](output, labels)
                

                
                
            losses.update(loss.item(), bsz)
            [acc1], _ = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            _, preds = torch.max(output, 1)
            
            
        
            for idx in range(preds.shape[0]):
                counts[labels[idx].item()] += 1.0
                if not args.two_cls_eval:
                    if preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                else:  # only when args.n_cls == 4
                    if labels[idx].item() == 0 and preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                    elif labels[idx].item() != 0 and preds[idx].item() > 0:  # abnormal
                        hits[labels[idx].item()] += 1.0

                

            sp, se, sc, f1_normal = get_score(hits, counts)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'S_p {sp:.3f}\t'
                      'S_e {se:.3f}\t'
                      'Score {sc:.3f}\t'
                      'F1 Score {f1:.3f}'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1, sp=sp, se=se, sc=sc,
                       f1=f1_normal))
                
    
    
    if sc > best_acc[-2] and se > 0.1:
        save_bool = True
        best_acc = [sp, se, sc, f1_normal]
        best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict()), deepcopy(projector[0].state_dict()), deepcopy(projector[1].state_dict())]

        # best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict()), deepcopy(projector.state_dict())]

    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(sp, se, sc, best_acc[0], best_acc[1], best_acc[2]))
    print(' * F1 Score: {:.2f} (F1 Score: {:.2f})'.format(f1_normal, best_acc[3]))
    print(' * Acc@1 {top1.avg:.2f}'.format(top1=top1))

    return best_acc, best_model, save_bool




def main():
    args = parse_args()
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    
    best_model = None
    if args.dataset == 'icbhi':
        best_acc = [0, 0, 0, 0]  # Specificity, Sensitivity, Score
    
    if not args.nospec:
        print("sepc")
        args.transforms = SpecAugment(args)
        
        
    train_loader, val_loader, args = set_loader(args)

    
    
    model, classifier, projector, criterion, optimizer = set_model(args)


            
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 1

    # use mix_precision:
    scaler = torch.cuda.amp.GradScaler()
    
    print('*' * 20)
    print('Checkpoint Name: {}'.format(args.model_name))
     
    if not args.eval:
        print('Training for {} epochs on {} dataset'.format(args.epochs, args.dataset))
        for epoch in range(args.start_epoch, args.epochs+1):
        
            
            adjust_learning_rate(args, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            
            loss, acc = train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2-time1, acc))
            
            # eval for one epoch
            best_acc, best_model, save_bool = validate(val_loader, model, classifier, criterion, args, best_acc, best_model, projector)
            
            # save a checkpoint of model and classifier when the best score is updated
            if save_bool:            
                save_file = os.path.join(args.save_folder, 'best_epoch_{}.pth'.format(epoch))
                print('Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc[2], epoch))
                # print('Best ckpt is modified with F1 = {:.2f} when Epoch = {}'.format(best_acc[3], epoch))
                save_model(model, optimizer, args, epoch, save_file,  classifier, projector)
                
                        
            if epoch % args.save_freq == 0:
                save_file = os.path.join(args.save_folder, 'epoch_{}.pth'.format(epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier, projector)
            

        # save a checkpoint of classifier with the best accuracy or score
        save_file = os.path.join(args.save_folder, 'best.pth')
        model.load_state_dict(best_model[0])
        classifier.load_state_dict(best_model[1])
        # projector.load_state_dict(best_model[2])
        projector[0].load_state_dict(best_model[2])
        projector[1].load_state_dict(best_model[3])
        save_model(model, optimizer, args, epoch, save_file, classifier, projector)
        
    else:
        visualize_train_test(train_loader, val_loader, model, classifier, args, projector)

        print("correct")
        print('Testing the pretrained checkpoint on {} dataset'.format(args.dataset))
        best_acc, _, _  = validate(val_loader, model, classifier, criterion, args, best_acc, best_model, projector)
        model.eval()  # Set the model to evaluation mode
        

    update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results.json'))
    print('Checkpoint {} finished'.format(args.model_name))
    
if __name__ == '__main__':
    main()

