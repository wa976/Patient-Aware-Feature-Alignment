import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .icbhi_util import get_annotations, generate_fbank
from .icbhi_util import get_individual_cycles_torchaudio, cut_pad_sample_torchaudio,get_individual_cycles_torchaudio_beats
from .augmentation import augment_raw_audio



class ICBHIDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True, mean_std=False):

        # 시드 설정
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        
        data_folder = os.path.join(args.data_folder, 'icbhi_dataset/audio_test_data')
        folds_file = os.path.join(args.data_folder, 'icbhi_dataset/patient_list_foldwise.txt')
        official_folds_file = os.path.join(args.data_folder, 'icbhi_dataset/official_split.txt')
        test_fold = args.test_fold
        
        self.data_folder = data_folder
        self.train_flag = train_flag
        self.split = 'train' if train_flag else 'test'
        self.transform = transform
        self.args = args
        self.mean_std = mean_std

        # parameters for spectrograms
        self.sample_rate = args.sample_rate
        self.desired_length = args.desired_length
        self.pad_types = args.pad_types
        self.n_mels = args.n_mels
        self.f_min = 50
        self.f_max = 2000
        
        
        
        cache_path = './data/training.pt' if self.train_flag else './data/test.pt'
        
        
        if not os.path.isfile(cache_path):


            # ==========================================================================
            """ get ICBHI dataset meta information """
            # store stethoscope device information for each file or patient
            self.file_to_device = {}
            self.device_to_id = {'Meditron': 0, 'LittC2SE': 1, 'Litt3200': 2, 'AKGC417L': 3}
            self.device_id_to_patient = {0: [], 1: [], 2: [], 3: []}

            filenames = sorted(os.listdir(data_folder))
            filenames = sorted(set([f.strip().split('.')[0] for f in filenames if '.wav' in f or '.txt' in f]))
            
            
            for f in filenames:
                f += '.wav'
                # get the total number of devices from original dataset (icbhi dataset has 4 stethoscope devices)
                device = f.strip().split('_')[-1].split('.')[0]

                # get the device information for each wav file
                self.file_to_device[f.strip().split('.')[0]] = self.device_to_id[device]

                pat_id = f.strip().split('_')[0]
                if pat_id not in self.device_id_to_patient[self.device_to_id[device]]:
                    self.device_id_to_patient[self.device_to_id[device]].append(pat_id)

            # store all metadata (age, sex, adult_BMI, child_weight, child_height, device_index)
            self.file_to_metadata = {}
            meta_file = pd.read_csv(os.path.join(args.data_folder, 'icbhi_dataset/metadata.txt'), names=['age', 'sex', 'adult_BMI', 'child_weight', 'child_height', 'chest_location'], delimiter= '\t')
            meta_file['chest_location'].replace({'Tc':0, 'Al':1, 'Ar':2, 'Pl':3, 'Pr':4, 'Ll':5, 'Lr':6}, inplace=True)
            for f in filenames:
                pat_idx = int(f.strip().split('_')[0])
                info = list(meta_file.loc[pat_idx])
                info[1] = 0 if info[1] == 'M' else 1

                info = np.array(info)
                for idx in np.argwhere(np.isnan(info)):
                    info[idx] = -1

                self.file_to_metadata[f] = torch.tensor(np.append(info, self.file_to_device[f.strip()]))

            # ==========================================================================
            
            # ==========================================================================
            """ train-test split based on train_flag and test_fold """
            if test_fold in ['0', '1', '2', '3', '4']:  # from RespireNet, 80-20% split
                patient_dict = {}
                all_patients = open(folds_file).read().splitlines()
                for line in all_patients:
                    idx, fold = line.strip().split(' ')
                    if train_flag and int(fold) != int(test_fold):
                        patient_dict[idx] = fold
                    elif train_flag == False and int(fold) == int(test_fold):
                        patient_dict[idx] = fold
                
                if print_flag:
                    print('*' * 20)
                    print('Train and test 80-20% split with test_fold {}'.format(test_fold))
                    print('Patience number in {} dataset: {}'.format(self.split, len(patient_dict)))
            else:  
                """ 
                args.test_fold == 'official', 60-40% split
                two patient dataset contain both train and test samples
                """
                patient_dict = {}
                all_fpath = open(official_folds_file).read().splitlines()
                for line in all_fpath:
                    fpath, fold = line.strip().split('\t')
                    if train_flag and fold == 'train':
                        # idx = fpath.strip().split('_')[0]
                        patient_dict[fpath] = fold
                    elif not train_flag and fold == 'test':
                        # idx = fpath.strip().split('_')[0]
                        patient_dict[fpath] = fold

                if print_flag:
                    print('*' * 20)
                    print('Train and test 60-40% split with test_fold {}'.format(test_fold))
                    print('File number in {} dataset: {}'.format(self.split, len(patient_dict)))
            # ==========================================================================

            # dict {filename: annotations}, annotation is for breathing cycle
            annotation_dict = get_annotations(args, data_folder)

            self.filenames = []
            for f in filenames:
                # for 'official' test_fold, two patient dataset contain both train and test samples
                idx = f.split('_')[0] if test_fold in ['0', '1', '2', '3', '4'] else f
               
                if idx in patient_dict:
                    self.filenames.append(f)
            
            self.audio_data = []  # each sample is a tuple with (audio_data, label, filename)
            self.labels = []

            if print_flag:
                print('*' * 20)  
                print("Extracting individual breathing cycles..")

            self.cycle_list = []
            self.filename_to_label = {}
            self.classwise_cycle_list = [[] for _ in range(args.n_cls)]

            # ==========================================================================
            """ extract individual cycles by librosa or torchaudio """
            for idx, filename in enumerate(self.filenames):
                # you can use self.filename_to_label to get statistics of original sample labels (will not be used on other function)
                self.filename_to_label[filename] = []
                

                # "SCL" version: get original cycles 6,898 by torchaudio and cut_pad samples
                sample_data = get_individual_cycles_torchaudio(args, annotation_dict[filename],  data_folder, filename, args.sample_rate, args.n_cls)

                
                patient_id = filename.split('_')[0]  
                cycles_with_labels = [(data[0], data[1], patient_id) for data in sample_data]

                self.cycle_list.extend(cycles_with_labels)
                for d in cycles_with_labels:
                    # {filename: [label for cycle 1, ...]}
                    self.filename_to_label[filename].append(d[1])
                    self.classwise_cycle_list[d[1]].append(d)
                    
         
            for sample in self.cycle_list:
                self.audio_data.append(sample)
            # ==========================================================================

            self.class_nums = np.zeros(args.n_cls)
            for sample in self.audio_data:
                self.class_nums[sample[1]] += 1
                self.labels.append(sample[1])
            self.class_ratio = self.class_nums / sum(self.class_nums) * 100
            
            if print_flag:
                print('[Preprocessed {} dataset information]'.format(self.split))
                print('total number of audio data: {}'.format(len(self.audio_data)))
                for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                    print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.cls_list[i]+')', int(n), p))    
            
            # ==========================================================================
            """ convert mel-spectrogram """
            self.audio_images = []
            
            self.patient_ids = [int(f.split('_')[0]) for f in self.filenames]
            unique_patients = np.unique(self.patient_ids)
            self.patient_map = {pid: idx for idx, pid in enumerate(unique_patients)}
            
            
            for index in range(len(self.audio_data)):
                audio, label,patient_id = self.audio_data[index][0], self.audio_data[index][1],self.audio_data[index][2]
                
                
                mapped_patient_id = self.patient_map[int(patient_id)]
                     
                filename = [f for f in self.filenames if f.startswith(f"{patient_id}_")][0]
                device_label = self.file_to_device[filename.rsplit('.', 1)[0]]
                
                
                if args.model == 'beats':
                    audio_data = []
                    for aug_idx in range(self.args.raw_augment+1):
                        if aug_idx > 0:
                            if self.train_flag and not mean_std:
                                augmented_audio = augment_raw_audio(audio, self.sample_rate, self.args)
                                augmented_audio = cut_pad_sample_torchaudio(torch.tensor(augmented_audio), args)
                                audio_data.append(audio)
                            else:
                                audio_data.append(None)
                        else:
                            audio_data.append(torch.tensor(audio))
                    
                    self.audio_images.append((audio_data, label, device_label, int(mapped_patient_id)))
                
                
                
                
                else:
                    audio_image = []
                    # self.aug_times = 1 + 5 * self.args.augment_times  # original + five naa augmentations * augment_times (optional)
                    for aug_idx in range(self.args.raw_augment+1): 
                        if aug_idx > 0:
                            if self.train_flag and not mean_std:
                                audio = augment_raw_audio(audio, self.sample_rate, self.args)

                                # "SCL" version: cut longer sample or pad sample
                                audio = cut_pad_sample_torchaudio(torch.tensor(audio), args)
                            else:
                                audio_image.append(None)
                                continue
        
                        image = generate_fbank(audio, self.sample_rate, n_mels=self.n_mels)


                        audio_image.append(image)
                        
                        
                        
                    self.audio_images.append((audio_image, label,device_label,int(mapped_patient_id)))
                
                
                
                    self.h, self.w, _ = self.audio_images[0][0][0].shape
            
            # if self.train_flag:
            #     torch.save(self.audio_images, './data/training.pt')
            # else:
            #     torch.save(self.audio_images, './data/test.pt')

        
        else:
            if self.train_flag:
                self.audio_images = torch.load('./data/training.pt')
            else:
                self.audio_images = torch.load('./data/test.pt')
            # ==========================================================================



    
        
    def __getitem__(self, index):

        audio_images, label, device_label, patient_id = (
            self.audio_images[index][0],
            self.audio_images[index][1],
            self.audio_images[index][2],
            self.audio_images[index][3]
        )
        
        
   
        audio_image = audio_images[0]
        
        
        if self.transform is not None:
            audio_image = self.transform(audio_image)
        
        if self.args.model == 'beats':
            # Ensure minimum length of 16000 samples (1 second)
            # Remove channel dimension if exists
            if len(audio_image.shape) == 2:
                audio_image = audio_image.squeeze(0)
                
        
        
        return audio_image, (torch.tensor(label), torch.tensor(device_label), torch.tensor(patient_id))

    
    def __len__(self):
        return len(self.audio_images)
    
    
    
    
    
    
    