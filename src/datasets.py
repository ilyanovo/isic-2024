import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2


class ISICDatasetSamplerW(Dataset):
    def __init__(self, meta_df, transforms=None, process_target: bool=False, n_classes:int=3, weight_adg = 1, do_augmentations: bool=True, *args, **kwargs):
        self.df_positive = meta_df[meta_df["target"] == 1].reset_index()
        self.df_negative = meta_df[meta_df["target"] == 0].reset_index()
        self.file_names_positive = self.df_positive['path'].values
        self.file_names_negative = self.df_negative['path'].values
        self.targets_positive = self.df_positive['target'].values
        self.targets_negative = self.df_negative['target'].values
        self.negative_weights = self.df_negative['weight'].values
        self.negative_ind = np.arange(0, self.negative_weights.shape[0])
        self.weight_adg = weight_adg
        self.transforms = transforms
        self.n_classes = n_classes
        self.process_target = process_target
        self.do_augmentations = do_augmentations
        
    def __len__(self):
        return len(self.df_positive) * 2
    
    def __getitem__(self, index):
        if random.random() >= 0.5:
            df = self.df_positive
            file_names = self.file_names_positive
            targets = self.targets_positive
            index = index % df.shape[0]
        else:
            df = self.df_negative
            file_names = self.file_names_negative
            targets = self.targets_negative
            index = random.choices(self.negative_ind, weights=self.negative_weights ** self.weight_adg, k=1)[0]
        
        
        img_path = file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = targets[index]

        if self.transforms and self.do_augmentations: 
            img = self.transforms(image=img)["image"]
            
        if self.process_target:
            target_pr = np.zeros(shape=(self.n_classes,))
            target_pr[int(target)] += 1.0
            target = target_pr
            
        return {
            'image': img,
            'target': target
        }


class ISICDatasetSampler(Dataset):
    def __init__(self, meta_df, transforms=None, process_target: bool=False, n_classes:int=3, do_augmentations: bool=True, *args, **kwargs):
        self.df_positive = meta_df[meta_df["target"] == 1].reset_index()
        self.df_negative = meta_df[meta_df["target"] == 0].reset_index()
        self.file_names_positive = self.df_positive['path'].values
        self.file_names_negative = self.df_negative['path'].values
        self.targets_positive = self.df_positive['target'].values
        self.targets_negative = self.df_negative['target'].values
        self.transforms = transforms
        self.n_classes = n_classes
        self.process_target = process_target
        self.do_augmentations = do_augmentations
        
    def __len__(self):
        return len(self.df_positive) * 2
    
    def __getitem__(self, index):
        if random.random() >= 0.5:
            df = self.df_positive
            file_names = self.file_names_positive
            targets = self.targets_positive
        else:
            df = self.df_negative
            file_names = self.file_names_negative
            targets = self.targets_negative
        index = index % df.shape[0]
        
        img_path = file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = targets[index]

        if self.transforms and self.do_augmentations: 
            img = self.transforms(image=img)["image"]
            

        if self.process_target:
            target_pr = np.zeros(shape=(self.n_classes,))
            target_pr[int(target)] += 1.0
            target = target_pr
        
        return {
            'image': img,
            'target': target
        }
        
class ISICDatasetSimple(Dataset):
    def __init__(self, meta_df, targets=None, transforms=None, process_target: bool=False, n_classes:int=3, do_augmentations: bool=True, *args, **kwargs):
        self.meta_df = meta_df
        self.targets = targets
        self.transforms = transforms
        self.n_classes = n_classes
        self.process_target = process_target
        self.do_augmentations = do_augmentations
        

    def __len__(self):
        return self.meta_df.shape[0]

    def __getitem__(self, idx):
        target = self.meta_df.iloc[idx].target
        path = self.meta_df.iloc[idx].path

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms and self.do_augmentations:
            transformed = self.transforms(image=img)
            img = transformed['image']

        if self.process_target:
            target_pr = np.zeros(shape=(self.n_classes,))
            target_pr[int(target)] += 1.0
            target = target_pr
            
        # target = self.targets[idx]    
            
        return {
            'image': img,
            'target': target
        }



class ISICDatasetSamplerMulticlass(Dataset):
    def __init__(self, meta_df, transforms=None, process_target: bool=False, n_classes:int=3):
        self.df_2 = meta_df[meta_df["target"] == 2].reset_index()
        self.df_1 = meta_df[meta_df["target"] == 1].reset_index()
        self.df_0 = meta_df[meta_df["target"] == 0].reset_index()
        self.file_names_2 = self.df_2['path'].values
        self.file_names_1 = self.df_1['path'].values
        self.file_names_0 = self.df_0['path'].values
        self.targets_2 = self.df_2['target'].values
        self.targets_1 = self.df_1['target'].values
        self.targets_0 = self.df_0['target'].values
        self.transforms = transforms
        self.n_classes = n_classes
        self.process_target = process_target
        
    def __len__(self):
        return len(self.df_2) * 3
    
    def __getitem__(self, index):
        target_p = random.choices([0,1,2], weights=[1,1,1],k=1)[0]
        if target_p == 1:
            df = self.df_1
            file_names = self.file_names_1
            targets = self.targets_1
        elif target_p == 2:
            df = self.df_2
            file_names = self.file_names_2
            targets = self.targets_2
        else:
            df = self.df_0
            file_names = self.file_names_0
            targets = self.targets_0
            
        index = index % df.shape[0]
        
        img_path = file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = targets[index]

        if self.process_target:
            target_pr = np.zeros(shape=(self.n_classes,))
            target_pr[int(target)] += 1.0
            target = target_pr
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target
        }





def prepare_loaders(df_train, df_valid, CONFIG, data_transforms, data_loader_base=ISICDatasetSampler, weight_adg=1, num_workers=10):
    
    train_dataset = data_loader_base(df_train, transforms=data_transforms["train"], weight_adg=weight_adg)
    valid_dataset = ISICDatasetSimple(df_valid, transforms=data_transforms["valid"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=num_workers, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader