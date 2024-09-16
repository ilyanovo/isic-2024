import os
import gc
import time
import copy
import pandas as pd
import numpy as np
import torch

from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
from tqdm import tqdm
from torcheval.metrics.functional import binary_auroc
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold

from utils import set_seed, print_trainable_parameters
from datasets import prepare_loaders
from models import setup_model


def fetch_scheduler(optimizer, CONFIG):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler


def custom_metric_raw(y_hat, y_true):
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)
    
    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_hat])
    
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    
    return partial_auc


def criterion(outputs, targets):
    return nn.BCELoss()(outputs, targets)


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, CONFIG, criterion=criterion, metric_function=binary_auroc, 
                num_classes=1):
    
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    running_auroc  = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)
        loss = loss / CONFIG['n_accumulate']
            
        loss.backward()
    
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        if num_classes > 1:
            auroc = metric_function(input=outputs.squeeze(), target=torch.argmax(targets, axis=-1), num_classes=num_classes).item()
        else:
            auroc = metric_function(input=outputs.squeeze(), target=targets).item()
        running_loss += (loss.item() * batch_size)
        running_auroc  += (auroc * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_Auroc=epoch_auroc,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return epoch_loss, epoch_auroc


@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch, optimizer, criterion=criterion, use_custom_score=True, metric_function=binary_auroc,
                       num_classes=1, return_preds=False):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    running_auroc = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    predictions_all = []
    targets_all = []
    for step, data in bar:        
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)

        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)

        predictions_all.append(outputs.cpu().numpy())
        targets_all.append(targets.cpu().numpy())

        if num_classes > 1:
            auroc = metric_function(input=outputs.squeeze(), target=torch.argmax(targets, axis=-1), num_classes=num_classes).item()
        else:
            auroc = metric_function(input=outputs.squeeze(), target=targets).item()
        running_loss += (loss.item() * batch_size)
        running_auroc  += (auroc * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Auroc=epoch_auroc, LR=optimizer.param_groups[0]['lr'])   
    
    gc.collect()

    targets_all = np.concatenate(targets_all)
    predictions_all = np.concatenate(predictions_all)
    
    epoch_custom_metric = None
    if use_custom_score:
        epoch_custom_metric = custom_metric_raw(predictions_all, targets_all)

    if return_preds:
        return epoch_loss, epoch_auroc, epoch_custom_metric, predictions_all, targets_all
    return epoch_loss, epoch_auroc, epoch_custom_metric


def get_nth_test_step(epoch):
    if epoch < 6:
        return 5
    if epoch < 10:
        return 4
    if epoch < 15:
        return 3
    if epoch < 20:
        return 2
    return 1

def run_training(
        train_loader, valid_loader, model, optimizer, scheduler, device, num_epochs, CONFIG, 
        model_folder=None, model_name="", seed=42, tolerance_max=15, criterion=criterion, 
        test_every_nth_step=get_nth_test_step, 
        num_classes=1, best_epoch_score_def=-np.inf):
    set_seed(seed)
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_score = best_epoch_score_def
    history = defaultdict(list)
    tolerance = 0

    for epoch in range(1, num_epochs + 1): 
        test_every_nth_step_upd = test_every_nth_step(epoch)
       
        if tolerance > tolerance_max:
            break
        gc.collect()
        train_epoch_loss, train_epoch_auroc = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CONFIG['device'], 
                                           CONFIG=CONFIG, epoch=epoch, criterion=criterion,
                                        metric_function = binary_auroc, num_classes=num_classes)

        if epoch % test_every_nth_step_upd == 0:
            val_epoch_loss, val_epoch_auroc, val_epoch_custom_metric = valid_one_epoch(
                model, valid_loader, device=CONFIG['device'], epoch=epoch, optimizer=optimizer, criterion=criterion,
                metric_function=binary_auroc, num_classes=num_classes)
        
            history['Train Loss'].append(train_epoch_loss)
            history['Valid Loss'].append(val_epoch_loss)
            history['Train AUROC'].append(train_epoch_auroc)
            history['Valid AUROC'].append(val_epoch_auroc)
            history['Valid Kaggle metric'].append(val_epoch_custom_metric)
            history['lr'].append( scheduler.get_lr()[0] )
            
            if best_epoch_score <= val_epoch_custom_metric:
                tolerance = 0
                print(f"Validation AUROC Improved ({best_epoch_score} ---> {val_epoch_custom_metric})")
                best_epoch_score = val_epoch_custom_metric
                best_model_wts = copy.deepcopy(model.state_dict())
                if model_folder is not None:
                    torch.save(model.state_dict(), os.path.join(model_folder, model_name))
            else:
                tolerance += 1
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best AUROC: {:.4f}".format(best_epoch_score))    
    model.load_state_dict(best_model_wts)
    return model, history


def get_metrics(drop_path_rate, drop_rate, models_folder, model_maker, CONFIG):
    tsp = StratifiedGroupKFold(5, shuffle=True, random_state=CONFIG['seed'])
    results_list = []
    fold_df_valid_list = []
    for fold_n, (train_index, val_index) in enumerate(tsp.split(df_train, y=df_train.target, groups=df_train[CONFIG["group_col"]])):
        fold_df_train = df_train.iloc[train_index].reset_index(drop=True)
        fold_df_valid = df_train.iloc[val_index].reset_index(drop=True)
        set_seed(CONFIG['seed'])
        model = setup_model(model_name, drop_path_rate=drop_path_rate, drop_rate=drop_rate, model_maker=model_maker)
        print_trainable_parameters(model)

        train_loader, valid_loader = prepare_loaders(fold_df_train, fold_df_valid)
    
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
                           weight_decay=CONFIG['weight_decay'])
        scheduler = fetch_scheduler(optimizer)
    
        model, history = run_training(
            train_loader, valid_loader,
            model, optimizer, scheduler,
            device=CONFIG['device'],
            num_epochs=CONFIG['epochs'],
            CONFIG=CONFIG,
            tolerance_max=20,
            test_every_nth_step=lambda x: 5,
            seed=CONFIG['seed'])
        torch.save(model.state_dict(), os.path.join(models_folder, f"model__{fold_n}"))
        results_list.append(np.max(history['Valid Kaggle metric']))

        val_epoch_loss, val_epoch_auroc, val_epoch_custom_metric, tmp_predictions_all, tmp_targets_all = valid_one_epoch(
            model, 
            valid_loader, 
            device=CONFIG['device'], 
            epoch=1, 
            optimizer=optimizer, 
            criterion=criterion, 
            use_custom_score=True,
            metric_function=binary_auroc, 
            num_classes=1,
            return_preds=True)

        fold_df_valid['tmp_targets_all'] = tmp_targets_all
        fold_df_valid['tmp_predictions_all'] = tmp_predictions_all
        fold_df_valid['fold_n'] = fold_n
        fold_df_valid_list.append(fold_df_valid)
    fold_df_valid_list = pd.concat(fold_df_valid_list).reset_index(drop=True)
    return results_list, fold_df_valid_list