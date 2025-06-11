##Train model and predict with UNet-Pocketnet
import argparse
import os
import sys
import time
import warnings
import pandas as pd
import gc
import numpy as np
import logging
import nibabel as nib

import torch
from torch.utils.data import Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from monai.apps import DecathlonDataset
from monai.data import Dataset, decollate_batch, CSVDataset, CacheDataset, partition_dataset, ThreadDataLoader, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandCropByPosNegLabeld,
    ResizeWithPadOrCropd,
    Spacingd,
    ToDeviced,
    EnsureTyped,
)
from monai.utils import set_determinism
from sklearn. model_selection import train_test_split
from sklearn.model_selection import KFold

import PN
from PN import unetPN
import json

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MASTER_ADDR"] = "localhost"

def main_worker(args):
    # disable logging for processes except 0 on every node
    if int(os.environ["LOCAL_RANK"]) != 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f
    
    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")

    total_start = time.time()
    
    #Create output directories
    outpath = args.folderpath
    os.makedirs(outpath, exist_ok=True)
    folderNames = ['index', 'model_weights', 'log', 'pred']
    for folder in folderNames:
        full_path = os.path.join(outpath,folder)
        os.makedirs(os.path.join(outpath,folder), exist_ok=True)

    #Read parameters
    parameter = os.path.join(outpath,'input.json')
    with open(parameter, 'r') as f:
        paths = json.load(f)
        INchannels = paths["input_channels"]
        OUTchannels = paths["output_channels"]
        max_epochs = int(paths["num_of_epochs"])
        Num_of_patches = int(paths["num_of_patches"])
        roi = paths["roi_size"]
        modal = paths["modality"]
        input_sequence = [f'{item}' for item in modal]
        input_label = paths["mask"]
        print(input_label)
        
    #Read data
    datafile = os.path.join(outpath,'dataset.csv')
    dataset = CSVDataset(
            src=[datafile], 
            col_groups = {"image": input_sequence, "label": [input_label]},
            )
    
    #Split Dataset as hold and test
    splits = KFold(n_splits = 5, shuffle = True, random_state = 2)
    for fold, (hold_idx, test_idx) in enumerate(splits.split(np.arange(len(dataset.data)))):   
        print(str(fold))

        te = pd.DataFrame(test_idx)
        out_testIdx = os.path.join(outpath,'index', 'test_idx_fold_' + str(fold) + '.csv')
        te.to_csv(out_testIdx, index=False)   

        train_idx, val_idx = train_test_split(hold_idx, test_size = 0.1)

        tr = pd.DataFrame(train_idx)
        out_trainIdx = os.path.join(outpath,'index', 'train_idx_fold_' + str(fold) + '.csv')
        tr.to_csv(out_trainIdx, index=False)

        val = pd.DataFrame(val_idx)
        out_valIdx = os.path.join(outpath,'index', 'val_idx_fold_' + str(fold) + '.csv')
        val.to_csv(out_valIdx, index=False)

        #create a training data 
        train_set = Subset(dataset.data, train_idx)
        val_set = Subset(dataset.data, val_idx)

        train_transforms = Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                EnsureTyped(keys=["image", "label"]),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size= roi,
                    pos=1,
                    neg=1,
                    num_samples= Num_of_patches,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True
                    ),
                ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=roi, mode='constant'),
            ]
        )
    
        # create a training data loader
        #train_ds = Dataset(data=train_set, transform=train_transforms)
        train_data = CacheDataset(data=train_set, transform=train_transforms, num_workers=4, cache_rate=1.0)
    
        train_ds = partition_dataset(
                    data=train_data,
                    num_partitions=dist.get_world_size(),
                    shuffle=True,
                    seed=0,
                    drop_last=False,
                    even_divisible=True,
                )[dist.get_rank()]
    
        # ThreadDataLoader can be faster if no IO operations when caching all the data in memory
        train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=4, shuffle=True)
    
    
        # validation transforms and dataset
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
    
        # create a training data loader
        #val_ds = Dataset(data=val_set, transform=train_transforms)
        val_data = CacheDataset(data=val_set, transform=val_transforms, num_workers=4, cache_rate=1.0)
    
        val_ds = partition_dataset(
                    data=val_data,
                    num_partitions=dist.get_world_size(),
                    shuffle=False,
                    seed=0,
                    drop_last=False,
                    even_divisible=False,
                )[dist.get_rank()]
    
        # ThreadDataLoader can be faster if no IO operations when caching all the data in memory
        val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=4, shuffle=False)
    
        # create network, loss function and optimizer
        device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        torch.cuda.set_device(device)
    
        # use amp to accelerate training
        scaler = torch.amp.GradScaler("cuda")
        torch.backends.cudnn.benchmark = True
    
        model = unetPN(channels = 32, input_channels = int(INchannels), output_channels = int(OUTchannels)).to(device)
        model = DistributedDataParallel(model, device_ids=[device])
    
        #to_onehot_y=False, for one channel prediction
    
        loss_function = DiceLoss(to_onehot_y=True, softmax=True) 
    
        # Create optimizer based on the argument
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), args.lr)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=0.00004)
    
        #optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=0.00004)
    
        dice_metric = DiceMetric(include_background=False, reduction="mean")
    
        val_interval = 10
        best_metric = -1
        best_metric_epoch = -1
        epoch_num = []
        epoch_loss_values = []
        epoch_times = []
        metric_values = []
        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=int(OUTchannels))]) 
        post_label = Compose([AsDiscrete(to_onehot=int(OUTchannels))])
    
        for epoch in range(max_epochs):
            epoch_start = time.time()
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
            epoch_loss /= step
            epoch_num.append(epoch + 1)
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
            ##Validation
            if (epoch + 1) % val_interval == 0:
                model.eval()
    
                with torch.no_grad():
                    for val_data in val_loader:
                        with torch.amp.autocast(device_type='cuda'):
                            val_inputs, val_labels = (
                                val_data["image"].to(device),
                                val_data["label"].to(device),
                            )
                            sw_batch_size = 4
                            val_outputs = sliding_window_inference(val_inputs, roi, sw_batch_size, model)
    
                        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                        val_labels = [post_label(i) for i in decollate_batch(val_labels)]
    
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=val_labels)
    
                    # aggregate the final mean dice result
                    metric = dice_metric.aggregate().item()
    
                    # reset the status for next validation round
                    dice_metric.reset()
                    metric_values.append(metric)
    
                    if metric > best_metric:
                        best_metric = metric
    
                        best_metric_epoch = epoch + 1
                        if dist.get_rank() == 0:
                            torch.save(model.state_dict(), os.path.join(outpath, "model_weights", "best_metric_model_fold_" + str(fold) +  ".pth"))
                            print("saved new best metric model")
                        print(
                            f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                            f"\nbest mean dice: {best_metric:.4f} "
                            f"at epoch: {best_metric_epoch}"
                            )
    
            print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
            epoch_times.append(f"{(time.time() - epoch_start):.4f}")

            #Write the loss values to a CSV file
            loss_values = os.path.join(outpath,'log', 'loss_' + str(fold) +'.csv')
            fieldnames = ['epoch','train_loss', 'epoch_times']
            df = pd.DataFrame(zip(epoch_num, epoch_loss_values, epoch_times), columns=fieldnames) 
            df.to_csv(loss_values, index=False)
    
    
        ##Start predicting 
        test_set = Subset(dataset.data, test_idx)
    
        test_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
    
        test_ds = CacheDataset(data=test_set, transform=test_transforms, num_workers=4, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)
    
        map_location = (f"cuda:{os.environ['LOCAL_RANK']}")
        model.load_state_dict(torch.load(os.path.join(outpath, 'model_weights', "best_metric_model_fold_" + str(fold) + ".pth"), map_location=map_location))
    
        pred_values = []
        pid_values = []
        pred_metric = DiceMetric(include_background=False, reduction="mean")
    
        with torch.no_grad():
            for test_data in test_loader:
                test_inputs, test_labels = test_data["image"].to(device), test_data["label"].to(device)
    
                original_affine = test_labels.meta["affine"][0].numpy()
                _, _, h, w, d = test_labels.shape
                target_shape = (h, w, d)
    
                img_name = test_inputs.meta["filename_or_obj"][0].split("/")[-1]
                pid = img_name.split(".")[0]
    
                roi_size= roi
                sw_batch_size=4 
    
                test_outputs = sliding_window_inference(test_inputs, roi, sw_batch_size, model)
    
                test_pred = [post_pred(i) for i in decollate_batch(test_outputs)]
                pred_metric(y_pred=test_pred, y=test_labels)
    
                test_outputs = torch.softmax(test_outputs, 1).cpu().numpy()
                test_outputs = np.argmax(test_outputs, axis=1).astype(np.uint8)[0]
    
                outFile = 'pred_' + pid + '.nii.gz'
                nib.save(nib.Nifti1Image(test_outputs.astype(np.uint8), original_affine), os.path.join(outpath,'pred', outFile))
    
                pred = pred_metric.aggregate().item()
                print(f"{pred:.4f}")
                pred_values.append(f"{pred:.4f}")
                pid_values.append(pid)
                pred_metric.reset()
    
            #Write the loss values to a CSV file
            diceFile = os.path.join(outpath,'log', 'dice_' + str(fold) +'.csv')
            fieldnames = ['pid','dice'] 
            df = pd.DataFrame(zip(pid_values, pred_values), columns=fieldnames) 
            df.to_csv(diceFile, index=False)
    
        print('clearing')
        gc.collect()
        torch.cuda.empty_cache()

    dist.destroy_process_group()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training.")
    parser.add_argument("--nproc_per_node")
    parser.add_argument("--folderpath")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use (default: Adam)')
    args = parser.parse_args()

    if args.seed is not None:
        set_determinism(seed=args.seed)
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    main_worker(args=args)
