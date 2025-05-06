# -*- coding: utf-8 -*-
'''
Train CIFAR10/CIFAR100 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47
modified to support CIFAR100 and looping through optimizers.
'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time
import traceback # For printing errors in loops

from models import * # Assuming this imports necessary model definitions like ResNets etc.
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT_LMA, ViT # Import both standard ViT and ViT_LMA
from models.convmixer import ConvMixer
from models.mobilevit import mobilevit_xxs # Assuming you have this file/class
from omegaconf import OmegaConf # Make sure OmegaConf is installed and imported

# Import summary libraries
try:
    from torchsummary import summary as pytorch_summary_call
    has_pytorch_summary = True
except ImportError:
    has_pytorch_summary = False
try:
    import torchinfo
    has_torchinfo = True
except ImportError:
    has_torchinfo = False
    print("torchinfo not found. Install with 'pip install torchinfo' to get model summaries.")

# Import custom optimizer if needed (adjust path if necessary)
try:
    from optim.sgd import DAG # Assuming DAG optimizer is in optim/sgd.py
except ImportError:
    print("Warning: Could not import DAG optimizer from optim.sgd")
    DAG = None # Define as None if import fails

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Optimizer Comparison')
# Remove opt and lr from main parser as they will be looped over
# Keep others for base configuration
# parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # Defined per optimizer later
# parser.add_argument('--opt', default="adam") # Looped over later
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint (NOTE: resume across different optimizers might be problematic)')
parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training.')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit', help='Model architecture to test (e.g., vit, vit_lma, res18)')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='100') # Reduced epochs for faster testing loop? Adjust as needed
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
# --dataset is removed, hardcoded to cifar100 later

base_args = parser.parse_args() # Parse the base arguments once

# --- Training Function ---
def run_training(model_name, optimizer_name, learning_rate, dataset_name, config_args):
    """Encapsulates the training loop for one configuration."""

    # --- Configuration for this run ---
    print(f"\n{'='*20} Starting Run {'='*20}")
    print(f"Model: {model_name}, Optimizer: {optimizer_name}, LR: {learning_rate}, Dataset: {dataset_name}")
    print(f"Base Args: {config_args}")
    print(f"{'='*55}")

    # --- SET SEED FOR REPRODUCIBILITY ---
    seed_value = 42
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # For multi-GPU
        # Optional: Forcing deterministic algorithms can hurt performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Set seed to {seed_value}")

    # Set run-specific variables based on inputs
    current_args = argparse.Namespace(**vars(config_args)) # Create a mutable copy
    current_args.net = model_name
    current_args.opt = optimizer_name
    current_args.lr = learning_rate
    current_args.dataset = dataset_name

    # --- WandB Setup (inside the function for separate runs) ---
    usewandb = not current_args.nowandb
    run_id_wandb = f"{model_name}_{optimizer_name}_lr{learning_rate}_{dataset_name}_p{current_args.patch}"
    if usewandb:
        import wandb
        try:
            wandb.init(
                project="cifar100-optimizer-comparison", # Changed project name
                name=run_id_wandb,
                config=vars(current_args), # Log effective config for this run
                reinit=True, # Allow re-initialization in the same process
                resume='allow' if current_args.resume else None # Allow resuming wandb run if script crashes
            )
        except Exception as e:
            print(f"Wandb initialization failed: {e}. Disabling wandb for this run.")
            usewandb = False


    # --- Reset State Variables ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0
    list_loss = []
    list_acc = []

    # --- Data Loading ---
    print(f'==> Preparing data for {dataset_name}...')
    if current_args.net=="vit_timm": # Use base_args here? Or should size depend on model_name? Assuming base_args.size
        size = 384
    else:
        size = int(current_args.size)

    if dataset_name == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100
        dataset_class = torchvision.datasets.CIFAR100
    else:
        print(f"Error: This script is configured for cifar100, but got {dataset_name}")
        return 0 # Indicate failure

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if current_args.noaug is False: # Check the flag correctly
        N = 2; M = 14;
        transform_train.transforms.insert(0, RandAugment(N, M))

    trainset = dataset_class(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(current_args.bs), shuffle=True, num_workers=8)
    testset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)


    # --- Model Building ---
    print(f'==> Building model {model_name}...')
    # Use current_args for model parameters
    try:
        if model_name=='res18':
            net = ResNet18(num_classes=num_classes)
        elif model_name=='vgg':
            net = VGG('VGG19', num_classes=num_classes)
        elif model_name=='res34':
            net = ResNet34(num_classes=num_classes)
        elif model_name=='res50':
            net = ResNet50(num_classes=num_classes)
        elif model_name=='res101':
            net = ResNet101(num_classes=num_classes)
        elif model_name=="convmixer":
            # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
            net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=num_classes)
        elif model_name=="mlpmixer":
            from models.mlpmixer import MLPMixer
            net = MLPMixer(
            image_size = 32,
            channels = 3,
            patch_size = args.patch,
            dim = 512,
            depth = 6,
            num_classes = num_classes
        )
        elif model_name=="vit_small":
            from models.vit_small import ViT
            net = ViT(
            image_size = size,
            patch_size = args.patch,
            num_classes = num_classes,
            dim = int(args.dimhead),
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        elif model_name=="vit_tiny":
            from models.vit_small import ViT
            net = ViT(
            image_size = size,
            patch_size = args.patch,
            num_classes = num_classes,
            dim = int(args.dimhead),
            depth = 4,
            heads = 6,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        elif model_name=="simplevit":
            from models.simplevit import SimpleViT
            net = SimpleViT(
            image_size = size,
            patch_size = args.patch,
            num_classes = num_classes,
            dim = int(args.dimhead),
            depth = 6,
            heads = 8,
            mlp_dim = 512
        )
        elif model_name=="vit":
            # ViT for cifar10/100
            net = ViT(
            image_size = size,
            patch_size = args.patch,
            num_classes = num_classes,
            dim = int(args.dimhead),
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        elif model_name=="vit_timm":
            import timm
            net = timm.create_model("vit_base_patch16_384", pretrained=True)
            net.head = nn.Linear(net.head.in_features, num_classes)
        elif model_name=="cait":
            from models.cait import CaiT
            net = CaiT(
            image_size = size,
            patch_size = args.patch,
            num_classes = num_classes,
            dim = int(args.dimhead),
            depth = 6,   # depth of transformer for patch to patch attention only
            cls_depth=2, # depth of cross attention of CLS tokens to patch
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05
        )
        elif model_name=="cait_small":
            from models.cait import CaiT
            net = CaiT(
            image_size = size,
            patch_size = args.patch,
            num_classes = num_classes,
            dim = int(args.dimhead),
            depth = 6,   # depth of transformer for patch to patch attention only
            cls_depth=2, # depth of cross attention of CLS tokens to patch
            heads = 6,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05
        )
        elif model_name=="swin":
            from models.swin import swin_t
            net = swin_t(window_size=args.patch,
                        num_classes=num_classes,
                        downscaling_factors=(2,2,2,1))
        elif model_name=="mobilevit":
            net = mobilevit_xxs(size, num_classes)
        elif model_name=="vit_lma":
            print("Using ViT_LMA model.")
            # Define LMA Config dynamically for this model
            latent_dim_d_new = int(current_args.dimhead) // 1 # Example
            num_heads_stacking = 8 # Example - TODO: Add arg if needed
            num_heads_latent = 8   # Example - TODO: Add arg if needed
            target_l_new = 64 # Example - TODO: Add arg if needed
            ff_latent_hidden = latent_dim_d_new * 4
            qkv_bias_lma = True

            lma_config_dict = {
                'd_new': latent_dim_d_new, 'num_heads_stacking': num_heads_stacking,
                'num_heads_latent': num_heads_latent, 'target_l_new': target_l_new,
                'ff_latent_hidden': ff_latent_hidden, 'qkv_bias': qkv_bias_lma,
            }
            lma_cfg = OmegaConf.create(lma_config_dict)
            print(f"  LMA Config: {OmegaConf.to_yaml(lma_cfg)}")

            net = ViT_LMA(
                image_size=size, patch_size=int(current_args.patch), num_classes=num_classes,
                dim=int(current_args.dimhead), depth=6, # Using fixed depth 6 for consistency
                lma_cfg=lma_cfg, pool='cls', dropout=0.1, emb_dropout=0.1
            )
        else:
            raise ValueError(f"'{model_name}' is not a valid model defined in the loop.")
    except Exception as e:
        print(f"!!!!!! ERROR Building Model {model_name} !!!!!!")
        print(e)
        traceback.print_exc()
        if usewandb and wandb.run is not None: wandb.finish(exit_code=1)
        return 0 # Indicate failure


    # --- Model Summary ---
    summary_printed_flag = False
    if has_pytorch_summary:
        print('\n==> Model Summary (using pytorch-summary)...')
        try:
            input_shape_pytorch_summary = (3, size, size)
            model_device_summary = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net.to(model_device_summary) # Move model temp to device for summary
            pytorch_summary_call(net, input_size=input_shape_pytorch_summary, device=str(model_device_summary))
            summary_printed_flag = True
        except Exception as e_pytorch_summary:
            print(f"pytorch-summary failed: {e_pytorch_summary}")
            print("Will attempt torchinfo if available, or basic count.")
    if not summary_printed_flag and has_torchinfo:
         print('\n==> Model Summary (using torchinfo)...')
         try:
             input_summary_size_ti = (1, 3, size, size)
             model_device_summary = torch.device("cuda" if torch.cuda.is_available() else "cpu")
             net.to(model_device_summary)
             torchinfo.summary(net, input_size=input_summary_size_ti, batch_dim=0,
                               col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                               col_width=16, row_settings=["var_names"], depth=5, device=model_device_summary, verbose=1)
             summary_printed_flag = True
         except Exception as e_ti:
             print(f"Could not print model summary using torchinfo: {e_ti}")
             print("Will fallback to basic parameter count.")
    if not summary_printed_flag:
        print("\n==> Model Summary (Libraries failed or unavailable, using basic count)...")
        try:
            total_params = sum(p.numel() for p in net.parameters())
            trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print(f"  Total Parameters: {total_params:,}")
            print(f"  Trainable Parameters: {trainable_params:,}")
        except Exception as e_manual:
            print(f"Could not count parameters manually: {e_manual}")

    # --- Move model to main training device and setup DP ---
    net = net.to(device)
    if 'cuda' in device and current_args.dp:
        print("Using Data Parallel")
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # --- Resume Logic (Uses unique checkpoint name) ---
    checkpoint_suffix = f"{model_name}_{optimizer_name}_lr{learning_rate}_{dataset_name}_p{current_args.patch}"
    checkpoint_path = f'./checkpoint/ckpt_{checkpoint_suffix}.t7'
    if current_args.resume and os.path.exists(checkpoint_path):
        print(f'==> Resuming from checkpoint {checkpoint_path}..')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            # Handle DP vs non-DP loading
            if isinstance(net, torch.nn.DataParallel):
                 if any(key.startswith('module.') for key in checkpoint['net'].keys()):
                      net.load_state_dict(checkpoint['net'])
                 else:
                      print("Loading non-DP checkpoint into DP model...")
                      net.module.load_state_dict(checkpoint['net'])
            else:
                 if any(key.startswith('module.') for key in checkpoint['net'].keys()):
                      print("Loading DP checkpoint into non-DP model...")
                      from collections import OrderedDict
                      new_state_dict = OrderedDict()
                      for k, v in checkpoint['net'].items():
                          name = k[7:] # remove `module.`
                          new_state_dict[name] = v
                      net.load_state_dict(new_state_dict)
                 else:
                      net.load_state_dict(checkpoint['net'])

            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch-1} with best_acc {best_acc:.2f}%")
            # NOTE: Optimizer state loading is skipped here because we are changing optimizers/LRs
            # You might want to load it if --resume is used *without* changing opt/LR, but the loop structure complicates this.
            # For simplicity, optimizer always starts fresh unless you add more complex resume logic.
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            best_acc = 0
            start_epoch = 0
    else:
         print(f"No checkpoint found at {checkpoint_path} or not resuming. Starting from scratch.")
         best_acc = 0
         start_epoch = 0

    # --- Optimizer and Scheduler ---
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "adam":
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == "DAG":
        if DAG is not None:
            optimizer = DAG(net.parameters(), lr=learning_rate, momentum=0.9) # Add other DAG params if needed
        else:
            print("ERROR: DAG optimizer not imported. Skipping run.")
            if usewandb and wandb.run is not None: wandb.finish(exit_code=1)
            return 0 # Indicate failure
    else:
        print(f"Error: Unknown optimizer '{optimizer_name}'")
        if usewandb and wandb.run is not None: wandb.finish(exit_code=1)
        return 0 # Indicate failure

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(current_args.n_epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=(not current_args.noamp)) # Use current_args

    # --- Define Train/Test Functions (Nested or Passed Args) ---
    # Define them here so they capture the current 'net', 'optimizer', 'criterion', etc.
    def train(epoch):
        # print('\nEpoch: %d' % epoch) # Moved print outside loop
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        return train_loss/(batch_idx+1)

    def test(epoch, current_best_acc): # Pass best_acc
        # global best_acc # Avoid global, use return value
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        acc = 100.*correct/total
        avg_test_loss = test_loss / (batch_idx + 1)

        is_best = acc > current_best_acc
        new_best_acc = max(acc, current_best_acc)

        if is_best:
            print('Saving best model...')
            state = {
                "net": net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "acc": acc,
                "epoch": epoch,
                # Store args used for this specific run in checkpoint
                "config_args": vars(current_args),
                "train_args": { # Explicitly log key train args for this run
                     "model_name": model_name,
                     "optimizer_name": optimizer_name,
                     "learning_rate": learning_rate,
                     "dataset_name": dataset_name
                }
            }
            if not os.path.isdir('checkpoint'): os.mkdir('checkpoint')
            torch.save(state, checkpoint_path) # Use unique path
            print(f"Best checkpoint saved to {checkpoint_path} (Acc: {acc:.2f}%)")
        else:
            print(f"Accuracy {acc:.2f}% did not improve from best {current_best_acc:.2f}%")

        # Log to file
        os.makedirs("log", exist_ok=True)
        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {avg_test_loss:.5f}, acc: {acc:.5f}'
        print(content)
        log_file = f'log/log_{checkpoint_suffix}.txt' # Unique log file
        with open(log_file, 'a') as appender:
            appender.write(content + "\n")

        return avg_test_loss, acc, new_best_acc # Return updated best_acc

    # --- Training Loop ---
    print(f"Starting training loop from epoch {start_epoch} for {current_args.n_epochs} epochs...")
    local_best_acc = best_acc # Use a local copy for this run's tracking
    for epoch in range(start_epoch, int(current_args.n_epochs)):
        print(f"\n--- {run_id_wandb} | Epoch: {epoch}/{current_args.n_epochs - 1} ---")
        start_time = time.time()
        trainloss = train(epoch)
        val_loss, acc, local_best_acc = test(epoch, local_best_acc) # Update local best
        scheduler.step()

        list_loss.append(val_loss)
        list_acc.append(acc)

        if usewandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': trainloss,
                'val_loss': val_loss,
                'val_acc': acc,
                'best_val_acc': local_best_acc, # Log best acc for this run
                'lr': optimizer.param_groups[0]["lr"],
                'epoch_time': time.time() - start_time
            })

        # Write out csv per epoch (can get large) or just at the end
        csv_file = f'log/log_{checkpoint_suffix}.csv' # Unique csv
        with open(csv_file, 'w') as f:
             writer = csv.writer(f, lineterminator='\n')
             writer.writerow(['Epoch'] + list(range(start_epoch, epoch + 1)))
             writer.writerow(['Val Loss'] + list_loss)
             writer.writerow(['Val Acc'] + list_acc)

    # --- Finish Run ---
    print(f"\n--- Finished Run: {run_id_wandb} ---")
    print(f"Best Validation Accuracy for this run: {local_best_acc:.2f}%")
    if usewandb:
        # Save the best checkpoint artifact to wandb
        if os.path.exists(checkpoint_path):
            try:
                # Check if best acc matches the saved checkpoint's acc
                # This requires loading the checkpoint again, might be slow
                # Alternatively, only save if the *last* epoch was the best
                # For simplicity, just save the best checkpoint found during the run
                 wandb.save(checkpoint_path, base_path=os.path.dirname(checkpoint_path))
                 print(f"Saved best checkpoint {checkpoint_path} to Wandb.")
            except Exception as e:
                 print(f"Failed to save checkpoint to Wandb: {e}")
        else:
             print("Best checkpoint not found for this run to save to Wandb.")
        wandb.finish()

    return local_best_acc # Return the best accuracy achieved in *this specific run*

# ===== Main Execution Logic =====
if __name__ == "__main__":

    # --- Define Experiment Grid ---
    # models_to_test = [base_args.net] # Test only the model passed via --net
    models_to_test = [
        'res18',
        'vgg',       # Corresponds to VGG19
        'res34',
        'res50',
        'res101',
        'convmixer',
        'mlpmixer',
        'vit_small', # From models.vit_small
        'vit_tiny',  # From models.vit_small
        'simplevit', # From models.simplevit
        'vit',       # From models.vit (standard ViT)
        'vit_timm',  # Pretrained timm ViT
        'cait',      # From models.cait
        'cait_small',# From models.cait
        'swin',      # From models.swin
        'mobilevit', # Assumes mobilevit_xxs specifically
        'vit_lma'    # Your custom ViT_LMA
    ]

    optimizers_config = {
        # Optimizer Name: Learning Rate
        'DAG': 7e-5,  # Adjust LR as needed
        'adam': 1e-4, # Adjust LR as needed
        'sgd': 1e-4   # Adjust LR as needed
    }
    dataset_to_use = 'cifar100' # Force CIFAR-100

    # --- Run Experiments ---
    results = {}
    for model_name in models_to_test:
        for opt_name, opt_lr in optimizers_config.items():
            run_key = f"{model_name}_{opt_name}_lr{opt_lr}"
            try:
                # Execute the training run
                best_acc_run = run_training(
                    model_name=model_name,
                    optimizer_name=opt_name,
                    learning_rate=opt_lr,
                    dataset_name=dataset_to_use,
                    config_args=base_args # Pass the original parsed args
                )
                results[run_key] = f"{best_acc_run:.2f}%"
            except KeyboardInterrupt:
                print("\n!!!!!! Training interrupted by user !!!!!!")
                # Optionally finish current wandb run if active
                if usewandb and wandb.run is not None:
                    wandb.finish(exit_code=130) # Indicate interruption
                exit() # Stop the entire script
            except Exception as e:
                print(f"\n!!!!!! ERROR during run {run_key} !!!!!!")
                print(e)
                traceback.print_exc()
                results[run_key] = "ERROR"
                # Ensure wandb run is closed if it was initialized for the failed run
                if usewandb and wandb.run is not None:
                    wandb.finish(exit_code=1)

    # --- Print Final Summary ---
    print("\n\n" + "="*25 + " FINAL RESULTS SUMMARY " + "="*25)
    for run_key, acc_str in results.items():
        print(f"{run_key}: {acc_str}")
    print("="*75)