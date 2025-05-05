# -*- coding: utf-8 -*-
'''
Train CIFAR10/CIFAR100 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47
modified to support CIFAR100
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

from models import * # Assuming this imports necessary model definitions like ResNets etc.
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT_LMA, ViT # Import both standard ViT and ViT_LMA
from models.convmixer import ConvMixer
from models.mobilevit import mobilevit_xxs # Assuming you have this file/class
from omegaconf import OmegaConf # Make sure OmegaConf is installed and imported
from torchsummary import summary as pytorch_summary_call
from optim.sgd import DAG

# Import torchinfo
try:
    import torchinfo
    has_torchinfo = True
except ImportError:
    has_torchinfo = False
    print("torchinfo not found. Install with 'pip install torchinfo' to get model summaries.")

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to use (cifar10 or cifar100)')

args = parser.parse_args()

# take in args
usewandb = ~args.nowandb
if usewandb:
    import wandb
    watermark = "{}_lr{}_{}".format(args.net, args.lr, args.dataset)
    wandb.init(project="cifar-challenge", # Replace with your project name if needed
            name=watermark)
    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize

# Set up normalization based on the dataset
if args.dataset == 'cifar10':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    num_classes = 10
    dataset_class = torchvision.datasets.CIFAR10
elif args.dataset == 'cifar100':
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    num_classes = 100
    dataset_class = torchvision.datasets.CIFAR100
else:
    raise ValueError("Dataset must be either 'cifar10' or 'cifar100'")

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

# Add RandAugment with N, M(hyperparameter)
if aug:
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
trainset = dataset_class(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

# Set up class names based on the dataset
if args.dataset == 'cifar10':
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
else:
    # CIFAR100 has 100 classes, so we don't list them all here
    classes = None

# Model factory..
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='res18':
    net = ResNet18(num_classes=num_classes)
elif args.net=='vgg':
    net = VGG('VGG19', num_classes=num_classes)
elif args.net=='res34':
    net = ResNet34(num_classes=num_classes)
elif args.net=='res50':
    net = ResNet50(num_classes=num_classes)
elif args.net=='res101':
    net = ResNet101(num_classes=num_classes)
elif args.net=="convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=num_classes)
elif args.net=="mlpmixer":
    from models.mlpmixer import MLPMixer
    net = MLPMixer(
    image_size = 32,
    channels = 3,
    patch_size = args.patch,
    dim = 512,
    depth = 6,
    num_classes = num_classes
)
elif args.net=="vit_small":
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
elif args.net=="vit_tiny":
    from models.vit_small import ViT # Assuming vit_small.py contains the ViT class
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
elif args.net=="simplevit":
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
elif args.net=="vit":
    # ViT for cifar10/100
    net = ViT( # Use the standard ViT imported from models.vit
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
# Note: Duplicate elif args.net=="vit" was removed here. Keep only one.
elif args.net=="vit_lma": # Changed name to distinguish
    # ViT_LMA for cifar10/100
    print("Using ViT_LMA model.") # Added print statement
    # --- Define LMA Configuration ---
    try:
        # Example values, replace with your actual config source (e.g., add more args)
        latent_dim_d_new = int(args.dimhead) // 2 # Example: Half of original dim
        num_heads_stacking = 32 # Example hyperparameter - Consider adding an arg for this
        num_heads_latent = 8   # Example hyperparameter (must divide latent_dim_d_new) - Consider adding arg
        target_l_new = 16 # Example: Let LMA calculate based on S=num_patches+1
        ff_latent_hidden = latent_dim_d_new * 4 # Example: Standard expansion in latent space
        qkv_bias_lma = True                    # Example setting
        
        # # Example values, replace with your actual config source (e.g., add more args)
        # latent_dim_d_new = int(args.dimhead) // 2 # Example: Half of original dim
        # num_heads_stacking = 8 # Example hyperparameter - Consider adding an arg for this
        # num_heads_latent = 4   # Example hyperparameter (must divide latent_dim_d_new) - Consider adding arg
        # target_l_new = 65 # Example: Let LMA calculate based on S=num_patches+1
        # ff_latent_hidden = latent_dim_d_new * 4 # Example: Standard expansion in latent space
        # qkv_bias_lma = False      

        lma_config_dict = {
            'd_new': latent_dim_d_new,
            'num_heads_stacking': num_heads_stacking,
            'num_heads_latent': num_heads_latent,
            'target_l_new': target_l_new,
            'ff_latent_hidden': ff_latent_hidden,
            'qkv_bias': qkv_bias_lma,
            # Note: static_seq_len and dropout_prob are set inside ViT_LMA.__init__
        }
        lma_cfg = OmegaConf.create(lma_config_dict)
        print(f"  LMA Config: {OmegaConf.to_yaml(lma_cfg)}") # Print LMA config for verification

        # --- Instantiate ViT_LMA ---
        net = ViT_LMA(
            image_size = size,
            patch_size = args.patch,
            num_classes = num_classes,
            dim = int(args.dimhead), # Original embedding dimension (d_0)
            depth = 6,               # Number of *LatentLayer* blocks
            lma_cfg = lma_cfg,       # Pass the LMA configuration
            pool = 'cls',            # Or 'mean'
            dropout = 0.1,           # Main dropout (also used inside LMA)
            emb_dropout = 0.1
        )
    except Exception as e:
        print(f"Error creating ViT_LMA model: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        exit() # Exit if model creation fails

elif args.net=="vit_timm":
    import timm
    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, num_classes)
elif args.net=="cait":
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
elif args.net=="cait_small":
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
elif args.net=="swin":
    from models.swin import swin_t
    net = swin_t(window_size=args.patch,
                num_classes=num_classes,
                downscaling_factors=(2,2,2,1))
elif args.net=="mobilevit":
    net = mobilevit_xxs(size, num_classes)
else:
    raise ValueError(f"'{args.net}' is not a valid model")


print("Attempting manual forward pass...") # Sanity Check
try:
    input_summary_size = (1, 3, size, size) # Use batch size 1 for summary
    # Ensure model and dummy input are on the same device BEFORE the forward pass
    current_device = next(net.parameters()).device # Get device model is actually on
    print(f"  Model device: {current_device}")
    dummy_input = torch.randn(input_summary_size).to(current_device)
    print(f"  Dummy input shape: {dummy_input.shape}, device: {dummy_input.device}")
    net.eval() # Set to eval mode for consistency
    with torch.no_grad():
         output = net(dummy_input)
    print(f"Manual forward pass successful. Output shape: {output.shape}")
    net.train() # Set back to train mode if needed
except Exception as e_manual:
    print(f"Manual forward pass FAILED: {e_manual}")
    import traceback
    traceback.print_exc()
    
# -------- MODEL SUMMARY --------

try:
    from torchsummary import summary as pytorch_summary_call
    has_pytorch_summary = True
except ImportError:
    has_pytorch_summary = False
    
# First, try pytorch-summary if available
if has_pytorch_summary:
    print('\n==> Model Summary (using pytorch-summary)...')
    try:
        # pytorch-summary expects (channels, height, width)
        input_shape_pytorch_summary = (3, size, size)

        # Determine device model is on
        if next(net.parameters(), None) is not None:
            current_device = next(net.parameters()).device
        else:
            current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure model is on the correct device before summary
        net.to(current_device)

        # Call pytorch-summary
        pytorch_summary_call(net, input_size=input_shape_pytorch_summary, device=str(current_device))
        summary_printed = True # Flag that we succeeded

    except Exception as e_pytorch_summary:
        print(f"pytorch-summary failed: {e_pytorch_summary}")
        print("Will attempt torchinfo if available, or basic count.")
        summary_printed = False # Flag that we failed

if has_torchinfo:
    print('\n==> Model Summary (using torchinfo - simplified)...')
    try:
        input_summary_size_ti = (1, 3, size, size)
        # VERY basic call
        torchinfo.summary(net, input_size=input_summary_size_ti, verbose=1)
    except Exception as e_ti:
        print(f"Simplified torchinfo call failed: {e_ti}")
        
# If pytorch-summary wasn't available OR it failed, try torchinfo
elif has_torchinfo: # Use elif here
    print('\n==> Model Summary (using torchinfo)...')
    try:
        # Define the input size for torchinfo
        input_summary_size_ti = (1, 3, size, size) # (batch, channels, height, width)

        # Determine device model is on (redundant if pytorch-summary was tried, but safe)
        if next(net.parameters(), None) is not None:
            current_device = next(net.parameters()).device
        else:
            current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure model is on the correct device before summary
        net.to(current_device)

        # Call torchinfo
        torchinfo.summary(net,
                          input_size=input_summary_size_ti,
                          batch_dim=0,
                          col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                          col_width=16,
                          row_settings=["var_names"],
                          depth=5,
                          device=current_device,
                          verbose=1)
        summary_printed = True # Flag that we succeeded

    except Exception as e_ti:
        print(f"Could not print model summary using torchinfo: {e_ti}")
        print("Will fallback to basic parameter count.")
        summary_printed = False # Flag that we failed

# Fallback if neither library is available or both failed
else:
     summary_printed = False # Neither library was available
     # Optional: you could add a check here: if not summary_printed: ...

if not summary_printed: # If no summary was printed by either library
    print("\n==> Model Summary (Libraries failed or unavailable, using basic count)...")
    try:
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
    except Exception as e_manual:
        print(f"Could not even count parameters manually: {e_manual}")

# -------------------------------------------

net = net.to(device) # Use the main script device variable 'device' here

# For Multi-GPU
if 'cuda' in device:
    print(f"Using device: {device}")
    if args.dp:
        print("Using Data Parallel")
        net = torch.nn.DataParallel(net) # make parallel AFTER summary
        cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint_path = './checkpoint/{}-{}-{}-ckpt.t7'.format(args.net, args.dataset, args.patch) # Updated path format
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device) # Load to correct device
        # Handle DataParallel state_dict keys
        if isinstance(net, torch.nn.DataParallel):
             # If current net is DP, load directly if checkpoint was saved from DP
             # If checkpoint was NOT from DP, need to load into net.module
             try:
                 net.load_state_dict(checkpoint['net'])
             except RuntimeError: # Likely mismatch (DP vs non-DP)
                 print("Loading non-DP checkpoint into DP model...")
                 net.module.load_state_dict(checkpoint['net'])
        else:
             # If current net is not DP
             # If checkpoint WAS from DP, need to load from state_dict['module']
             # If checkpoint was NOT from DP, load directly
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

        # Load optimizer, scaler, etc.
        if 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                 print(f"Warning: Could not load optimizer state: {e}")
        if 'scaler' in checkpoint and use_amp:
             try:
                 scaler.load_state_dict(checkpoint['scaler'])
             except Exception as e:
                 print(f"Warning: Could not load AMP scaler state: {e}")
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1 # Start from next epoch
        print(f"Resumed from epoch {start_epoch-1} with best_acc {best_acc:.2f}%")
    else:
        print(f"Checkpoint path not found: {checkpoint_path}")
        print("Starting training from scratch.")


# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) # Added common SGD params
if args.opt == "DAG":
    optimizer = DAG(net.parameters(), lr=args.lr, momentum=0.9)

# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
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

##### Validation
def test(epoch):
    global best_acc
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

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            # Save module state dict if using DataParallel
            "net": net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        # Consistent checkpoint naming
        save_path = './checkpoint/{}-{}-{}-ckpt.t7'.format(args.net, args.dataset, args.patch)
        torch.save(state, save_path)
        print(f"Checkpoint saved to {save_path}")
        best_acc = acc
    else:
         print(f"Accuracy {acc:.2f}% did not improve from best {best_acc:.2f}%")

    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss/(batch_idx+1):.5f}, acc: {acc:.5f}' # Corrected val_loss calculation
    print(content)
    log_file = f'log/log_{args.net}_{args.dataset}_patch{args.patch}.txt'
    with open(log_file, 'a') as appender:
        appender.write(content + "\n")
    return test_loss/(batch_idx+1), acc # Return average loss


list_loss = []
list_acc = []

if usewandb:
    wandb.watch(net)

# It's often better to move the net to cuda *before* defining the optimizer
# net = net.to(device) # Moved earlier, before summary and DP wrapping

# Load checkpoint needs to happen *after* model definition and *before* DP/moving to device sometimes
# The updated resume logic handles device mapping and DP state dicts better.

for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)

    scheduler.step() # CosineAnnealingLR typically steps each epoch without epoch arg after PyTorch 1.1.0

    list_loss.append(val_loss)
    list_acc.append(acc)

    # Log training..
    if usewandb:
        wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time()-start})

    # Write out csv..
    csv_file = f'log/log_{args.net}_{args.dataset}_patch{args.patch}.csv'
    with open(csv_file, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['Epoch'] + list(range(start_epoch, epoch + 1))) # Add header
        writer.writerow(['Val Loss'] + list_loss)
        writer.writerow(['Val Acc'] + list_acc)
    print(f"Best validation accuracy: {best_acc:.2f}%") # Print best accuracy at the end of epoch

# writeout wandb
if usewandb:
    # Saving the best model to wandb might be more useful than the final state
    best_ckpt_path = './checkpoint/{}-{}-{}-ckpt.t7'.format(args.net, args.dataset, args.patch)
    if os.path.exists(best_ckpt_path):
         wandb.save(best_ckpt_path)
    else:
         print("Best checkpoint not found for wandb save.")
    # wandb.save("wandb_{}_{}.h5".format(args.net, args.dataset)) # .h5 is unusual for pytorch

print(f"Finished training. Best validation accuracy: {best_acc:.2f}%")