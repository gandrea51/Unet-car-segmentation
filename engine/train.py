import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from engine.metrics import IoU, dice_score

def training(model: torch.nn.Module, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: torch.device):
    '''Esecuzione di una epoca di training'''

    # Modalità training
    model.train()
    total_loss = 0
    iou, dice = [], []

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()                       # Azzero i gradienti
        outputs = model(images)                     # Forward pass
        
        loss = criterion(outputs, masks)            # Calcolo la loss
        loss.backward()                             # Backpropagation
        optimizer.step()

        total_loss += loss.item()                   # Accumulo della loss

        preds = outputs.argmax(1)                   # Predizione delle classi
        
        # Calcolo delle metriche
        iou.append(np.nanmean(IoU(preds, masks)))
        dice.append(np.nanmean(dice_score(preds, masks)))
    
    return total_loss / len(loader), np.nanmean(iou), np.nanmean(dice)

def validating(model: torch.nn.Module, loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: torch.device):
    '''Esecuzione di una epoca di validazione'''

    # Modalità valutazione
    model.eval()
    total_loss = 0
    iou, dice = [], []

    # Disabilito il calcolo dei gradienti
    with torch.no_grad():

        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)                 # Forward pass

            loss = criterion(outputs, masks)        # Calcolo la loss
            total_loss += loss.item()

            preds = outputs.argmax(1)               # Predizione delle classi
            
            # Calcolo delle metriche
            iou.append(np.nanmean(IoU(preds, masks)))
            dice.append(np.nanmean(dice_score(preds, masks)))

    return total_loss / len(loader), np.nanmean(iou), np.nanmean(dice)

def freeze_layer(model, num_blocks):
    '''Freeze dei blocchi iniziali'''

    # Freeze dello STEM
    for p in model.stem.parameters():
        p.requires_grad = False
    print(f'Stem freezzato.')

    # Freeze progressivo dei primi blocchi
    for i in range(1, num_blocks + 1):
        block_name = f'block{i}'

        if hasattr(model, block_name):
            block = getattr(model, block_name)

            for p in block.parameters():
                p.requires_grad = False
            print(f'Blocco {block_name} freezzato.')
        else:

            print(f'Attenzione. Blocco {block_name} non trovato.')
            break

def freeze_all(model, num_blocks=4):
    '''Freeze iniziale'''

    # Freeze dello STEM
    for p in model.stem.parameters():
        p.requires_grad = False
    print("Stem freezzato.")

    for i in range(1, num_blocks + 1):
        block_name = f'block{i}'

        if hasattr(model, block_name):
            block = getattr(model, block_name)
            
            for p in block.parameters():
                p.requires_grad = False
            print(f'Blocco {block_name} freezzato.')

def unfreeze_step(model, current_epoch, start_epoch=5, step=1):
    '''Unfreeze progressivo a ritroso a partire da current_epoch'''

    # Indice dei blocchi
    blocks_to_freeze = [1,2,3,4]

    # Numero di blocchi da scongelare in base all'epoca
    num_unfreeze = max(0, (current_epoch - start_epoch + 1) // step)
    num_unfreeze = min(num_unfreeze, 4)

    # Freeze dei blocchi ancora bloccati
    for i in range(4 - num_unfreeze):
        block_name = f'block{blocks_to_freeze[i]}'

        if hasattr(model, block_name):
            block = getattr(model, block_name)

            for p in block.parameters():
                p.requires_grad = False

    # Unfreeze progressivo
    for i in range(4 - num_unfreeze, 4):
        block_name = f'block{blocks_to_freeze[i]}'

        if hasattr(model, block_name):
            block = getattr(model, block_name)

            for p in block.parameters():
                p.requires_grad = True
            print(f'Blocco {block_name} scongelato.')

    # Unfreeze dello stem
    if num_unfreeze >= 4:
        for p in model.stem.parameters():
            p.requires_grad = True
        print(f'Stem scongelato.')
