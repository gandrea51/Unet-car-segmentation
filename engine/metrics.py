import torch

def IoU(preds: torch.Tensor, masks: torch.Tensor, num_classes: int=5) -> list:
    '''Intersection over union per classe'''

    preds, masks = preds.detach().cpu(), masks.detach().cpu()
    ious = []

    for c in range(num_classes):

        intersect = ((preds == c) & (masks == c)).sum().item()      # Intersezione
        union = ((preds == c) | (masks == c)).sum().item()          # Unione

        if union == 0:

            ious.append(float('nan'))           # Se una classe non è presente
        else:

            ious.append(intersect / union)      # IoU
    
    return ious

def dice_score(preds: torch.Tensor, masks: torch.Tensor, num_classes: int=5) -> list:
    '''Dice score per classe'''

    preds, masks = preds.detach().cpu(), masks.detach().cpu()
    dices = []

    for c in range(num_classes):

        intersect = ((preds == c) & (masks == c)).sum().item()              # Intersezione
        denom = (preds == c).sum().item() + (masks == c).sum().item()       # Denominatore: somma dei predetti e reali per la classe c

        if denom == 0:

            dices.append(float('nan'))              # Se una classe non è presente
        else:

            dices.append(2 * intersect / denom)     # Dice score
    
    return dices
