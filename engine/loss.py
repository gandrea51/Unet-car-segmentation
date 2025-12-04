import torch
import torch.nn as nn
from kornia.losses import FocalLoss
import torch.nn.functional as F

def dice_loss(logits, targets, smooth=1e-6):
    '''Dice loss per segmentazione multiclasse'''

    num_classes = logits.shape[1]
    
    # Conversione logits in probablità
    probs = F.softmax(logits, dim=1)
    targets_ohe = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    # Flatten (H * W) per il calcolo di Dice
    probs = probs.reshape(probs.size(0), num_classes, -1)
    targets_ohe = targets_ohe.reshape(targets_ohe.size(0), num_classes, -1)

    # Formula del Dice
    intersection = (probs * targets_ohe).sum(-1)
    union = (probs * probs).sum(-1) + (targets_ohe * targets_ohe).sum(-1)
    dice = (2 * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()

class FocalLoss(nn.Module):

    def __init__(self, gamma: float = 2.0, alpha = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma              # Fattore che pesa di più gli esempi difficili
        self.alpha = alpha              # Pesi per classe
        self.reduction = reduction      # Tipo di riduzione

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)        # Probablità del target corretto
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Se alpha è impostato:
        if self.alpha is not None:
            num_classes = inputs.shape[1]
            targets_ohe = F.one_hot(targets, num_classes=num_classes).float()
            
            if isinstance(self.alpha, torch.Tensor):

                # Se alpha è un tensore
                alpha_t = self.alpha.to(inputs.device)
                alpha_factor = (alpha_t * targets_ohe).sum(dim=-1)
            else:

                # Se alpha è uno scalare
                alpha_factor = self.alpha
                
            focal_loss = alpha_factor * focal_loss
        
        # Riduzione finale
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def combine_loss(logits, targets):
    '''Combinazione di Cross Entropy e Focal loss'''

    ce_loss = nn.CrossEntropyLoss()
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    
    return ce_loss(logits, targets) + focal_loss(logits, targets) 