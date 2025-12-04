import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_transform():
    '''Trasformazioni per training e validation/test set'''

    # Training: Data Augmentation
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(0.05, 0.15, 10, border_mode=cv2.BORDER_CONSTANT, p=0.5),     # Traslazione, Scaling e Rotazione
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),                                    # Variazioni di luminosità e contrasto
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.RandomShadow(p=0.2),                                          # Ombre casuali
        A.RandomSunFlare(p=0.1),                                        # Riflessi solari 
        A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.3),                       # Alterazione di luminosità, contrasto, saturazione e tonalità
        A.Resize(384, 384),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),         # Resize nell'intervallo standard 
        ToTensorV2()
    ])

    # Val / Test: Trasformazioni
    val_transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    return train_transform, val_transform