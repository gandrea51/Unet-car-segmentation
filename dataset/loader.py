import torch
from torch.utils.data import DataLoader, random_split, Subset
from dataset.dataset import CarSegmentationDataset
from dataset.transform import get_transform

def make_dataloader(img_dir, mask_dir, batch=4, workers=2, seed=42):
    '''Creazione dei dataloader (train, validation, test)'''

    train_tf, val_tf = get_transform()

    # Dataset base per la length
    base_ds = CarSegmentationDataset(img_dir, mask_dir, transform=None)

    # Dimensioni dei subset
    n = len(base_ds)
    train_n = int(0.7 * n)
    val_n = int(0.15 * n)
    test_n = n - train_n - val_n
    gen = torch.Generator().manual_seed(seed)

    train_index, val_index, test_index = random_split(range(n), [train_n, val_n, test_n], generator=gen)

    # Creazione dei dataset
    train_ds = Subset(CarSegmentationDataset(img_dir, mask_dir, transform=train_tf), train_index.indices)
    val_ds = Subset(CarSegmentationDataset(img_dir, mask_dir, transform=val_tf), val_index.indices)
    test_ds = Subset(CarSegmentationDataset(img_dir, mask_dir, transform=val_tf), test_index.indices)

    # Creazione dei dataloader
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=workers)

    return train_loader, val_loader, test_loader
