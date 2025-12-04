from torch.utils.data import Dataset
import cv2, os

class CarSegmentationDataset(Dataset):
    '''Dataset per la segmentazione semantica'''

    def __init__(self, img_dir, mask_dir, transform=None):
        '''Inizializzazione del dataset'''

        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Lettura - ordinamento dei file
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.png')])

        if len(self.image_files) != len(self.mask_files):
            raise ValueError(f'Numero di immagini ({len(self.image_files)}) != Numero di maschere ({len(self.mask_files)}).')
    
    def __len__(self):
        '''Return: num. totale di campioni'''

        return len(self.image_files)
    
    def __getitem__(self, index):
        '''Return: (immagine, maschera) dato un indice'''

        img_path = os.path.join(self.img_dir, self.image_files[index])
        mask_path = os.path.join(self.mask_dir, self.mask_files[index])

        # Lettura - conversione dati
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Trasformazioni
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()
        
        return image, mask
