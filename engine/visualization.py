import matplotlib.pyplot as plt

def plot_history(history: dict):
    '''Plot dell'andamento dell'addestramento'''

    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train loss')
    plt.plot(epochs, history['val_loss'], label='Val loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_iou'], label='Train IoU')
    plt.plot(epochs, history['val_iou'], label='Val IoU')
    plt.title('IoU')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_dice'], label='Train dice')
    plt.plot(epochs, history['val_dice'], label='Val dice')
    plt.title('Dice score')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()


