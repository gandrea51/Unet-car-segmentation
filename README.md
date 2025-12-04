# 🚀 Car's Semantic Segmentation: Comparison between Different UNets

## 🎯 Project Objective
Computer Vision project work for the implementation and optimization of UNet architectures for multi-class semantic segmentation on the [Car segmentation dataset](https://www.kaggle.com/datasets/intelecai/car-segmentation).
The project follows an iterative comparison path on a total of 6 models, focused on solving the class imbalance and generalization bound problem, culminating with the use of Transfer Learning to try to maximize segmentation accuracy.

## 🏆 Final Results (Optimized V4 Model)
The final winning setup (UNet V4 with pre-trained Encoder EfficientNet-B0, Loss Focal+Dice, and Class Weights) produced the following performance on the Test set:

| Metrics | Final value | Objective achieved |
| :--- | :--- | :--- |
| **Mean IoU** | 81.68 | Excellent global accuracy |
| **Mean Dice** | 89.27 | Maximum geometric precision |
| **IoU Fanali** | 57.95 | Solving the critical imbalance problem |

## 🛠️ Final Architecture (Model V4)
The final architecture that allowed us to overcome the 39% IoU limit on Fanali is based on Transfer Learning
- **Architecture**: UNet with EfficientNet-B0 Encoder pre-trained on ImageNet
- **Input Resolution**: 384x384
- **Loss Function**: Combination of Focal Loss and Dice Loss, enhanced by statistical Class Weights
- **Advanced Fine-Tuning**: Implementation of backward stepwise Unfreeze to adapt pre-learned features to the specific dataset

## 💾 Repository Structure
The project is organized into clear and separate modules to ensure the maintainability and clarity of the code:

```bash
Car-segmentation-project/
│
├── archive/                # Contains the original dataset (images/, masks/)
├── architecture/           # Model class definitions (UNet_V1, UNet_Efficient, etc.)
├── dataset/                # I/O logic, Data Augmentation (Albumentations), and DataLoader creation
├── engine/                 # Modules for training, metrics (IoU, Dice Score) and early stopping
├── run/                    # Execution script (train.py, inference.py)
├── model/                  # Weights of the final model (unet_efficient_final.pth)
├── graph/                  # Documentation and graphs
├── infer/                  # Images and test results
│
├── LICENSE         # Do you want to use the project? Here are the rules
└── README.md       # This file
```

## 🚀 Execution Instructions
Prerequisites
- Python 3.8+
- PyTorch & Torchvision
- Libraries: NumPy, OpenCV (cv2), Albumentations, Kornia (for Loss Focal).
