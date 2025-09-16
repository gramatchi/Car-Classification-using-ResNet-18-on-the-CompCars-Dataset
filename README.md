# Car Classification using ResNet-18 on the CompCars Dataset

A deep learning project for classifying car manufacturers from surveillance camera images using convolutional neural networks.

## 🔍 Overview

This project tackles the challenging task of fine-grained car manufacturer classification from surveillance camera images. Unlike web-based car images, surveillance footage presents unique challenges including poor lighting conditions, occlusions, varying viewing angles, and low resolution.

The solution employs ResNet-18 architecture with specialized techniques to handle class imbalance and improve classification performance on underrepresented car brands.

## 📊 Dataset

**CompCars Dataset - Surveillance Subset**
- **67 car manufacturers**
- **Surveillance camera images** captured under real-world conditions
- **Severe class imbalance**: Range from 12 samples (Infiniti) to 2,526 samples (Volkswagen)
- **Challenging conditions**: Blur, extreme lighting, occlusions, limited feature visibility

### Data Preprocessing
- Image resizing to 224×224 pixels
- RGB normalization using ImageNet statistics
- Targeted data augmentation for underrepresented classes (<500 samples)
- Standard train/validation/test split

### Key Improvements
- **Focal Loss + Augmentation**: +4-5% accuracy improvement
- **Class Balance**: Better performance on minority classes
- **Generalization**: Strong performance on unseen surveillance images

### Common Misclassifications
- Similar-looking brands (BMW vs Mercedes-Benz)
- Poor lighting conditions
- Heavy occlusions
- Extreme viewing angles

## 🤝 Contributors

- **Daria Nikolaeva** - Model architecture and training implementation
- **Andrey Malikov** - Data preprocessing and augmentation pipeline  
- **Nichita Gramatchi** - Model evaluation and performance analysis

## 🔬 Technical Details

### Training Configuration
- **Framework**: PyTorch
- **Hardware**: Google Colab with L4 GPU
- **Optimizer**: Adam (lr=1e-4)
- **Batch Size**: 32
- **Training Time**: ~10 minutes/epoch

📄 **Full report available in PDF format. Implementation code provided in Jupyter notebook (.ipynb) files.**
