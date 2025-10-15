# Indian Art Work Classifier

A comprehensive deep learning project for classifying traditional Indian cultural art forms using both traditional computer vision features and modern deep learning approaches.

## 🎨 Project Overview

This project implements a dual-pipeline classification system for Indian art styles:
- **Traditional Pipeline**: Extracts handcrafted features (HOG, LBP, GLCM, Edge) and trains shallow classifiers
- **Deep Learning Pipeline**: Uses pre-trained CNN models (EfficientNet-B0, ResNet50) with transfer learning

The classifier can identify **30 different Indian art forms** including Madhubani paintings, Warli art, Kalamkari, Pattachitra, and many more regional art styles.

## 🎯 Features

- **Multi-approach Classification**: Traditional computer vision + Deep learning
- **Flexible Architecture**: Support for EfficientNet-B0 and ResNet50 backbones
- **Comprehensive Evaluation**: Accuracy, F1-score, Cohen's Kappa, confusion matrices
- **Visualization Tools**: GradCAM attention maps, prediction visualizations
- **Caching System**: Intelligent feature caching for faster experimentation
- **Robustness Testing**: Evaluation under image corruptions
- **Mixed Precision Training**: AMP support for faster training

## 🏗️ Architecture

### Art Forms Supported
The classifier recognizes 30 traditional Indian art forms:
- **Madhubani Painting** (Bihar)
- **Warli Folk Painting** (Maharashtra) 
- **Pattachitra Painting** (Odisha/Bengal)
- **Kalamkari Painting** (AP/Telangana)
- **Pichwai Painting** (Rajasthan)
- **Kerala Mural Painting** (Kerala)
- **Gond Painting** (Madhya Pradesh)
- **Kalighat Painting** (West Bengal)
- **Mughal Paintings**
- **Mandala Art**
- And 20+ more regional art styles

### Technical Pipeline
```
Input Image → Feature Extraction → Classification → Visualization
     ↓              ↓                    ↓             ↓
Traditional:   HOG, LBP, GLCM     → SVM/RF/KNN    → Metrics
Deep Learning: CNN Features       → Neural Net    → GradCAM
```

## 📦 Dataset Structure

```
DLimages/
├── indian_art_dataset_100/     # 100 images per class
├── indian_art_dataset_1000/    # 1000 images per class  
├── indian_art_dataset_5000/    # 5000 images per class
└── webscrapCode.py            # Data collection script
```

Each dataset contains folders for all 30 art forms with images scraped from Bing Images.

## 🛠️ Installation & Setup

### Environment Setup (Anaconda)

```bash
# Create conda environment with Python 3.11
conda create -n artstyles python=3.11 -y
conda activate artstyles

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Verify CUDA installation
python -c "import torch; print('CUDA:', torch.cuda.is_available())"  # should print CUDA: True

# Install additional dependencies
pip install timm albumentations grad-cam umap-learn seaborn opencv-python scikit-image scikit-learn pandas matplotlib tqdm
```

### Alternative CPU-only Installation
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
pip install timm albumentations pytorch-grad-cam umap-learn seaborn opencv-python scikit-image scikit-learn pandas matplotlib tqdm
```

## 🚀 Quick Start

### Basic Usage
```bash
# Full pipeline with default settings
python art.py --data_root DLimages/indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp

# Fast experimentation mode
python art.py --data_root DLimages/indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 10 --amp --feature_size 128 --batch_size 64

# Deep learning only (skip traditional pipeline)
python art.py --data_root DLimages/indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp --skip_traditional
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_root` | Required | Path to dataset with class folders |
| `--backbone` | `efficientnet_b0` | CNN backbone (`efficientnet_b0`, `resnet50`) |
| `--img_size` | `256` | Input image size for deep learning |
| `--feature_size` | `256` | Image size for traditional features |
| `--epochs` | `25` | Number of training epochs |
| `--batch_size` | `32` | Training batch size |
| `--lr` | `0.0003` | Learning rate |
| `--amp` | `False` | Enable mixed precision training |
| `--skip_traditional` | `False` | Skip traditional pipeline |

## 📊 Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall classification accuracy
- **Macro F1-Score**: Average F1 across all classes
- **Cohen's Kappa**: Agreement correcting for chance
- **Confusion Matrix**: Per-class performance analysis

### Expected Results
- **Deep Learning**: ~85-95% accuracy on test set
- **Traditional + ML**: ~70-85% accuracy depending on features
- **Best Performance**: Fused features with Random Forest

## 📁 Output Structure

```
outputs/
├── traditional/                 # Traditional ML results
│   ├── hog_models_results.csv
│   ├── lbp_models_results.csv
│   ├── glcm_models_results.csv
│   ├── edge_models_results.csv
│   └── fused_models_results.csv
├── deep/                       # Deep learning results
│   ├── deep_confusion.png      # Confusion matrix
│   ├── deep_calibration.png    # Calibration plot
│   ├── gradcam/               # Attention visualizations
│   └── predictions/           # Sample predictions
├── summary.json               # Complete results
└── feature_cache/            # Cached features
```

## 🎛️ Advanced Usage

### Custom Training Configurations

```bash
# High-quality setup with maximum resolution
python art.py --data_root DLimages/indian_art_dataset_1000 --backbone efficientnet_b0 --epochs 50 --feature_size 384 --img_size 384

# Production setup with comprehensive evaluation  
python art.py --data_root DLimages/indian_art_dataset_5000 --backbone resnet50 --epochs 50 --extract_deep_features --do_robustness --save_topk 5

# Memory-efficient training
python art.py --data_root DLimages/indian_art_dataset_100 --batch_size 16 --num_workers 4 --feature_size 128
```

### Feature Engineering Options
- `--freeze_backbone`: Freeze CNN layers, train only classifier head
- `--extract_deep_features`: Extract CNN features → train shallow classifier
- `--clear_cache`: Clear cached features and recompute
- `--do_robustness`: Evaluate on corrupted images

## 🔬 Technical Details

### Traditional Features
- **HOG (Histogram of Oriented Gradients)**: Shape and edge patterns
- **LBP (Local Binary Patterns)**: Texture analysis
- **GLCM (Gray-Level Co-occurrence Matrix)**: Spatial texture relationships
- **Edge Histograms**: Orientation distribution of edges

### Deep Learning Architecture
- **Backbone**: EfficientNet-B0 or ResNet50 (ImageNet pre-trained)
- **Training**: AdamW optimizer, Cosine LR scheduling, Cross-entropy loss
- **Augmentation**: Color jittering, brightness/contrast, HSV shifts
- **Regularization**: Weight decay, balanced sampling

### Performance Optimizations
- Intelligent disk caching for traditional features
- Mixed precision training (AMP) for faster GPU training
- Weighted sampling for class imbalance
- Multi-worker data loading

## 📈 Monitoring & Visualization

The system provides comprehensive visualization:
- **Training Curves**: Loss, accuracy, F1-score over epochs
- **Confusion Matrices**: Per-class performance heatmaps  
- **Calibration Plots**: Model confidence analysis
- **GradCAM**: Visual attention maps showing what the model focuses on
- **Prediction Gallery**: Sample predictions with confidence scores

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 📞 Contact

For questions, issues, or collaborations:
- **GitHub**: [@Subhash-777](https://github.com/Subhash-777)
- **Project Issues**: [GitHub Issues](https://github.com/Subhash-777/Indian-Art-Work-Classifier/issues)

---

**Note**: This project is designed for educational and research purposes. Please respect copyright and cultural significance when using or modifying the datasets and models.
