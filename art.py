"""
Indian Cultural Artforms Classification - COMPLETE OPTIMIZED VERSION WITH FLEXIBLE FEATURE SIZE
- Disk caching for extracted features (never compute twice)
- Progress estimation with time remaining
- Fixed GradCAM usage (no use_cuda parameter)
- Traditional features (HOG, LBP, GLCM, Edge) + shallow ML
- Deep transfer learning (EfficientNet-B0 or ResNet50) + metrics, confusion, Grad-CAM

Usage Examples:

    Full pipeline with default 256x256 features:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp

    Fast mode with 128x128 features for speed:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp --feature_size 128

    Skip entire traditional pipeline (deep learning only):
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp --skip_traditional

    Use ResNet50 backbone instead of default EfficientNet-B0:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone resnet50 --epochs 25 --amp

    Use custom deep learning image size (512x512):
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp --img_size 512

    Use custom cache directory for feature storage:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --cache_dir my_cache --epochs 25 --amp

    Freeze backbone layers, train only classification head:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp --freeze_backbone

    Extract deep features and train shallow classifier on them:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp --extract_deep_features

    Use specific batch size of 64 images:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp --batch_size 64

    Train for 50 epochs instead of default 25:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 50 --amp

    Use learning rate of 0.001 instead of default 0.0003:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp --lr 0.001

    Use custom random seed for reproducibility:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp --seed 123

    Use custom validation split (20% instead of 15%):
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp --val_split 0.2

    Use custom test split (10% instead of 15%):
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp --test_split 0.1

    Use more CPU workers for data loading (8 instead of 4):
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp --num_workers 8

    Run without mixed precision (AMP disabled):
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25

    Limit maximum images per class to 500:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp --max_images_per_class 500

    Save top 5 predictions instead of default 3:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp --save_topk 5

    Run robustness evaluation with corrupted images:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp --do_robustness

    Clear cached features and recompute all from scratch:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp --clear_cache

    Complete minimal run with only required argument:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100

    High-quality setup with maximum resolution features:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp --feature_size 384 --img_size 384

    Speed-optimized setup for quick experimentation:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 10 --amp --feature_size 128 --batch_size 64

    Production setup with comprehensive evaluation:
    python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone resnet50 --epochs 50 --amp --extract_deep_features --do_robustness --save_topk 5

"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =========================
# Args & Reproducibility
# =========================

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Art Styles Classification - Complete Optimized with Flexible Feature Size")
    parser.add_argument("--data_root", type=str, required=True, help="Root with class subfolders of images")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory for logs and artifacts")
    parser.add_argument("--img_size", type=int, default=256, help="Square image size for deep learning")
    # NEW: Separate feature size for traditional extraction with default 256
    parser.add_argument("--feature_size", type=int, default=256,help="Image size for traditional feature extraction (e.g., 128 for speed, 256 for quality)")
    parser.add_argument("--cache_dir", type=str, default="feature_cache", help="Directory for caching features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze CNN backbone and train head only")
    parser.add_argument("--extract_deep_features", action="store_true",
                        help="Extract penultimate features and train a shallow classifier")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--max_images_per_class", type=int, default=100000)
    parser.add_argument("--save_topk", type=int, default=3)
    parser.add_argument("--do_robustness", action="store_true", help="Run corruption robustness evaluation")
    parser.add_argument("--clear_cache", action="store_true", help="Clear feature cache before running")
    parser.add_argument("--skip_traditional", action="store_true", help="Skip traditional feature extraction (faster execution)")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# File Discovery & Splits
# =========================

def list_images(root: str | Path) -> List[Tuple[str, str]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    root = Path(root)
    items = []
    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in exts:
                items.append((str(p), cls_dir.name))
    return items


def build_splits(
    items: List[Tuple[str, str]],
    labels_map: Dict[str, int],
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    max_per_class: int = 100000,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    by_class: Dict[str, List[str]] = {}
    for path, cls in items:
        by_class.setdefault(cls, []).append(path)

    train_items: List[Tuple[str, str]] = []
    val_items: List[Tuple[str, str]] = []
    test_items: List[Tuple[str, str]] = []

    for cls, paths in by_class.items():
        paths = sorted(paths)
        if len(paths) > max_per_class:
            paths = random.sample(paths, max_per_class)

        tr, tv = train_test_split(paths, test_size=(val_split + test_split), random_state=seed, shuffle=True)
        rel = val_split / (val_split + test_split + 1e-8)
        va, te = train_test_split(tv, test_size=1 - rel, random_state=seed, shuffle=True)

        train_items.extend([(p, cls) for p in tr])
        val_items.extend([(p, cls) for p in va])
        test_items.extend([(p, cls) for p in te])

    return train_items, val_items, test_items


# =========================
# Adaptive Feature Extractors (based on feature_size)
# =========================

def get_cache_key(img_path: str, feature_size: int) -> str:
    """Generate cache key based on image path and feature size"""
    try:
        stat = os.stat(img_path)
        key_data = f"{img_path}_{feature_size}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(key_data.encode()).hexdigest()
    except:
        # Fallback if file stat fails
        key_data = f"{img_path}_{feature_size}"
        return hashlib.md5(key_data.encode()).hexdigest()


def extract_lbp_adaptive(gray: np.ndarray, feature_size: int) -> np.ndarray:
    """LBP with parameters adapted to feature size"""
    if feature_size <= 128:
        # Smaller radius and fewer points for 128x128
        radius, n_points = 1, 8
    else:
        # Standard parameters for 256x256
        radius, n_points = 2, 16
    
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    bins = np.arange(0, n_points + 3)  # fixed-length p+2 bins
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, n_points + 2), density=True)
    return hist.astype(np.float32)


def compute_glcm_features_adaptive(
    gray: np.ndarray,
    feature_size: int,
) -> np.ndarray:
    """GLCM features with parameters adapted to feature size"""
    if feature_size <= 128:
        # Simpler parameters for 128x128
        distances = [1, 2]
        angles = [0, np.pi / 2]
        levels = 16
    else:
        # Full parameters for 256x256
        distances = [1, 2, 4]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        levels = 32
    
    g = cv2.normalize(gray, None, 0, levels - 1, cv2.NORM_MINMAX).astype(np.uint8)
    glcm = graycomatrix(g, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    feats: List[float] = []
    for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]:
        try:
            vals = graycoprops(glcm, prop).ravel()
            feats.extend(vals.tolist())
        except Exception:
            feats.extend([0.0] * (len(distances) * len(angles)))
    return np.array(feats, dtype=np.float32)


def compute_edge_hist_adaptive(gray: np.ndarray, feature_size: int) -> np.ndarray:
    """Edge histogram with bins adapted to feature size"""
    bins = 9 if feature_size <= 128 else 18
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    _, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist, _ = np.histogram(ang.ravel(), bins=bins, range=(0, 180))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def extract_traditional_features_cached(img_bgr: np.ndarray, img_path: str, feature_size: int = 256, cache_dir: str = "feature_cache") -> Dict[str, np.ndarray]:
    """Extract features with disk caching - adaptive to feature size"""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check cache first
    cache_key = get_cache_key(img_path, feature_size)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            # If cache is corrupted, recompute
            pass
    
    # Compute features at specified feature_size
    img = cv2.resize(img_bgr, (feature_size, feature_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive HOG parameters based on feature size
    if feature_size <= 128:
        # Larger cells for smaller images to maintain reasonable feature count
        pixels_per_cell = (16, 16)
    else:
        # Standard cell size for 256x256
        pixels_per_cell = (8, 8)
    
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        transform_sqrt=True,
        feature_vector=True,
    ).astype(np.float32)
    
    lbp_feat = extract_lbp_adaptive(gray, feature_size)
    glcm_feat = compute_glcm_features_adaptive(gray, feature_size)
    edge_hist = compute_edge_hist_adaptive(gray, feature_size)
    
    feats = {
        "hog": hog_feat,
        "lbp": lbp_feat,
        "glcm": glcm_feat,
        "edge": edge_hist,
    }
    feats["fused"] = np.concatenate([feats["hog"], feats["lbp"], feats["glcm"], feats["edge"]], axis=0)
    
    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(feats, f)
    except:
        pass  # Continue even if caching fails
    
    return feats


# =========================
# Dataset for Deep Track
# =========================

class ImageFolderList(Dataset):
    def __init__(self, items: List[Tuple[str, str]], labels_map: Dict[str, int], img_size: int = 256, aug: bool = False):
        self.items = items
        self.labels_map = labels_map
        self.img_size = img_size
        self.aug = aug

        self.train_tf = A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
                A.OneOf(
                    [
                        A.ColorJitter(0.2, 0.2, 0.2, 0.1),
                        A.RandomBrightnessContrast(0.2, 0.2),
                        A.HueSaturationValue(10, 15, 10),
                    ],
                    p=0.8,
                ),
                A.ToFloat(max_value=255.0),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )
        self.val_tf = A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
                A.ToFloat(max_value=255.0),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, cls = self.items[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tf = self.train_tf if self.aug else self.val_tf
        t = tf(image=img)["image"]
        y = self.labels_map[cls]
        return t, y, path


# =========================
# Models & Training Utils
# =========================

def build_model(backbone: str, num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    if backbone == "efficientnet_b0":
        model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
        if freeze_backbone:
            for n, p in model.named_parameters():
                if "classifier" not in n:
                    p.requires_grad = False
    else:
        model = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
        if freeze_backbone:
            for n, p in model.named_parameters():
                if "fc" not in n:
                    p.requires_grad = False
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    scaler: torch.cuda.amp.GradScaler | None = None,
    criterion: nn.Module | None = None,
) -> Tuple[float, float, float]:
    model.train()
    losses: List[float] = []
    logits_all: List[np.ndarray] = []
    ys_all: List[np.ndarray] = []
    criterion = criterion or nn.CrossEntropyLoss()

    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        logits_all.append(out.detach().cpu().numpy())
        ys_all.append(y.cpu().numpy())

    logits = np.concatenate(logits_all, 0)
    ys = np.concatenate(ys_all, 0)
    preds = logits.argmax(1)
    return float(np.mean(losses)), float(accuracy_score(ys, preds)), float(f1_score(ys, preds, average="macro"))


@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, object]:
    model.eval()
    logits_all: List[np.ndarray] = []
    ys_all: List[np.ndarray] = []
    paths: List[str] = []

    for x, y, p in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        logits_all.append(out.detach().cpu().numpy())
        ys_all.append(y.cpu().numpy())
        paths.extend(p)

    logits = np.concatenate(logits_all, 0)
    ys = np.concatenate(ys_all, 0)
    preds = logits.argmax(1)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    acc = float(accuracy_score(ys, preds))
    macro_f1 = float(f1_score(ys, preds, average="macro"))
    kappa = float(cohen_kappa_score(ys, preds))
    cm = confusion_matrix(ys, preds, normalize="true")

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "kappa": kappa,
        "cm": cm,
        "logits": logits,
        "probs": probs,
        "y_true": ys,
        "y_pred": preds,
        "paths": paths,
    }


def plot_confusion(cm: np.ndarray, classes: List[str], path: str) -> None:
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, xticklabels=classes, yticklabels=classes, cmap="Blues", annot=False, fmt='.2f')
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_calibration(probs: np.ndarray, y_true: np.ndarray, path: str) -> None:
    y_pred_prob = probs.max(1)
    bins = np.linspace(0, 1, 11)
    digitized = np.digitize(y_pred_prob, bins) - 1
    accs: List[float] = []
    confs: List[float] = []
    for b in range(len(bins) - 1):
        idx = digitized == b
        if idx.sum() > 0:
            accs.append((np.argmax(probs[idx], 1) == y_true[idx]).mean())
            confs.append(((bins[b] + bins[b + 1]) / 2))
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "--", c="gray", label="Perfect calibration")
    plt.plot(confs, accs, marker="o", linewidth=2, markersize=8, label="Model calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Calibration Plot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def topk_predictions(probs: np.ndarray, classes: List[str], k: int = 3) -> List[Tuple[str, float]]:
    idxs = np.argsort(-probs)[:k]
    return [(classes[i], float(probs[i])) for i in idxs]


# =========================
# Optimized Traditional Pipeline with Configurable Feature Size
# =========================

def batch_extract_fast(items: List[Tuple[str, str]], labels_map: Dict[str, int], feature_size: int, cache_dir: str, desc: str = "extract_feats") -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    """Fast batch extraction with caching and detailed progress tracking"""
    feats: Dict[str, List[np.ndarray]] = {"hog": [], "lbp": [], "glcm": [], "edge": [], "fused": []}
    ys: List[int] = []
    paths: List[str] = []
    
    start_time = time.time()
    cache_hits = 0
    
    print(f"Extracting {desc} features for {len(items)} images at {feature_size}x{feature_size} resolution...")
    
    for i, (p, cls) in enumerate(tqdm(items, desc=desc)):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((feature_size, feature_size, 3), dtype=np.uint8)
        
        # Check if this will be a cache hit
        cache_key = get_cache_key(p, feature_size)
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            cache_hits += 1
        
        # Extract features with caching
        f = extract_traditional_features_cached(img, p, feature_size=feature_size, cache_dir=cache_dir)
        
        for k in feats.keys():
            feats[k].append(f[k])
        ys.append(labels_map[cls])
        paths.append(p)
        
        # Print detailed progress every 100 images
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(items) - i - 1) / rate
            cache_ratio = cache_hits / (i + 1)
            print(f"  ✓ Processed {i+1:,}/{len(items):,} | Rate: {rate:.1f} img/s | ETA: {remaining/60:.1f}m | Cache: {cache_ratio:.1%}")
    
    # Final statistics
    total_time = time.time() - start_time
    cache_ratio = cache_hits / len(items)
    avg_rate = len(items) / total_time
    print(f"  🎉 Completed {len(items):,} images in {total_time/60:.1f} minutes")
    print(f"  📊 Average rate: {avg_rate:.1f} images/second | Cache hits: {cache_ratio:.1%}")
    
    # Validate and stack features
    for k in feats.keys():
        shapes = {tuple(np.array(v).shape) for v in feats[k]}
        if len(shapes) != 1:
            print(f"  ⚠️ Feature '{k}' has varying shapes: {shapes}")
            raise ValueError(f"Feature '{k}' has varying shapes: {shapes}")
        feats[k] = np.stack(feats[k], 0)
        print(f"  📏 {k.upper()} features: {feats[k].shape}")
    
    return feats, np.array(ys, dtype=np.int64), paths


def run_traditional_optimized(
    train_items: List[Tuple[str, str]],
    val_items: List[Tuple[str, str]],
    test_items: List[Tuple[str, str]],
    labels_map: Dict[str, int],
    out_dir: str,
    feature_size: int,  # Use feature_size instead of img_size
    cache_dir: str,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray], np.ndarray, List[str], List[str]]:
    os.makedirs(out_dir, exist_ok=True)
    classes = [c for c, _ in sorted(labels_map.items(), key=lambda x: x[1])]

    print(f"\n🎨 TRADITIONAL FEATURE EXTRACTION")
    print(f"📐 Feature resolution: {feature_size}x{feature_size}")
    print(f"💾 Cache directory: {cache_dir}")
    print(f"🎯 Classes: {len(classes)}")
    
    # Extract features for all splits using feature_size
    train_feats, y_train, _ = batch_extract_fast(train_items, labels_map, feature_size, cache_dir, "train")
    val_feats, y_val, _ = batch_extract_fast(val_items, labels_map, feature_size, cache_dir, "val")
    test_feats, y_test, test_paths = batch_extract_fast(test_items, labels_map, feature_size, cache_dir, "test")

    # Train models on each feature type
    results: Dict[str, object] = {}
    feature_types = ["hog", "lbp", "glcm", "edge", "fused"]
    
    print(f"\n🤖 TRAINING TRADITIONAL MODELS")
    
    for feat_name in feature_types:
        print(f"\n📈 Training models with {feat_name.upper()} features...")
        Xtr = train_feats[feat_name]
        Xv = val_feats[feat_name]
        Xte = test_feats[feat_name]
        
        print(f"   Feature dimensions: {Xtr.shape[1]:,}")

        # Standardize features
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr_s = scaler.fit_transform(Xtr)
        Xv_s = scaler.transform(Xv)
        Xte_s = scaler.transform(Xte)

        # Define models (n_jobs=1 for Windows compatibility)
        models = {
            "logreg": LogisticRegression(max_iter=2000, n_jobs=1, random_state=42),
            "svm_linear": LinearSVC(random_state=42),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "rf": RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=1, 
                                       class_weight="balanced", random_state=42),
        }

        results[feat_name] = {}
        
        for mname, model in models.items():
            print(f"   🔧 Training {mname}...", end=" ")
            t0 = time.time()
            model.fit(Xtr_s, y_train)
            t1 = time.time()
            
            # Evaluate on validation and test sets
            ypv = model.predict(Xv_s)
            ypte = model.predict(Xte_s)

            # Calculate metrics
            acc_v = float(accuracy_score(y_val, ypv))
            f1_v = float(f1_score(y_val, ypv, average="macro"))
            kap_v = float(cohen_kappa_score(y_val, ypv))

            acc = float(accuracy_score(y_test, ypte))
            f1m = float(f1_score(y_test, ypte, average="macro"))
            kap = float(cohen_kappa_score(y_test, ypte))

            results[feat_name][mname] = {
                "val_acc": acc_v,
                "val_macro_f1": f1_v,
                "val_kappa": kap_v,
                "test_acc": acc,
                "test_macro_f1": f1m,
                "test_kappa": kap,
                "train_time_s": float(t1 - t0),
            }
            
            print(f"val_acc={acc_v:.3f}, test_acc={acc:.3f} ({t1-t0:.1f}s)")

        # Save results for this feature type
        df_results = pd.DataFrame(results[feat_name])
        df_results.to_csv(os.path.join(out_dir, f"{feat_name}_models_results.csv"))
        print(f"   💾 Results saved to {feat_name}_models_results.csv")

    # Save complete results
    with open(os.path.join(out_dir, "traditional_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results, test_feats, y_test, test_paths, classes


# =========================
# Deep Pipeline with Fixed GradCAM
# =========================

def build_dataloaders(
    train_items: List[Tuple[str, str]],
    val_items: List[Tuple[str, str]],
    test_items: List[Tuple[str, str]],
    labels_map: Dict[str, int],
    img_size: int,  # Keep img_size for deep learning
    batch_size: int,
    num_workers: int,
):
    train_ds = ImageFolderList(train_items, labels_map, img_size=img_size, aug=True)
    val_ds = ImageFolderList(val_items, labels_map, img_size=img_size, aug=False)
    test_ds = ImageFolderList(test_items, labels_map, img_size=img_size, aug=False)

    n_classes = len(labels_map)
    y_train = np.array([labels_map[c] for _, c in train_items])
    class_counts = np.bincount(y_train, minlength=n_classes)
    sample_weights = 1.0 / (class_counts[y_train] + 1e-6)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train), replacement=True)

    train_ld = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_ld, val_ld, test_ld, test_ds


def run_deep(
    train_items: List[Tuple[str, str]],
    val_items: List[Tuple[str, str]],
    test_items: List[Tuple[str, str]],
    labels_map: Dict[str, int],
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, float]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🧠 DEEP LEARNING PIPELINE")
    print(f"⚡ Device: {device}")
    print(f"🏗️ Backbone: {args.backbone}")
    print(f"📐 Image size: {args.img_size}x{args.img_size}")
    print(f"🎯 Classes: {len(labels_map)}")
    
    classes = [c for c, _ in sorted(labels_map.items(), key=lambda x: x[1])]
    n_classes = len(classes)

    # Build data loaders using img_size for deep learning
    train_ld, val_ld, test_ld, test_ds = build_dataloaders(
        train_items, val_items, test_items, labels_map, args.img_size, args.batch_size, args.num_workers
    )
    
    print(f"📊 Dataset sizes - Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)}")

    # Build model
    model = build_model(args.backbone, n_classes, freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    print(f"🔥 Mixed precision: {'Enabled' if args.amp else 'Disabled'}")
    print(f"🏃 Training for {args.epochs} epochs...")

    # Training loop
    best = {"val_macro_f1": -1.0, "state": None}
    history: List[Dict[str, float]] = []

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_ld, optimizer, device, scaler, criterion)
        val_metrics = eval_model(model, val_ld, device)
        scheduler.step()
        elapsed = time.time() - t0
        
        rec = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_macro_f1": tr_f1,
            "val_acc": float(val_metrics["acc"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_kappa": float(val_metrics["kappa"]),
            "time_s": float(elapsed),
        }
        history.append(rec)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}: val_acc={rec['val_acc']:.3f}, val_f1={rec['val_macro_f1']:.3f}, time={elapsed:.0f}s")
        
        # Save best model
        if float(val_metrics["macro_f1"]) > best["val_macro_f1"]:
            best["val_macro_f1"] = float(val_metrics["macro_f1"])
            best["state"] = copy.deepcopy(model.state_dict())

    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "deep_train_history.csv"), index=False)

    # Load best model and evaluate
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    print(f"\n📊 FINAL EVALUATION")
    val_metrics = eval_model(model, val_ld, device)
    test_metrics = eval_model(model, test_ld, device)
    
    print(f"🎯 Final Results:")
    print(f"   Validation - Acc: {val_metrics['acc']:.3f}, F1: {val_metrics['macro_f1']:.3f}, Kappa: {val_metrics['kappa']:.3f}")
    print(f"   Test       - Acc: {test_metrics['acc']:.3f}, F1: {test_metrics['macro_f1']:.3f}, Kappa: {test_metrics['kappa']:.3f}")

    # Save plots
    plot_confusion(test_metrics["cm"], classes, os.path.join(args.out_dir, "deep_confusion.png"))
    plot_calibration(test_metrics["probs"], test_metrics["y_true"], os.path.join(args.out_dir, "deep_calibration.png"))
    
    # Save metrics
    with open(os.path.join(args.out_dir, "deep_test_metrics.json"), "w") as f:
        json.dump({
            "acc": float(test_metrics["acc"]), 
            "macro_f1": float(test_metrics["macro_f1"]), 
            "kappa": float(test_metrics["kappa"])
        }, f, indent=2)

    # FIXED: Grad-CAM visualization (removed use_cuda parameter)
    print(f"\n🔍 GENERATING GRAD-CAM VISUALIZATIONS")
    target_layer = None
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            target_layer = m
    
    if target_layer is not None:
        print(f"   Target layer: {target_layer}")
        # FIXED: Removed use_cuda parameter - it detects device automatically
        cam = GradCAM(model=model, target_layers=[target_layer])
        
        gradcam_dir = os.path.join(args.out_dir, "gradcam")
        os.makedirs(gradcam_dir, exist_ok=True)
        
        for i in range(min(10, len(test_ds))):
            x, y, p = test_ds[i]
            xx = x.unsqueeze(0).to(device)
            targets = [ClassifierOutputTarget(int(y))]
            grayscale_cam = cam(input_tensor=xx, targets=targets)[0]
            
            # Denormalize image for visualization
            img = x.permute(1, 2, 0).cpu().numpy()
            img = (img * 0.5 + 0.5).clip(0, 1)
            
            # Create overlay
            overlay = show_cam_on_image(img.astype(np.float32), grayscale_cam, use_rgb=True)
            output_path = os.path.join(gradcam_dir, f"{i:02d}_{Path(p).stem}_gradcam.png")
            cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        print(f"   💾 Saved {min(10, len(test_ds))} Grad-CAM visualizations to {gradcam_dir}")
    else:
        print(f"   ⚠️ No convolutional layer found for Grad-CAM")

    # Optional: Deep features + shallow classifier
    deep_sklearn_results: Dict[str, float] = {}
    if args.extract_deep_features:
        print(f"\n🔗 EXTRACTING DEEP FEATURES FOR SHALLOW CLASSIFIER")
        feat_model_name = "efficientnet_b0" if args.backbone == "efficientnet_b0" else "resnet50"
        feat_model = timm.create_model(feat_model_name, pretrained=True, num_classes=0).to(device)
        feat_model.eval()

        def pool_features(loader: DataLoader, desc: str) -> Tuple[np.ndarray, np.ndarray]:
            feats: List[np.ndarray] = []
            ys: List[np.ndarray] = []
            for x, y, _ in tqdm(loader, desc=desc):
                with torch.no_grad():
                    f = feat_model(x.to(device))
                feats.append(f.cpu().numpy())
                ys.append(y.numpy())
            return np.concatenate(feats, 0), np.concatenate(ys, 0)

        f_tr, y_tr = pool_features(train_ld, "deep_train")
        f_va, y_va = pool_features(val_ld, "deep_val")
        f_te, y_te = pool_features(test_ld, "deep_test")

        print(f"   Feature dimensions: {f_tr.shape[1]:,}")

        scaler2 = StandardScaler()
        f_trs = scaler2.fit_transform(f_tr)
        f_vas = scaler2.transform(f_va)
        f_tes = scaler2.transform(f_te)

        clf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=1, random_state=42)
        t0 = time.time()
        clf.fit(f_trs, y_tr)
        t1 = time.time()
        ypv = clf.predict(f_vas)
        ypte = clf.predict(f_tes)

        deep_sklearn_results = {
            "val_acc": float(accuracy_score(y_va, ypv)),
            "val_macro_f1": float(f1_score(y_va, ypv, average="macro")),
            "val_kappa": float(cohen_kappa_score(y_va, ypv)),
            "test_acc": float(accuracy_score(y_te, ypte)),
            "test_macro_f1": float(f1_score(y_te, ypte, average="macro")),
            "test_kappa": float(cohen_kappa_score(y_te, ypte)),
            "train_time_s": float(t1 - t0),
        }
        
        print(f"   🎯 Deep features RF: test_acc={deep_sklearn_results['test_acc']:.3f}, "
              f"test_f1={deep_sklearn_results['test_macro_f1']:.3f}")
        
        with open(os.path.join(args.out_dir, "deep_features_rf.json"), "w") as f:
            json.dump(deep_sklearn_results, f, indent=2)

    return val_metrics, test_metrics, deep_sklearn_results


# =========================
# Visualization & Utils
# =========================

def letterbox_resize(img: np.ndarray, size: int, color=(0, 0, 0)) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    img_res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left
    return cv2.copyMakeBorder(img_res, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def visualize_predictions(test_metrics: Dict[str, object], classes: List[str], out_dir: str, save_topk: int = 3) -> None:
    pred_dir = os.path.join(out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    probs = test_metrics["probs"]
    paths = test_metrics["paths"]
    y_true = test_metrics["y_true"]
    y_pred = test_metrics["y_pred"]
    
    print(f"\n🖼️ GENERATING PREDICTION VISUALIZATIONS")
    
    n_viz = min(len(paths), 50)
    for i in tqdm(range(n_viz), desc="predictions"):
        p = paths[i]
        prob = probs[i]
        yt = y_true[i]
        yp = y_pred[i]
        
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        
        img = letterbox_resize(img, 512)
        topk = topk_predictions(prob, classes, k=save_topk)
        
        # Create detailed label
        true_class = classes[yt][:20]  # Truncate long class names
        pred_class = classes[yp][:20]
        confidence = prob[yp]
        
        label = f"True: {true_class}"
        pred_label = f"Pred: {pred_class} ({confidence:.2%})"
        top3_label = "Top3: " + ", ".join([f"{n[:15]}({s:.1%})" for n, s in topk])
        
        canvas = img.copy()
        
        # Draw labels with background for readability
        labels = [label, pred_label, top3_label]
        y_positions = [30, 60, 90]
        
        for text, y_pos in zip(labels, y_positions):
            # Get text size for background
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background rectangle
            cv2.rectangle(canvas, (5, y_pos - text_height - 5), (15 + text_width, y_pos + 5), (0, 0, 0), -1)
            
            # Draw text
            color = (0, 255, 0) if yt == yp else (0, 165, 255)  # Green if correct, orange if wrong
            cv2.putText(canvas, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        # Save with descriptive filename
        filename = f"{i:02d}_{Path(p).stem}_{'correct' if yt == yp else 'wrong'}.jpg"
        cv2.imwrite(os.path.join(pred_dir, filename), canvas)
    
    print(f"   💾 Saved {n_viz} prediction visualizations to {pred_dir}")


# =========================
# Main Pipeline (MODIFIED with --feature_size support)
# =========================

def main() -> None:
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Clear cache if requested
    if args.clear_cache and os.path.exists(args.cache_dir):
        import shutil
        print(f"🗑️ Clearing cache directory: {args.cache_dir}")
        shutil.rmtree(args.cache_dir)

    print("="*70)
    print("🎨 INDIAN CULTURAL ARTFORMS CLASSIFICATION")
    print("="*70)
    print(f"🧠 Deep learning image size: {args.img_size}x{args.img_size}")
    print(f"🔧 Traditional feature size: {args.feature_size}x{args.feature_size}")
    print(f"💾 Cache directory: {args.cache_dir}")
    print(f"🎯 Output directory: {args.out_dir}")
    print(f"🌱 Random seed: {args.seed}")
    if args.skip_traditional:
        print(f"⚡ Mode: Deep learning only (traditional pipeline SKIPPED)")
    else:
        print(f"🔄 Mode: Full pipeline (traditional @ {args.feature_size}x{args.feature_size} + deep @ {args.img_size}x{args.img_size})")

    # Discover data
    print(f"\n📁 DISCOVERING DATA")
    items = list_images(args.data_root)
    if not items:
        raise RuntimeError(f"No images found in: {args.data_root}")

    labels = sorted(list({c for _, c in items}))
    labels_map = {c: i for i, c in enumerate(labels)}
    
    print(f"   📊 Found {len(labels)} classes with {len(items):,} total images")
    print(f"   🏷️ Classes: {', '.join(labels[:5])}{'...' if len(labels) > 5 else ''}")

    # Build splits
    print(f"\n✂️ BUILDING DATA SPLITS")
    train_items, val_items, test_items = build_splits(
        items, labels_map, 
        val_split=args.val_split, 
        test_split=args.test_split, 
        seed=args.seed, 
        max_per_class=args.max_images_per_class
    )
    
    print(f"   🚂 Train: {len(train_items):,} images")
    print(f"   ✅ Validation: {len(val_items):,} images") 
    print(f"   🧪 Test: {len(test_items):,} images")

    # Get class names for later use
    classes = [c for c, _ in sorted(labels_map.items(), key=lambda x: x[1])]

    # Traditional pipeline (CONDITIONAL) - NOW USES args.feature_size
    trad_dir = os.path.join(args.out_dir, "traditional")
    if not args.skip_traditional:
        t0 = time.time()
        # MODIFIED: Pass args.feature_size instead of args.img_size
        trad_results, test_feats, y_test, test_paths, classes = run_traditional_optimized(
            train_items, val_items, test_items, labels_map, trad_dir, args.feature_size, args.cache_dir
        )
        t1 = time.time()
        print(f"✅ Traditional pipeline completed in {(t1 - t0)/60:.1f} minutes")
    else:
        print(f"\n⚡ SKIPPING TRADITIONAL PIPELINE")
        print(f"   🚀 Jumping directly to deep learning for faster execution")
        trad_results = {}
        # Create empty traditional directory for consistency
        os.makedirs(trad_dir, exist_ok=True)
        with open(os.path.join(trad_dir, "skipped.txt"), "w") as f:
            f.write("Traditional pipeline was skipped using --skip_traditional flag\n")

    # Deep learning pipeline (ALWAYS RUNS) - USES args.img_size
    deep_dir = os.path.join(args.out_dir, "deep")
    os.makedirs(deep_dir, exist_ok=True)
    dval, dtest, deep_rf = run_deep(
        train_items, val_items, test_items, labels_map, 
        argparse.Namespace(**{**vars(args), "out_dir": deep_dir})
    )

    # Final results summary
    print(f"\n📈 SAVING COMPREHENSIVE RESULTS")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        summary_data = {
            "metadata": {
                "total_images": len(items),
                "num_classes": len(labels),
                "deep_img_size": args.img_size,
                "feature_size": args.feature_size,
                "backbone": args.backbone,
                "epochs": args.epochs,
                "skip_traditional": args.skip_traditional,
            },
            "traditional": trad_results,
            "deep_val": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in dval.items() if k in ["acc", "macro_f1", "kappa"]},
            "deep_test": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                        for k, v in dtest.items() if k in ["acc", "macro_f1", "kappa"]},
            "deep_features_rf": deep_rf,
        }
        json.dump(summary_data, f, indent=2)

    # Generate prediction visualizations
    visualize_predictions(dtest, classes, deep_dir, save_topk=args.save_topk)

    # Final summary
    print(f"\n🎉 CLASSIFICATION COMPLETE!")
    print("="*70)
    print(f"📊 Results Summary:")
    print(f"   🧠 Deep Learning - Test Accuracy: {dtest['acc']:.1%}")
    print(f"   🧠 Deep Learning - Test F1-Score: {dtest['macro_f1']:.1%}")
    
    # Find best traditional result (if not skipped)
    if not args.skip_traditional and trad_results:
        best_trad = 0
        best_trad_name = ""
        for feat_type, models in trad_results.items():
            for model_name, metrics in models.items():
                if metrics['test_acc'] > best_trad:
                    best_trad = metrics['test_acc']
                    best_trad_name = f"{feat_type}+{model_name}"
        
        print(f"   🔧 Best Traditional ({args.feature_size}x{args.feature_size}) - {best_trad_name}: {best_trad:.1%}")
    else:
        print(f"   ⚡ Traditional pipeline: SKIPPED")
    
    print(f"\n📁 Output Files:")
    print(f"   📋 {args.out_dir}/summary.json (complete results)")
    if not args.skip_traditional:
        print(f"   📊 {args.out_dir}/traditional/*.csv (traditional model results)")
    print(f"   🧠 {args.out_dir}/deep/deep_confusion.png (confusion matrix)")
    print(f"   🔍 {args.out_dir}/deep/gradcam/ (attention visualizations)")
    print(f"   🖼️ {args.out_dir}/deep/predictions/ (sample predictions)")
    if not args.skip_traditional:
        print(f"   💾 {args.cache_dir}/ (cached features @ {args.feature_size}x{args.feature_size})")
    print("="*70)


if __name__ == "__main__":
    main()
