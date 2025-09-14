"""
Indian Cultural Artforms Classification
- Traditional features (HOG, LBP fixed bins, GLCM, Edge orientation) + shallow ML
- Deep transfer learning (EfficientNet-B0 or ResNet50) + metrics, confusion, Grad-CAM
- Windows-safe: sklearn models run single-threaded to avoid DLL issues in subprocesses

Usage:
  python artstyles.py --data_root DLimages\\indian_art_dataset_100 --out_dir outputs --backbone efficientnet_b0 --epochs 25 --amp
"""

from __future__ import annotations

import argparse
import copy
import json
import os
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
    parser = argparse.ArgumentParser("Art Styles Classification")
    parser.add_argument("--data_root", type=str, required=True, help="Root with class subfolders of images")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory for logs and artifacts")
    parser.add_argument("--img_size", type=int, default=256, help="Square image size")
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
# Traditional Feature Extractors
# =========================

def strict_resize(img_bgr: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)


def extract_hog(gray: np.ndarray) -> np.ndarray:
    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        transform_sqrt=True,
        feature_vector=True,
    )
    return feat.astype(np.float32)


def extract_lbp(gray: np.ndarray, radius: int = 2, n_points: int | None = None) -> np.ndarray:
    if n_points is None:
        n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    bins = np.arange(0, n_points + 3)  # fixed-length p+2 bins
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, n_points + 2), density=True)
    return hist.astype(np.float32)


def compute_glcm_features(
    gray: np.ndarray,
    distances: List[int] = [1, 2, 4],
    angles: List[float] = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    levels: int = 32,
) -> np.ndarray:
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


def compute_edge_hist(gray: np.ndarray, bins: int = 18) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    _, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist, _ = np.histogram(ang.ravel(), bins=bins, range=(0, 180))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def extract_traditional_features(img_bgr: np.ndarray, size: int = 256) -> Dict[str, np.ndarray]:
    img = strict_resize(img_bgr, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = {
        "hog": extract_hog(gray),
        "lbp": extract_lbp(gray, radius=2, n_points=16),
        "glcm": compute_glcm_features(gray),
        "edge": compute_edge_hist(gray),
    }
    feats["fused"] = np.concatenate([feats["hog"], feats["lbp"], feats["glcm"], feats["edge"]], axis=0)
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
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=classes, yticklabels=classes, cmap="Blues", annot=False)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
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
    plt.figure()
    plt.plot([0, 1], [0, 1], "--", c="gray")
    plt.plot(confs, accs, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def topk_predictions(probs: np.ndarray, classes: List[str], k: int = 3) -> List[Tuple[str, float]]:
    idxs = np.argsort(-probs)[:k]
    return [(classes[i], float(probs[i])) for i in idxs]


# =========================
# Traditional Pipeline
# =========================

def run_traditional(
    train_items: List[Tuple[str, str]],
    val_items: List[Tuple[str, str]],
    test_items: List[Tuple[str, str]],
    labels_map: Dict[str, int],
    out_dir: str,
    img_size: int,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray], np.ndarray, List[str], List[str]]:
    os.makedirs(out_dir, exist_ok=True)
    # FIXED: Changed x[21] to x[1] in the lambda function
    classes = [c for c, _ in sorted(labels_map.items(), key=lambda x: x[1])]

    def batch_extract(items: List[Tuple[str, str]]):
        feats: Dict[str, List[np.ndarray]] = {"hog": [], "lbp": [], "glcm": [], "edge": [], "fused": []}
        ys: List[int] = []
        paths: List[str] = []
        for p, cls in tqdm(items, desc="extract_feats"):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            f = extract_traditional_features(img, size=img_size)
            for k in feats.keys():
                feats[k].append(f[k])
            ys.append(labels_map[cls])
            paths.append(p)

        for k in feats.keys():
            shapes = {tuple(np.array(v).shape) for v in feats[k]}
            if len(shapes) != 1:
                raise ValueError(f"Feature '{k}' has varying shapes: {shapes}")
            feats[k] = np.stack(feats[k], 0)

        ys_arr = np.array(ys, dtype=np.int64)
        return feats, ys_arr, paths

    train_feats, y_train, _ = batch_extract(train_items)
    val_feats, y_val, _ = batch_extract(val_items)
    test_feats, y_test, test_paths = batch_extract(test_items)

    results: Dict[str, object] = {}
    for feat_name in ["hog", "lbp", "glcm", "edge", "fused"]:
        Xtr = train_feats[feat_name]
        Xv = val_feats[feat_name]
        Xte = test_feats[feat_name]

        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr_s = scaler.fit_transform(Xtr)
        Xv_s = scaler.transform(Xv)
        Xte_s = scaler.transform(Xte)

        models = {
            "logreg": LogisticRegression(max_iter=2000, n_jobs=1),
            "svm_linear": LinearSVC(),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "rf": RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=1, class_weight="balanced"),
        }

        results[feat_name] = {}
        for mname, model in models.items():
            t0 = time.time()
            model.fit(Xtr_s, y_train)
            t1 = time.time()
            ypv = model.predict(Xv_s)
            ypte = model.predict(Xte_s)

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

        pd.DataFrame(results[feat_name]).to_csv(os.path.join(out_dir, f"{feat_name}_models_results.csv"))

    with open(os.path.join(out_dir, "traditional_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results, test_feats, y_test, test_paths, classes


# =========================
# Deep Pipeline
# =========================

def build_dataloaders(
    train_items: List[Tuple[str, str]],
    val_items: List[Tuple[str, str]],
    test_items: List[Tuple[str, str]],
    labels_map: Dict[str, int],
    img_size: int,
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
    classes = [c for c, _ in sorted(labels_map.items(), key=lambda x: x[1])]
    n_classes = len(classes)

    train_ld, val_ld, test_ld, test_ds = build_dataloaders(
        train_items, val_items, test_items, labels_map, args.img_size, args.batch_size, args.num_workers
    )

    model = build_model(args.backbone, n_classes, freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

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
        print(rec)
        if float(val_metrics["macro_f1"]) > best["val_macro_f1"]:
            best["val_macro_f1"] = float(val_metrics["macro_f1"])
            best["state"] = copy.deepcopy(model.state_dict())

    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "deep_train_history.csv"), index=False)
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    val_metrics = eval_model(model, val_ld, device)
    test_metrics = eval_model(model, test_ld, device)

    plot_confusion(test_metrics["cm"], classes, os.path.join(args.out_dir, "deep_confusion.png"))
    plot_calibration(test_metrics["probs"], test_metrics["y_true"], os.path.join(args.out_dir, "deep_calibration.png"))
    with open(os.path.join(args.out_dir, "deep_test_metrics.json"), "w") as f:
        json.dump({"acc": float(test_metrics["acc"]), "macro_f1": float(test_metrics["macro_f1"]), "kappa": float(test_metrics["kappa"])}, f, indent=2)

    target_layer = None
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            target_layer = m
    if target_layer is not None:
        cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device == "cuda"))
        os.makedirs(os.path.join(args.out_dir, "gradcam"), exist_ok=True)
        for i in range(min(10, len(test_ds))):
            x, y, p = test_ds[i]
            xx = x.unsqueeze(0).to(device)
            targets = [ClassifierOutputTarget(int(y))]
            grayscale_cam = cam(input_tensor=xx, targets=targets)[0]
            img = x.permute(1, 2, 0).cpu().numpy()
            img = (img * 0.5 + 0.5).clip(0, 1)
            overlay = show_cam_on_image(img.astype(np.float32), grayscale_cam, use_rgb=True)
            cv2.imwrite(os.path.join(args.out_dir, "gradcam", f"{i}_{Path(p).stem}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    deep_sklearn_results: Dict[str, float] = {}
    if args.extract_deep_features:
        feat_model_name = "efficientnet_b0" if args.backbone == "efficientnet_b0" else "resnet50"
        feat_model = timm.create_model(feat_model_name, pretrained=True, num_classes=0).to(device)
        feat_model.eval()

        def pool_features(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
            feats: List[np.ndarray] = []
            ys: List[np.ndarray] = []
            for x, y, _ in tqdm(loader, desc="feat"):
                with torch.no_grad():
                    f = feat_model(x.to(device))
                feats.append(f.cpu().numpy())
                ys.append(y.numpy())
            return np.concatenate(feats, 0), np.concatenate(ys, 0)

        f_tr, y_tr = pool_features(train_ld)
        f_va, y_va = pool_features(val_ld)
        f_te, y_te = pool_features(test_ld)

        scaler2 = StandardScaler()
        f_trs = scaler2.fit_transform(f_tr)
        f_vas = scaler2.transform(f_va)
        f_tes = scaler2.transform(f_te)

        clf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=1)
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
        with open(os.path.join(args.out_dir, "deep_features_rf.json"), "w") as f:
            json.dump(deep_sklearn_results, f, indent=2)

    return val_metrics, test_metrics, deep_sklearn_results


# =========================
# Robustness (Optional)
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


def apply_corruptions(img: np.ndarray) -> Dict[str, np.ndarray]:
    outs: Dict[str, np.ndarray] = {}
    for sigma in [10, 25, 50]:
        n = img + np.random.normal(0, sigma, img.shape)
        outs[f"gauss_{sigma}"] = np.clip(n, 0, 255).astype(np.uint8)
    for q in [90, 60, 30]:
        enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])[1]
        dec = cv2.imdecode(enc, 1)
        outs[f"jpeg_{q}"] = dec
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    outs["gray"] = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return outs


@torch.no_grad()
def robustness_eval(
    model: nn.Module,
    test_items: List[Tuple[str, str]],
    labels_map: Dict[str, int],
    img_size: int,
    device: str,
    out_dir: str,
) -> None:
    os.makedirs(os.path.join(out_dir, "robustness"), exist_ok=True)
    records: List[Dict[str, object]] = []

    for p, cls in tqdm(test_items, desc="robust"):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        corrs = apply_corruptions(img)
        for tag, im in corrs.items():
            imr = cv2.cvtColor(letterbox_resize(im, img_size), cv2.COLOR_BGR2RGB)
            ten = A.Compose([A.ToFloat(max_value=255.0), A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])(image=imr)["image"]
            ten = torch.from_numpy(ten.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            out = model(ten)
            pred = out.argmax(1).item()
            records.append({"path": p, "true": labels_map[cls], "pred": pred, "corr": tag})

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, "robustness", "deep_corr_predictions.csv"), index=False)
    summary = df.groupby("corr").apply(lambda g: (g["true"] == g["pred"]).mean()).reset_index(name="acc")
    summary.to_csv(os.path.join(out_dir, "robustness", "summary.csv"), index=False)


# =========================
# Visualization of Predictions
# =========================

def visualize_predictions(test_metrics: Dict[str, object], classes: List[str], out_dir: str, save_topk: int = 3) -> None:
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    probs = test_metrics["probs"]
    paths = test_metrics["paths"]
    y_true = test_metrics["y_true"]
    y_pred = test_metrics["y_pred"]

    for i in range(min(len(paths), 50)):
        p = paths[i]
        prob = probs[i]
        yt = y_true[i]
        yp = y_pred[i]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = letterbox_resize(img, 512)
        topk = topk_predictions(prob, classes, k=save_topk)
        label = f"True: {classes[yt]} | Pred: {classes[yp]} | Top3: " + ", ".join([f"{n}:{s:.2f}" for n, s in topk])
        canvas = img.copy()
        cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, "predictions", f"{Path(p).stem}.jpg"), canvas)


# =========================
# Main
# =========================

def main() -> None:
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Discover data")
    items = list_images(args.data_root)
    if not items:
        raise RuntimeError(f"No images found in: {args.data_root}")

    print("Limit per-class if needed")
    labels = sorted(list({c for _, c in items}))
    labels_map = {c: i for i, c in enumerate(labels)}

    print("Build splits")
    train_items, val_items, test_items = build_splits(
        items, labels_map, val_split=args.val_split, test_split=args.test_split, seed=args.seed, max_per_class=args.max_images_per_class
    )

    print("Traditional pipeline")
    trad_dir = os.path.join(args.out_dir, "traditional")
    t0 = time.time()
    trad_results, test_feats, y_test, test_paths, classes = run_traditional(
        train_items, val_items, test_items, labels_map, trad_dir, args.img_size
    )
    t1 = time.time()
    print(f"Traditional pipeline time: {(t1 - t0)/3600:.2f} h")

    print("Deep pipeline")
    deep_dir = os.path.join(args.out_dir, "deep")
    os.makedirs(deep_dir, exist_ok=True)
    dval, dtest, deep_rf = run_deep(
        train_items, val_items, test_items, labels_map, argparse.Namespace(**{**vars(args), "out_dir": deep_dir})
    )

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "traditional": trad_results,
                "deep_val": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in dval.items() if k in ["acc", "macro_f1", "kappa"]},
                "deep_test": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in dtest.items() if k in ["acc", "macro_f1", "kappa"]},
                "deep_features_rf": deep_rf,
            },
            f,
            indent=2,
        )

    visualize_predictions(dtest, classes, deep_dir, save_topk=args.save_topk)
    print("Done.")


if __name__ == "__main__":
    main()
