"""
SEVIR Storm Classifier — Full Pipeline
=======================================
Single file containing everything:
    - Data loading & preprocessing pipeline
    - ResNet-18 classifier adapted for radar
    - Weighted loss for class imbalance
    - Training loop with early stopping
    - Test evaluation with confusion matrix + ROC-AUC curves
    - Single image inference helper

TWO TRAINING MODES (set TRAINING_MODE below):
    "with_no_storm"     — includes NO_STORM (unlabelled RANDOMEVENTS) as a class.
                          Checkpoint saved to: sevir_storm_classifier_with_no_storm.pt
    "without_no_storm"  — drops NO_STORM events entirely; classifies storm types only.
                          Checkpoint saved to: sevir_storm_classifier_without_no_storm.pt

Both checkpoints are compatible with eval_from_checkpoint.py / the RF script.

SETUP:
    pip install torch torchvision h5py numpy matplotlib scikit-learn pandas seaborn torch-directml

USAGE:
    python New_combined_fixed.py
"""

import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
import pandas as pd
import seaborn as sns
import time
import warnings
import joblib
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# ★  TRAINING MODE — change this to switch between the two model variants
# ─────────────────────────────────────────────────────────────────────────────

TRAINING_MODE = "without_no_storm"   # "with_no_storm"  |  "without_no_storm"

assert TRAINING_MODE in ("with_no_storm", "without_no_storm"), (
    f"TRAINING_MODE must be 'with_no_storm' or 'without_no_storm', got: {TRAINING_MODE!r}"
)

INCLUDE_NO_STORM = (TRAINING_MODE == "with_no_storm")

print(f"\n{'='*60}")
print(f"  TRAINING MODE : {TRAINING_MODE}")
print(f"  NO_STORM class: {'INCLUDED' if INCLUDE_NO_STORM else 'EXCLUDED'}")
print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DEVICE SETUP
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    try:
        import torch_directml
        device = torch_directml.device()
        _ = torch.zeros(1).to(device)
        print("✓ Using AMD GPU via DirectML")
        return device
    except ImportError:
        print("⚠  torch-directml not found — AMD GPU unavailable.")
    except Exception as e:
        print(f"⚠  DirectML init failed ({e}) — falling back.")

    if torch.cuda.is_available():
        print(f"✓ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    print("✗ No GPU found — using CPU (training will be slow)")
    return torch.device("cpu")


DEVICE = get_device()


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "data_dir":     "./data/sevir/vil",
    "catalog_path": "./data/sevir/CATALOG.csv",
    "channel":      "vil",
    "frame_index":  24,
    "image_size":   64,
    "max_events":   None,
    "train_frac":   0.70,
    "val_frac":     0.15,
    "test_frac":    0.15,
    "batch_size":   32,
    "num_workers":  0,
    "seed":         42,
}

_suffix = TRAINING_MODE   # e.g. "with_no_storm"

TRAIN_CONFIG = {
    "epochs":         100,
    "learning_rate":  5e-3,
    "weight_decay":   1.5e-4,
    "warmup_epochs":  5,
    "patience":       50,
    "save_path":      f"./sevir_storm_classifier_{_suffix}.pt",
}

LABEL_ENCODER_SAVE = f"./sevir_storm_classifier_{_suffix}_label_encoder.pkl"
CONFUSION_MATRIX_PATH = f"./sevir_confusion_matrix_{_suffix}.png"
ROC_AUC_PATH = f"./sevir_roc_auc_{_suffix}.png"
TRAINING_HISTORY_PATH = f"./sevir_training_history_{_suffix}.png"

STORM_TYPES = [
    "THUNDERSTORM WIND",
    "HAIL",
    "FLASH FLOOD",
    "TORNADO",
    "FLOOD",
    "FUNNEL CLOUD",
    "HEAVY RAIN",
    "LIGHTNING",
]
FALLBACK_LABEL = "NO_STORM"


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_catalog(catalog_path: str) -> pd.DataFrame:
    """
    Read SEVIR catalog CSV, assign storm type labels, and optionally
    drop NO_STORM rows depending on INCLUDE_NO_STORM.
    """
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(
            f"Catalog not found at {catalog_path}\n"
            "Download: https://raw.githubusercontent.com/MIT-AI-Accelerator/eie-sevir/master/CATALOG.csv"
        )

    df = pd.read_csv(catalog_path, low_memory=False)
    df = df[df["img_type"] == CONFIG["channel"]].copy()

    df["event_type_clean"] = (
        df["event_type"].fillna("NO_STORM").str.strip().str.upper()
    )
    df["label"] = df["event_type_clean"].apply(
        lambda x: x if x in STORM_TYPES else FALLBACK_LABEL
    )
    df["is_labelled"] = df["label"] != FALLBACK_LABEL

    print(f"Catalog loaded: {len(df)} total events")
    print(f"  Labelled (STORMEVENTS)   : {df['is_labelled'].sum()}")
    print(f"  Unlabelled (RANDOMEVENTS): {(~df['is_labelled']).sum()}")
    print("\nStorm type distribution (labelled events):")
    print(df[df["is_labelled"]]["label"].value_counts().to_string())

    if not INCLUDE_NO_STORM:
        before = len(df)
        df = df[df["is_labelled"]].copy()
        print(f"\n[Mode: without_no_storm] Dropped {before - len(df)} NO_STORM rows → {len(df)} remaining")
    else:
        print(f"\n[Mode: with_no_storm] Keeping all {len(df)} rows including NO_STORM")

    print()
    return df


def read_event(h5_path: str, event_id: str, frame_idx: int):
    with h5py.File(h5_path, "r") as f:
        channel = CONFIG["channel"]
        if channel not in f:
            return None
        ids_in_file = f["id"][:].astype(str)
        matches = np.where(ids_in_file == event_id)[0]
        if len(matches) == 0:
            return None
        return f[channel][matches[0], :, :, frame_idx].astype(np.float32)


def normalise(frame: np.ndarray) -> np.ndarray:
    return frame / 255.0


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
    return torch.nn.functional.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False
    ).squeeze().numpy()


def augment(frame: np.ndarray, is_training: bool) -> np.ndarray:
    if not is_training:
        return frame
    if np.random.rand() > 0.5:
        frame = np.fliplr(frame)
    if np.random.rand() > 0.5:
        frame = np.flipud(frame)
    frame = np.rot90(frame, k=np.random.randint(0, 4))
    return np.ascontiguousarray(frame)


class SEVIRDataset(Dataset):
    def __init__(self, catalog, data_dir, label_encoder,
                 is_training=True, image_size=128, frame_index=24, max_events=None):
        self.label_encoder = label_encoder
        self.is_training   = is_training
        self.image_size    = image_size
        self.frame_index   = frame_index

        all_h5_files = {}
        for root, _, files in os.walk(data_dir):
            for fname in files:
                if fname.endswith(".h5"):
                    all_h5_files[fname] = os.path.join(root, fname)

        print(f"Found {len(all_h5_files)} .h5 files under {data_dir}")

        self.events = []
        for _, row in catalog.iterrows():
            fname   = os.path.basename(str(row["file_name"]))
            h5_path = all_h5_files.get(fname)
            if h5_path:
                self.events.append({
                    "event_id": str(row["id"]),
                    "h5_path":  h5_path,
                    "label":    row["label"],
                })
            if max_events and len(self.events) >= max_events:
                break

        split = "Train" if is_training else "Eval"
        print(f"{split} dataset: {len(self.events)} events found on disk")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        meta  = self.events[idx]
        frame = read_event(meta["h5_path"], meta["event_id"], self.frame_index)
        if frame is None:
            frame = np.zeros((384, 384), dtype=np.float32)
        frame        = normalise(frame)
        frame        = resize_frame(frame, self.image_size)
        frame        = augment(frame, self.is_training)
        image_tensor = torch.from_numpy(frame).unsqueeze(0).float()
        label_idx    = self.label_encoder.transform([meta["label"]])[0]
        return {
            "image":      image_tensor,
            "label":      label_idx,
            "label_name": meta["label"],
            "event_id":   meta["event_id"],
        }


def build_dataloaders(config: dict):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    catalog = load_catalog(config["catalog_path"])

    le = LabelEncoder()
    le.fit(catalog["label"].unique().tolist())
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}\n")

    # Save label encoder immediately so the RF eval script can load it
    joblib.dump(le, LABEL_ENCODER_SAVE)
    print(f"Saved label encoder: {LABEL_ENCODER_SAVE}\n")

    full_dataset = SEVIRDataset(
        catalog       = catalog,
        data_dir      = config["data_dir"],
        label_encoder = le,
        is_training   = False,
        image_size    = config["image_size"],
        frame_index   = config["frame_index"],
        max_events    = config.get("max_events"),
    )

    n_total = len(full_dataset)
    n_train = int(n_total * config["train_frac"])
    n_val   = int(n_total * config["val_frac"])
    n_test  = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    train_ds.dataset.is_training = True
    print(f"Split: {n_train} train / {n_val} val / {n_test} test\n")

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=config["batch_size"], shuffle=shuffle,
                          num_workers=config["num_workers"], pin_memory=False)

    return make_loader(train_ds, True), make_loader(val_ds, False), make_loader(test_ds, False), le


def run_sanity_checks(loader, label_encoder):
    batch = next(iter(loader))
    images = batch["image"]
    labels = batch["label"]
    print("=== Sanity checks ===")
    print(f"  Image tensor shape : {images.shape}")
    print(f"  Pixel value range  : [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Unique labels seen : {labels.unique().tolist()}")
    print(f"  Classes            : {list(label_encoder.classes_)}")
    print("=== All checks passed ✓ ===\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────────────────────────────────────────

class SEVIRClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        backbone  = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        orig_conv = backbone.conv1
        new_conv  = nn.Conv2d(1, orig_conv.out_channels,
                              orig_conv.kernel_size, orig_conv.stride,
                              orig_conv.padding, bias=False)
        new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv
        self.features   = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(train_loader, num_classes, device):
    counts = torch.zeros(num_classes)
    print("Computing class weights...")
    for batch in train_loader:
        for c in range(num_classes):
            counts[c] += (batch["label"] == c).sum()
    counts  = counts.clamp(min=1)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return weights.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns loss, accuracy, predictions, true labels, and softmax scores."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_scores = [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(1)
        total_loss += loss.item() * images.size(0)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_scores.append(probs.cpu().numpy())
    all_scores = np.vstack(all_scores)
    return total_loss / total, correct / total, all_preds, all_labels, all_scores


def train(model, train_loader, val_loader, label_encoder, config, device):
    num_classes   = len(label_encoder.classes_)
    class_weights = compute_class_weights(train_loader, num_classes, device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = optim.AdamW(model.parameters(),
                                lr=config["learning_rate"],
                                weight_decay=config["weight_decay"])

    warmup_epochs  = config.get("warmup_epochs", 5)
    cosine_epochs  = config["epochs"] - warmup_epochs

    def warmup_lambda(epoch):
        return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(cosine_epochs, 1), eta_min=1e-6)

    history          = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss    = float("inf")
    patience_counter = 0

    dummy_criterion = nn.CrossEntropyLoss()   # unweighted, for val pass

    print(f"\nStarting training — up to {config['epochs']} epochs")
    print(f"  LR warmup     : {warmup_epochs} epochs")
    print(f"  Early stopping: patience={config['patience']}")
    print(f"  Save path     : {config['save_path']}\n")
    print(f"{'Epoch':>6} {'LR':>10} {'Train Loss':>12} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>9} {'Time':>7}")
    print("─" * 72)

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, dummy_criterion, device)

        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        marker = " ← best" if val_loss < best_val_loss else ""
        print(f"{epoch:>6} {current_lr:>10.2e} {train_loss:>12.4f} "
              f"{train_acc:>10.3f} {val_loss:>10.4f} {val_acc:>9.3f} "
              f"{time.time()-t0:>6.0f}s{marker}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "epoch":         epoch,
                "model_state":   model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "val_loss":      val_loss,
                "val_acc":       val_acc,
                "label_encoder": label_encoder,
                "training_mode": TRAINING_MODE,
                "config":        CONFIG,
            }, config["save_path"])
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest val loss: {best_val_loss:.4f} — saved to {config['save_path']}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 7. PLOTS: TRAINING HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training history — {TRAINING_MODE}", fontsize=13, fontweight="bold")
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train", color="#4a90d9")
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="#e85d30")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train", color="#4a90d9")
    ax2.plot(epochs, history["val_acc"],   label="Val",   color="#e85d30")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(TRAINING_HISTORY_PATH, dpi=120, bbox_inches="tight")
    print(f"Saved: {TRAINING_HISTORY_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. PLOTS: ROC-AUC
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(all_labels, all_scores, class_names, save_path):
    """
    One ROC curve per class (One-vs-Rest) + macro and micro averages.
    Saved as a grid of per-class subplots with a combined summary panel.
    """
    n_classes  = len(class_names)
    labels_bin = label_binarize(all_labels, classes=list(range(n_classes)))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], all_scores[:, i])
        roc_auc[i]        = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(labels_bin.ravel(), all_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr  = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr         /= n_classes
    fpr["macro"]      = all_fpr
    tpr["macro"]      = mean_tpr
    roc_auc["macro"]  = auc(fpr["macro"], tpr["macro"])

    cmap   = plt.cm.get_cmap("tab10", n_classes)
    colors = [cmap(i) for i in range(n_classes)]

    n_cols = 3
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig    = plt.figure(figsize=(6 * n_cols, 4 * (n_rows + 1) + 1))
    fig.suptitle(
        f"ROC-AUC Curves — SEVIR Storm Classifier ({TRAINING_MODE})",
        fontsize=15, fontweight="bold", y=0.99,
    )
    gs = fig.add_gridspec(n_rows + 1, n_cols, hspace=0.55, wspace=0.35,
                          top=0.95, bottom=0.04)

    for i, (name, color) in enumerate(zip(class_names, colors)):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(gs[row, col])
        ax.plot(fpr[i], tpr[i], color=color, lw=2, label=f"AUC = {roc_auc[i]:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

    ax_sum = fig.add_subplot(gs[n_rows, :])
    for i, (name, color) in enumerate(zip(class_names, colors)):
        ax_sum.plot(fpr[i], tpr[i], color=color, lw=1.5, alpha=0.8,
                    label=f"{name}  (AUC={roc_auc[i]:.3f})")
    ax_sum.plot(fpr["micro"], tpr["micro"], color="black", lw=2.5, linestyle=":",
                label=f"Micro-avg  (AUC={roc_auc['micro']:.3f})")
    ax_sum.plot(fpr["macro"], tpr["macro"], color="black", lw=2.5, linestyle="--",
                label=f"Macro-avg  (AUC={roc_auc['macro']:.3f})")
    ax_sum.plot([0, 1], [0, 1], "gray", lw=1, linestyle="--", alpha=0.5)
    ax_sum.set_xlim([0.0, 1.0]); ax_sum.set_ylim([0.0, 1.05])
    ax_sum.set_xlabel("False Positive Rate", fontsize=11)
    ax_sum.set_ylabel("True Positive Rate", fontsize=11)
    ax_sum.set_title("All Classes — Summary", fontsize=12, fontweight="bold")
    ax_sum.legend(loc="lower right", fontsize=8.5, ncol=2)
    ax_sum.grid(alpha=0.3)

    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"Saved: {save_path}")

    print("\n  ROC-AUC per class:")
    print(f"  {'Class':<24} AUC")
    print("  " + "-" * 34)
    for i, name in enumerate(class_names):
        print(f"  {name:<24} {roc_auc[i]:.4f}")
    print("  " + "-" * 34)
    print(f"  {'Macro average':<24} {roc_auc['macro']:.4f}")
    print(f"  {'Micro average':<24} {roc_auc['micro']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. TEST EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(model, test_loader, label_encoder, device):
    criterion   = nn.CrossEntropyLoss()
    test_loss, test_acc, all_preds, all_labels, all_scores = evaluate(
        model, test_loader, criterion, device
    )
    class_names = list(label_encoder.classes_)

    print("\n=== Test Set Results ===")
    print(f"  Loss    : {test_loss:.4f}")
    print(f"  Accuracy: {test_acc*100:.1f}%\n")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names, digits=3, zero_division=0))

    # Confusion matrix
    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"Confusion matrix — test set ({TRAINING_MODE}, normalised)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=120, bbox_inches="tight")
    print(f"Saved: {CONFUSION_MATRIX_PATH}")

    # ROC-AUC
    print("\n=== ROC-AUC Curves ===")
    plot_roc_curves(all_labels, all_scores, class_names, ROC_AUC_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# 10. INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(model, image_tensor, label_encoder, device):
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(image_tensor.unsqueeze(0).to(device)), dim=1).squeeze().cpu()
    class_names = list(label_encoder.classes_)
    top_idx     = probs.argmax().item()
    prob_dict   = {name: probs[i].item() for i, name in enumerate(class_names)}
    return class_names[top_idx], probs[top_idx].item(), prob_dict


# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(f"SEVIR Storm Classifier — {TRAINING_MODE}")
    print("=" * 60 + "\n")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, label_encoder = build_dataloaders(CONFIG)
    run_sanity_checks(train_loader, label_encoder)

    # ── Model ─────────────────────────────────────────────────────────────
    num_classes = len(label_encoder.classes_)
    model       = SEVIRClassifier(num_classes=num_classes, dropout=0.3).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ResNet-18 (radar-adapted)")
    print(f"  Parameters : {total_params:,}")
    print(f"  Classes    : {num_classes}  {list(label_encoder.classes_)}")
    print(f"  Input size : (1, {CONFIG['image_size']}, {CONFIG['image_size']})\n")

    # ── Train ─────────────────────────────────────────────────────────────
    history = train(model, train_loader, val_loader, label_encoder, TRAIN_CONFIG, DEVICE)
    plot_training_history(history)

    # ── Load best checkpoint & evaluate ───────────────────────────────────
    print("\nLoading best checkpoint for evaluation...")
    checkpoint = torch.load(TRAIN_CONFIG["save_path"], map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print(f"  Epoch {checkpoint['epoch']} — val_loss={checkpoint['val_loss']:.4f}, "
          f"val_acc={checkpoint['val_acc']:.3f}\n")

    evaluate_on_test(model, test_loader, label_encoder, DEVICE)

    # ── Example prediction ─────────────────────────────────────────────────
    print("\n=== Example prediction ===")
    sample     = next(iter(test_loader))
    image      = sample["image"][0]
    true_label = label_encoder.classes_[sample["label"][0]]

    pred_class, confidence, all_probs = predict_single(model, image, label_encoder, DEVICE)
    print(f"  True label : {true_label}")
    print(f"  Predicted  : {pred_class} ({confidence*100:.1f}% confidence)")
    print("\n  All probabilities:")
    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"    {cls:<22} {prob:.3f}  {bar}")

    print(f"\nDone. Outputs saved with suffix: _{TRAINING_MODE}")
    print(f"  Checkpoint     : {TRAIN_CONFIG['save_path']}")
    print(f"  Label encoder  : {LABEL_ENCODER_SAVE}")
    print(f"  Confusion matrix: {CONFUSION_MATRIX_PATH}")
    print(f"  ROC-AUC        : {ROC_AUC_PATH}")
    print(f"  Training history: {TRAINING_HISTORY_PATH}")