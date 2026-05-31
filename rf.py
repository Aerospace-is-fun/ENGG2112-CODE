"""
SEVIR Storm Classifier — Eval & RF from Checkpoint
====================================================
Loads a saved CNN checkpoint and runs:
    - Test set evaluation (confusion matrix, classification report)
    - ROC-AUC curves (per-class OvR + macro/micro average)
    - CNN feature extraction → Random Forest training & evaluation
    - Example single-image prediction

Does NOT retrain the CNN. Requires:
    sevir_storm_classifier.pt        (saved by the training script)
    sevir_storm_classifier_label_encoder.pkl   (if saved separately)
    OR re-derives the label encoder from CATALOG.csv (automatic fallback)

USAGE:
    python eval_from_checkpoint.py
"""

import os
import glob
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DEVICE
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    try:
        import torch_directml
        device = torch_directml.device()
        _ = torch.zeros(1).to(device)
        print("✓ Using AMD GPU via DirectML")
        return device
    except ImportError:
        pass
    except Exception:
        pass
    if torch.cuda.is_available():
        print(f"✓ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("✗ No GPU found — using CPU")
    return torch.device("cpu")


DEVICE = get_device()


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIG  (must match what was used during training)
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

CHECKPOINT_PATH    = "./sevir_storm_classifier.pt"
LABEL_ENCODER_PATH = "./sevir_storm_classifier_label_encoder.pkl"
RF_SAVE_PATH       = "./sevir_rf_model.pkl"

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
# 3. DATA  (same pipeline as training script)
# ─────────────────────────────────────────────────────────────────────────────

def load_catalog(catalog_path):
    df = pd.read_csv(catalog_path, low_memory=False)
    df = df[df["img_type"] == CONFIG["channel"]].copy()
    df["event_type_clean"] = (
        df["event_type"].fillna("NO_STORM").str.strip().str.upper()
    )
    df["label"] = df["event_type_clean"].apply(
        lambda x: x if x in STORM_TYPES else FALLBACK_LABEL
    )
    print(f"Catalog loaded: {len(df)} events")
    return df


def read_event(h5_path, event_id, frame_idx):
    with h5py.File(h5_path, "r") as f:
        channel = CONFIG["channel"]
        if channel not in f:
            return None
        ids = f["id"][:].astype(str)
        matches = np.where(ids == event_id)[0]
        if len(matches) == 0:
            return None
        frame = f[channel][matches[0], :, :, frame_idx]
        return frame.astype(np.float32)


def normalise(frame):
    return frame / 255.0


def resize_frame(frame, size):
    t = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
    return torch.nn.functional.interpolate(
        t, size=(size, size), mode="bilinear", align_corners=False
    ).squeeze().numpy()


class SEVIRDataset(Dataset):
    def __init__(self, catalog, data_dir, label_encoder, image_size, frame_index, max_events=None):
        self.label_encoder = label_encoder
        self.image_size    = image_size
        self.frame_index   = frame_index

        all_h5 = {}
        for root, _, files in os.walk(data_dir):
            for fname in files:
                if fname.endswith(".h5"):
                    all_h5[fname] = os.path.join(root, fname)

        print(f"Found {len(all_h5)} .h5 files")

        self.events = []
        for _, row in catalog.iterrows():
            fname   = os.path.basename(str(row["file_name"]))
            h5_path = all_h5.get(fname)
            if h5_path:
                self.events.append({
                    "event_id": str(row["id"]),
                    "h5_path":  h5_path,
                    "label":    row["label"],
                })
            if max_events and len(self.events) >= max_events:
                break

        print(f"Dataset: {len(self.events)} events on disk")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        meta  = self.events[idx]
        frame = read_event(meta["h5_path"], meta["event_id"], self.frame_index)
        if frame is None:
            frame = np.zeros((384, 384), dtype=np.float32)
        frame        = normalise(frame)
        frame        = resize_frame(frame, self.image_size)
        image_tensor = torch.from_numpy(np.ascontiguousarray(frame)).unsqueeze(0).float()
        label_idx    = self.label_encoder.transform([meta["label"]])[0]
        return {"image": image_tensor, "label": label_idx, "label_name": meta["label"], "event_id": meta["event_id"]}


def build_dataloaders(label_encoder):
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    catalog = load_catalog(CONFIG["catalog_path"])

    full_ds = SEVIRDataset(
        catalog       = catalog,
        data_dir      = CONFIG["data_dir"],
        label_encoder = label_encoder,
        image_size    = CONFIG["image_size"],
        frame_index   = CONFIG["frame_index"],
        max_events    = CONFIG.get("max_events"),
    )

    n_total = len(full_ds)
    n_train = int(n_total * CONFIG["train_frac"])
    n_val   = int(n_total * CONFIG["val_frac"])
    n_test  = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(CONFIG["seed"]),
    )
    print(f"Split: {n_train} train / {n_val} val / {n_test} test\n")

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=shuffle,
                          num_workers=CONFIG["num_workers"], pin_memory=False)

    return make_loader(train_ds, False), make_loader(val_ds, False), make_loader(test_ds, False)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL  (identical architecture to training script)
# ─────────────────────────────────────────────────────────────────────────────

class SEVIRClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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
        f = self.features(x).flatten(1)
        return self.classifier(f)


# ─────────────────────────────────────────────────────────────────────────────
# 5. LOAD CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path, label_encoder_path):
    """
    Load model weights + label encoder.

    Tries three strategies in order:
        1. Separate .pkl label encoder (cleanest)
        2. Embedded label encoder in checkpoint (old format) with safe globals
        3. weights_only=False fallback (trusted local file only)
    """
    # --- Label encoder ---
    if os.path.exists(label_encoder_path):
        label_encoder = joblib.load(label_encoder_path)
        print(f"✓ Label encoder loaded from {label_encoder_path}")
    else:
        print(f"  No separate label encoder found at {label_encoder_path}")
        print("  Will derive label encoder from catalog instead.")
        label_encoder = None   # derived below after catalog load

    # --- Checkpoint ---
    print(f"Loading checkpoint: {checkpoint_path}")

    # Strategy 1: weights_only=True with safe globals (PyTorch 2.6+)
    try:
        from sklearn.preprocessing import LabelEncoder as LE
        torch.serialization.add_safe_globals([LE])
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        print("✓ Checkpoint loaded (weights_only=True + safe globals)")
    except Exception as e1:
        print(f"  Safe load failed ({e1}), trying weights_only=False...")
        # Strategy 2: weights_only=False — fine for your own files
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            print("✓ Checkpoint loaded (weights_only=False)")
        except Exception as e2:
            raise RuntimeError(f"Could not load checkpoint: {e2}") from e2

    # Pull label encoder out of checkpoint if we didn't get one from disk
    if label_encoder is None:
        if "label_encoder" in checkpoint:
            label_encoder = checkpoint["label_encoder"]
            print("✓ Label encoder pulled from checkpoint")
        else:
            print("  No label encoder in checkpoint — rebuilding from catalog.")
            label_encoder = None   # handled in main()

    return checkpoint, label_encoder


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns loss, accuracy, predictions, true labels, and raw softmax scores."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_scores = [], [], []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        probs       = torch.softmax(logits, dim=1)
        preds       = probs.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_scores.append(probs.cpu().numpy())

    all_scores = np.vstack(all_scores)
    return total_loss / total, correct / total, all_preds, all_labels, all_scores


# ─────────────────────────────────────────────────────────────────────────────
# 7. ROC-AUC PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(all_labels, all_scores, class_names, save_path="./sevir_roc_auc.png"):
    """
    Plots one ROC curve per class (One-vs-Rest) plus macro and micro averages.

    Parameters
    ----------
    all_labels  : list[int]  — integer ground-truth labels
    all_scores  : np.ndarray — shape (N, n_classes), softmax probabilities
    class_names : list[str]  — ordered class names matching label indices
    save_path   : str        — where to save the figure
    """
    n_classes  = len(class_names)
    labels_bin = label_binarize(all_labels, classes=list(range(n_classes)))

    # ── Per-class ROC ──────────────────────────────────────────────────────
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], all_scores[:, i])
        roc_auc[i]        = auc(fpr[i], tpr[i])

    # ── Micro-average (flatten all classes) ───────────────────────────────
    fpr["micro"], tpr["micro"], _ = roc_curve(
        labels_bin.ravel(), all_scores.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # ── Macro-average (average of per-class curves) ───────────────────────
    all_fpr   = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr  = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr        /= n_classes
    fpr["macro"]     = all_fpr
    tpr["macro"]     = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # ── Colour palette ─────────────────────────────────────────────────────
    cmap   = plt.cm.get_cmap("tab10", n_classes)
    colors = [cmap(i) for i in range(n_classes)]

    # ── Figure layout: individual curves grid + summary panel ─────────────
    n_cols   = 3
    n_rows   = (n_classes + n_cols - 1) // n_cols
    fig      = plt.figure(figsize=(6 * n_cols, 4 * (n_rows + 1) + 1))
    fig.suptitle("ROC-AUC Curves — SEVIR Storm Classifier", fontsize=16, fontweight="bold", y=0.99)

    gs = fig.add_gridspec(
        n_rows + 1, n_cols,
        hspace=0.55, wspace=0.35,
        top=0.95, bottom=0.04,
    )

    # ── Per-class subplots ─────────────────────────────────────────────────
    for i, (name, color) in enumerate(zip(class_names, colors)):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(gs[row, col])
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f"AUC = {roc_auc[i]:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

    # ── Summary panel: all classes + macro/micro on one axes ──────────────
    ax_sum = fig.add_subplot(gs[n_rows, :])
    for i, (name, color) in enumerate(zip(class_names, colors)):
        ax_sum.plot(fpr[i], tpr[i], color=color, lw=1.5, alpha=0.8,
                    label=f"{name}  (AUC={roc_auc[i]:.3f})")
    ax_sum.plot(fpr["micro"], tpr["micro"],
                color="black", lw=2.5, linestyle=":",
                label=f"Micro-avg  (AUC={roc_auc['micro']:.3f})")
    ax_sum.plot(fpr["macro"], tpr["macro"],
                color="black", lw=2.5, linestyle="--",
                label=f"Macro-avg  (AUC={roc_auc['macro']:.3f})")
    ax_sum.plot([0, 1], [0, 1], "gray", lw=1, linestyle="--", alpha=0.5)
    ax_sum.set_xlim([0.0, 1.0])
    ax_sum.set_ylim([0.0, 1.05])
    ax_sum.set_xlabel("False Positive Rate", fontsize=11)
    ax_sum.set_ylabel("True Positive Rate", fontsize=11)
    ax_sum.set_title("All Classes — Summary", fontsize=12, fontweight="bold")
    ax_sum.legend(loc="lower right", fontsize=8.5, ncol=2)
    ax_sum.grid(alpha=0.3)

    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"Saved: {save_path}")

    # ── Print AUC table ────────────────────────────────────────────────────
    print("\n  ROC-AUC per class:")
    print(f"  {'Class':<24} AUC")
    print("  " + "-" * 34)
    for i, name in enumerate(class_names):
        print(f"  {name:<24} {roc_auc[i]:.4f}")
    print("  " + "-" * 34)
    print(f"  {'Macro average':<24} {roc_auc['macro']:.4f}")
    print(f"  {'Micro average':<24} {roc_auc['micro']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. TEST EVALUATION (confusion matrix + ROC)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(model, test_loader, label_encoder, device):
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, all_preds, all_labels, all_scores = evaluate(
        model, test_loader, criterion, device
    )
    class_names = list(label_encoder.classes_)

    print("\n=== Test Set Results ===")
    print(f"  Loss    : {test_loss:.4f}")
    print(f"  Accuracy: {test_acc*100:.1f}%\n")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3, zero_division=0))

    # ── Confusion matrix ───────────────────────────────────────────────────
    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Confusion matrix — test set (normalised)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("./sevir_confusion_matrix.png", dpi=120, bbox_inches="tight")
    print("Saved: sevir_confusion_matrix.png")

    # ── ROC-AUC curves ─────────────────────────────────────────────────────
    print("\n=== ROC-AUC Curves ===")
    plot_roc_curves(all_labels, all_scores, class_names)


# ─────────────────────────────────────────────────────────────────────────────
# 9. CNN FEATURES → RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_cnn_features(model, loader, device):
    model.eval()
    all_features, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].cpu().numpy()
        features = model.features(images).flatten(1)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels)
    return np.vstack(all_features), np.concatenate(all_labels)


def train_random_forest(model, train_loader, test_loader, label_encoder, device):
    print("\nExtracting CNN features for Random Forest...")
    X_train, y_train = extract_cnn_features(model, train_loader, device)
    X_test,  y_test  = extract_cnn_features(model, test_loader,  device)
    print(f"  Train features: {X_train.shape}")
    print(f"  Test features : {X_test.shape}")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    print("Training Random Forest...")
    rf.fit(X_train, y_train)
    joblib.dump(rf, RF_SAVE_PATH)
    print(f"Saved: {RF_SAVE_PATH}")

    y_pred    = rf.predict(X_test)
    y_scores  = rf.predict_proba(X_test)
    class_names = list(label_encoder.classes_)

    print("\n=== CNN Features + Random Forest Results ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # ── ROC-AUC for RF ────────────────────────────────────────────────────
    print("\n=== Random Forest ROC-AUC Curves ===")
    plot_roc_curves(y_test, y_scores, class_names, save_path="./sevir_rf_roc_auc.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. SINGLE-IMAGE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(model, image_tensor, label_encoder, device):
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        probs  = torch.softmax(logits, dim=1).squeeze().cpu()
    class_names = list(label_encoder.classes_)
    top_idx     = probs.argmax().item()
    prob_dict   = {name: probs[i].item() for i, name in enumerate(class_names)}
    return class_names[top_idx], probs[top_idx].item(), prob_dict


# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SEVIR — Eval & RF from Checkpoint (no CNN retraining)")
    print("=" * 60 + "\n")

    # ── Load checkpoint + label encoder ───────────────────────
    checkpoint, label_encoder = load_checkpoint(CHECKPOINT_PATH, LABEL_ENCODER_PATH)

    # If no label encoder was found anywhere, rebuild it from the catalog
    if label_encoder is None:
        catalog = load_catalog(CONFIG["catalog_path"])
        label_encoder = LabelEncoder()
        label_encoder.fit(catalog["label"].unique().tolist())
        print(f"✓ Label encoder rebuilt from catalog ({len(label_encoder.classes_)} classes)")

    print(f"Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}\n")

    # ── Build model and load weights ──────────────────────────
    num_classes = len(label_encoder.classes_)
    model       = SEVIRClassifier(num_classes=num_classes, dropout=0.3).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    saved_epoch = checkpoint.get("epoch", "?")
    saved_val   = checkpoint.get("val_loss", float("nan"))
    saved_acc   = checkpoint.get("val_acc", float("nan"))
    print(f"✓ Model weights loaded — epoch {saved_epoch}, val_loss={saved_val:.4f}, val_acc={saved_acc:.3f}\n")

    # ── Rebuild dataloaders with same seed/split ───────────────
    # This reproduces the exact same train/val/test split as training
    train_loader, val_loader, test_loader = build_dataloaders(label_encoder)

    # ── CNN test evaluation + ROC-AUC ─────────────────────────
    evaluate_on_test(model, test_loader, label_encoder, DEVICE)

    # ── CNN features → Random Forest + RF ROC-AUC ─────────────
    train_random_forest(model, train_loader, test_loader, label_encoder, DEVICE)

    # ── Example prediction ─────────────────────────────────────
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

    print("\nDone.")