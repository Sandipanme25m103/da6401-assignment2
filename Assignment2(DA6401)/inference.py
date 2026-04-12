"""Inference and evaluation
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt

from data.pets_dataset import OxfordIIITPetDataset
from models import (
    VGG11Classifier,
    VGG11Localizer,
    VGG11UNet,
    MultiTaskPerceptionModel,
)


# Helpers

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(task: str, args) -> torch.nn.Module:
    if task == "classification":
        return VGG11Classifier(
            num_classes=args.num_classes, in_channels=3, dropout_p=args.dropout
        )
    elif task == "localization":
        return VGG11Localizer(in_channels=3, dropout_p=args.dropout)
    elif task == "segmentation":
        return VGG11UNet(num_classes=args.seg_classes, in_channels=3)
    elif task == "multitask":
        return MultiTaskPerceptionModel(
            num_breeds=args.num_classes,
            seg_classes=args.seg_classes,
            in_channels=3,
            dropout_p=args.dropout,
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def compute_iou(pred_boxes: np.ndarray, tgt_boxes: np.ndarray, eps: float = 1e-6) -> float:
    """Compute mean IoU between predicted and target boxes (cx,cy,w,h format)."""
    px1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    py1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    px2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    py2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    tx1 = tgt_boxes[:, 0] - tgt_boxes[:, 2] / 2
    ty1 = tgt_boxes[:, 1] - tgt_boxes[:, 3] / 2
    tx2 = tgt_boxes[:, 0] + tgt_boxes[:, 2] / 2
    ty2 = tgt_boxes[:, 1] + tgt_boxes[:, 3] / 2

    ix1 = np.maximum(px1, tx1); iy1 = np.maximum(py1, ty1)
    ix2 = np.minimum(px2, tx2); iy2 = np.minimum(py2, ty2)

    iw = np.maximum(ix2 - ix1, 0)
    ih = np.maximum(iy2 - iy1, 0)
    inter = iw * ih

    pred_area = np.maximum(px2 - px1, 0) * np.maximum(py2 - py1, 0)
    tgt_area  = np.maximum(tx2 - tx1, 0) * np.maximum(ty2 - ty1, 0)
    union = pred_area + tgt_area - inter

    return float(np.mean(inter / (union + eps)))


def compute_pixel_accuracy(pred_masks: np.ndarray, tgt_masks: np.ndarray) -> float:
    return float(np.mean(pred_masks == tgt_masks))


def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    """Save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


# Inference loop

@torch.no_grad()
def run_inference(model, loader, device, task):
    model.eval()

    all_cls_preds, all_cls_labels = [], []
    all_loc_preds, all_loc_labels = [], []
    all_seg_preds, all_seg_labels = [], []

    for images, labels in loader:
        images = images.to(device)

        if task == "classification":
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_cls_preds.extend(preds)
            all_cls_labels.extend(labels["class_id"].numpy())

        elif task == "localization":
            boxes = model(images).cpu().numpy()
            all_loc_preds.append(boxes)
            all_loc_labels.append(labels["bbox"].numpy())

        elif task == "segmentation":
            seg_logits = model(images)
            seg_preds = seg_logits.argmax(dim=1).cpu().numpy()
            all_seg_preds.append(seg_preds.flatten())
            all_seg_labels.append(labels["mask"].numpy().flatten())

        elif task == "multitask":
            outputs = model(images)
            cls_preds = outputs["classification"].argmax(dim=1).cpu().numpy()
            all_cls_preds.extend(cls_preds)
            all_cls_labels.extend(labels["class_id"].numpy())

            boxes = outputs["localization"].cpu().numpy()
            all_loc_preds.append(boxes)
            all_loc_labels.append(labels["bbox"].numpy())

            seg_preds = outputs["segmentation"].argmax(dim=1).cpu().numpy()
            all_seg_preds.append(seg_preds.flatten())
            all_seg_labels.append(labels["mask"].numpy().flatten())

    return (
        np.array(all_cls_preds),
        np.array(all_cls_labels),
        np.concatenate(all_loc_preds, axis=0) if all_loc_preds else None,
        np.concatenate(all_loc_labels, axis=0) if all_loc_labels else None,
        np.concatenate(all_seg_preds) if all_seg_preds else None,
        np.concatenate(all_seg_labels) if all_seg_labels else None,
    )


# Main

def main():
    parser = argparse.ArgumentParser(description="DA6401 Assignment-2 Inference & Evaluation")

    parser.add_argument("--data_root", type=str, default="./oxford_pets")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model weights (.pth)")
    parser.add_argument("--task", type=str, default="multitask",
                        choices=["classification", "localization", "segmentation", "multitask"])
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--num_classes", type=int, default=37)
    parser.add_argument("--seg_classes", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save evaluation outputs")

    args = parser.parse_args()

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset 
    dataset = OxfordIIITPetDataset(root=args.data_root, split=args.split)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Evaluating on {args.split} split: {len(dataset)} samples")

    # Load model 
    model = build_model(args.task, args).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded weights from {args.model_path}")

    # Run inference
    cls_preds, cls_labels, loc_preds, loc_labels, seg_preds, seg_labels = run_inference(
        model, loader, device, args.task
    )

    # Report metrics 
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS  |  Task: {args.task}  |  Split: {args.split}")
    print("=" * 60)

    if len(cls_preds) > 0:
        acc  = accuracy_score(cls_labels, cls_preds)
        prec = precision_score(cls_labels, cls_preds, average="macro", zero_division=0)
        rec  = recall_score(cls_labels, cls_preds, average="macro", zero_division=0)
        f1   = f1_score(cls_labels, cls_preds, average="macro", zero_division=0)

        print(f"\n[Classification]")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}  (macro)")
        print(f"  Recall    : {rec:.4f}  (macro)")
        print(f"  F1-Score  : {f1:.4f}  (macro)")

        # Confusion matrix
        cm = confusion_matrix(cls_labels, cls_preds)
        plot_confusion_matrix(cm, str(output_dir / "confusion_matrix.png"))

        # Full per-class report
        report = classification_report(cls_labels, cls_preds, zero_division=0)
        report_path = output_dir / "classification_report.txt"
        report_path.write_text(report)
        print(f"  Per-class report saved to {report_path}")

    if loc_preds is not None:
        mean_iou = compute_iou(loc_preds, loc_labels)
        print(f"\n[Localization]")
        print(f"  Mean IoU  : {mean_iou:.4f}")

    if seg_preds is not None:
        pix_acc = compute_pixel_accuracy(seg_preds, seg_labels)
        seg_f1  = f1_score(seg_labels, seg_preds, average="macro", zero_division=0)
        print(f"\n[Segmentation]")
        print(f"  Pixel Accuracy : {pix_acc:.4f}")
        print(f"  F1-Score (macro): {seg_f1:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
