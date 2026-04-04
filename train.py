"""Training entrypoint Assignment-2."""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models import (
    VGG11Classifier,
    VGG11Localizer,
    VGG11UNet,
    MultiTaskPerceptionModel,
)
from losses import IoULoss


# Helpers 

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    """Return albumentations transforms (normalized input as required by assignment)."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        train_tf = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        val_tf = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        return train_tf, val_tf
    except ImportError:
        return None, None


def load_pretrained_encoder(encoder):
    """Load ImageNet pretrained VGG11-BN weights into the encoder."""
    try:
        import torchvision.models as tv
        vgg11_bn = tv.vgg11_bn(weights="IMAGENET1K_V1")
        pretrained_features = list(vgg11_bn.features.children())
        
        # Map torchvision VGG11-BN layers to our block structure
        # torchvision order: Conv, BN, ReLU, MaxPool per block
        block_map = {
            "block1": pretrained_features[0:4],    # conv+bn+relu+pool
            "block2": pretrained_features[4:8],    # conv+bn+relu+pool
            "block3": pretrained_features[8:15],   # 2x(conv+bn+relu)+pool
            "block4": pretrained_features[15:22],  # 2x(conv+bn+relu)+pool
            "block5": pretrained_features[22:29],  # 2x(conv+bn+relu)+pool
        }

        loaded = 0
        for block_name, pt_layers in block_map.items():
            our_block = getattr(encoder, block_name)
            our_layers = list(our_block.children())
            for our_layer, pt_layer in zip(our_layers, pt_layers):
                if hasattr(our_layer, "weight") and hasattr(pt_layer, "weight"):
                    if our_layer.weight.shape == pt_layer.weight.shape:
                        our_layer.weight.data.copy_(pt_layer.weight.data)
                        if our_layer.bias is not None and pt_layer.bias is not None:
                            our_layer.bias.data.copy_(pt_layer.bias.data)
                        loaded += 1
                # BatchNorm
                if isinstance(our_layer, torch.nn.BatchNorm2d) and isinstance(pt_layer, torch.nn.BatchNorm2d):
                    our_layer.weight.data.copy_(pt_layer.weight.data)
                    our_layer.bias.data.copy_(pt_layer.bias.data)
                    our_layer.running_mean.copy_(pt_layer.running_mean)
                    our_layer.running_var.copy_(pt_layer.running_var)
                    loaded += 1

        print(f"✓ Loaded {loaded} pretrained ImageNet layers into encoder")
    except Exception as e:
        print(f"⚠ Could not load pretrained weights: {e} — training from scratch")

    return encoder


def build_model(task: str, args) -> nn.Module:
    """Instantiate the model for the requested task."""
    if task == "classification":
        model = VGG11Classifier(
            num_classes=args.num_classes,
            in_channels=3,
            dropout_p=args.dropout,
        )
        model.encoder = load_pretrained_encoder(model.encoder)
        return model
    elif task == "localization":
        model = VGG11Localizer(in_channels=3, dropout_p=args.dropout)
        model.encoder = load_pretrained_encoder(model.encoder)
        return model
    elif task == "segmentation":
        model = VGG11UNet(num_classes=args.seg_classes, in_channels=3)
        model.encoder = load_pretrained_encoder(model.encoder)
        return model
    elif task == "multitask":
        # Multitask loads from checkpoints — no need for pretrained
        return MultiTaskPerceptionModel(
            num_breeds=args.num_classes,
            seg_classes=args.seg_classes,
            in_channels=3,
            dropout_p=args.dropout,
        )
    else:
        raise ValueError(f"Unknown task: {task}")


class LocalizationLoss(nn.Module):
    """Combined MSE + IoU loss for localization (both in pixel space).
    
    MSE operates on raw pixel values.
    IoU loss expects pixel-space (cx, cy, w, h).
    Total = MSE + IoU, both naturally in reasonable ranges.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.iou = IoULoss(reduction=reduction)
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # inputs are already [0,1] normalized
        mse_loss = self.mse(pred/224.0, target/224.0)
        iou_loss = self.iou(pred, target)
        return mse_loss + iou_loss


def compute_multitask_loss(outputs, labels, cls_criterion, loc_criterion, seg_criterion, args):
    """Compute the combined multi-task loss."""
    cls_loss = cls_criterion(outputs["classification"], labels["class_id"])
    loc_loss = loc_criterion(outputs["localization"], labels["bbox"])
    seg_loss = seg_criterion(outputs["segmentation"], labels["mask"])
    total = args.cls_weight * cls_loss + args.loc_weight * loc_loss + args.seg_weight * seg_loss
    return total, cls_loss, loc_loss, seg_loss


# Training loop 

def train_one_epoch(model, loader, optimizer, criterions, device, task, args):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    cls_criterion, loc_criterion, seg_criterion = criterions

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        optimizer.zero_grad()

        if task == "classification":
            logits = model(images)
            loss = cls_criterion(logits, labels["class_id"])
            preds = logits.argmax(dim=1)
            correct += (preds == labels["class_id"]).sum().item()
            total += images.size(0)

        elif task == "localization":
            boxes = model(images)
            loss = loc_criterion(boxes, labels["bbox"])

        elif task == "segmentation":
            seg_logits = model(images)
            loss = seg_criterion(seg_logits, labels["mask"])

        elif task == "multitask":
            outputs = model(images)
            loss, cls_l, loc_l, seg_l = compute_multitask_loss(
                outputs, labels, cls_criterion, loc_criterion, seg_criterion, args
            )
            preds = outputs["classification"].argmax(dim=1)
            correct += (preds == labels["class_id"]).sum().item()
            total += images.size(0)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterions, device, task, args):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    cls_criterion, loc_criterion, seg_criterion = criterions

    for images, labels in loader:
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        if task == "classification":
            logits = model(images)
            loss = cls_criterion(logits, labels["class_id"])
            preds = logits.argmax(dim=1)
            correct += (preds == labels["class_id"]).sum().item()
            total += images.size(0)

        elif task == "localization":
            boxes = model(images)
            loss = loc_criterion(boxes, labels["bbox"])

        elif task == "segmentation":
            seg_logits = model(images)
            loss = seg_criterion(seg_logits, labels["mask"])

        elif task == "multitask":
            outputs = model(images)
            loss, *_ = compute_multitask_loss(
                outputs, labels, cls_criterion, loc_criterion, seg_criterion, args
            )
            preds = outputs["classification"].argmax(dim=1)
            correct += (preds == labels["class_id"]).sum().item()
            total += images.size(0)

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


#  Main 

def main():
    parser = argparse.ArgumentParser(description="DA6401 Assignment-2 Training")

    parser.add_argument("--data_root", type=str, default="./oxford_pets")
    parser.add_argument("--task", type=str, default="multitask",
                        choices=["classification", "localization", "segmentation", "multitask"])
    parser.add_argument("--num_classes", type=int, default=37)
    parser.add_argument("--seg_classes", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--loc_weight", type=float, default=1.0)
    parser.add_argument("--seg_weight", type=float, default=1.0)

    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every", type=int, default=5)

    parser.add_argument("--wandb_project", type=str, default="da6401-assignment2")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # Data
    train_tf, val_tf = get_transforms()
    train_ds = OxfordIIITPetDataset(root=args.data_root, split="train", transform=train_tf)
    val_ds   = OxfordIIITPetDataset(root=args.data_root, split="val",   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Model
    model = build_model(args.task, args).to(device)
    print(f"Model: {args.task} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss functions
    cls_criterion = nn.CrossEntropyLoss()
    loc_criterion = LocalizationLoss(reduction="mean")   # MSE + IoU
    seg_criterion = nn.CrossEntropyLoss()
    criterions = (cls_criterion, loc_criterion, seg_criterion)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterions, device, args.task, args
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterions, device, args.task, args
        )
        scheduler.step()

        log_dict = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
        }
        if args.task in ("classification", "multitask"):
            log_dict["train/acc"] = train_acc
            log_dict["val/acc"]   = val_acc

        if not args.no_wandb:
            wandb.log(log_dict)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            + (f" | Val Acc: {val_acc:.4f}" if args.task in ("classification", "multitask") else "")
        )

        # Save best model with task-specific name for multitask loading
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_name_map = {
                "classification": "classifier.pth",
                "localization":   "localizer.pth",
                "segmentation":   "unet.pth",
                "multitask":      "best_model.pth",
            }
            torch.save(model.state_dict(), save_dir / ckpt_name_map[args.task])
            print(f"  New best model saved (val_loss={val_loss:.4f})")

        if epoch % args.save_every == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                save_dir / f"checkpoint_epoch{epoch:03d}.pth",
            )

    if not args.no_wandb:
        wandb.finish()

    print("Training complete.")


if __name__ == "__main__":
    main()
