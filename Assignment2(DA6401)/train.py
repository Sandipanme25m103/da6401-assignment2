

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

    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        train_tf = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.2),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.6),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), p=0.3),
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
        model.backbone = load_pretrained_encoder(model.backbone)
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


    def __init__(self, reduction: str = "mean", image_size: float = 224.0):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction=reduction, beta=1.0)
        self.iou = IoULoss(reduction=reduction)
        self.S = image_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalise both by image_size for smooth-L1 so scale matches IoU loss
        smooth_l1_loss = self.smooth_l1(pred / self.S, target / self.S)
        iou_loss = self.iou(pred, target)
        return smooth_l1_loss + iou_loss


class DiceLoss(nn.Module):


    def __init__(self, num_classes: int = 3, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, H, W]   targets: [B, H, W] long
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
        B, C, H, W = probs.shape

        # One-hot encode targets -> [B, C, H, W]
        target_one_hot = torch.zeros_like(probs)
        valid = targets.clone()
        if self.ignore_index >= 0:
            valid = valid.clamp(0, C - 1)
        target_one_hot.scatter_(1, valid.unsqueeze(1), 1.0)
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).float().unsqueeze(1)  # [B,1,H,W]
            probs = probs * mask
            target_one_hot = target_one_hot * mask

        # Dice per class
        dims = (0, 2, 3)  # reduce over batch, H, W
        intersection = (probs * target_one_hot).sum(dim=dims)
        cardinality = (probs + target_one_hot).sum(dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class CombinedSegLoss(nn.Module):


    def __init__(self, num_classes: int = 3, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + \
               self.dice_weight * self.dice(logits, targets)


def compute_multitask_loss(outputs, labels, cls_criterion, loc_criterion, seg_criterion, args):

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

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--epochs_loc", type=int, default=None,
                        help="Override epochs for localization task")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--loc_weight", type=float, default=1.0)
    parser.add_argument("--seg_weight", type=float, default=1.0)

    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every", type=int, default=5)

    parser.add_argument("--wandb_project", type=str, default="da6401-assignment2")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()
    if args.epochs_loc is None:
        args.epochs_loc = args.epochs

    device = get_device()
    print(f"Using device: {device}")

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # Data
    train_tf, val_tf = get_transforms()
    train_ds = OxfordIIITPetDataset(root=args.data_root, split="train", transform=train_tf)
    val_ds   = OxfordIIITPetDataset(root=args.data_root, split="val",   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name_map = {
        "classification": "classifier.pth",
        "localization":   "localizer.pth",
        "segmentation":   "unet.pth",
        "multitask":      "best_model.pth",
    }

    # ── Dedicated localization training (matches proven training approach) ────
    if args.task == "localization":
        model = build_model(args.task, args).to(device)
        print(f"Model: localization | Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Differential LR: backbone at 5% of head LR (same as reference approach)
        opt = optim.AdamW([
            {"params": list(model.backbone.parameters()),        "lr": args.lr * 0.05},
            {"params": list(model.regression_head.parameters()), "lr": args.lr},
        ], weight_decay=args.weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_loc)

        iou_loss_fn   = IoULoss(reduction="mean")
        iou_loss_none = IoULoss(reduction="none")
        S = float(args.image_size)

        best_iou = 0.0
        for epoch in range(1, args.epochs_loc + 1):
            # Train
            model.train()
            tl, total = 0.0, 0
            for images, labels in train_loader:
                imgs = images.to(device)
                bb   = labels["bbox"].to(device)
                opt.zero_grad()
                pred = model(imgs)
                loss = iou_loss_fn(pred, bb) + nn.functional.smooth_l1_loss(pred / S, bb / S)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl += loss.item() * imgs.size(0)
                total += imgs.size(0)
            tl /= max(total, 1)

            # Validate — track IoU directly
            model.eval()
            vis, vt = 0.0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    imgs = images.to(device)
                    bb   = labels["bbox"].to(device)
                    pred = model(imgs)
                    vis += (1.0 - iou_loss_none(pred, bb)).sum().item()
                    vt  += imgs.size(0)
            val_iou = vis / max(vt, 1)
            sched.step()

            if not args.no_wandb:
                wandb.log({"epoch": epoch, "train/loss": tl, "val/iou": val_iou,
                           "lr": opt.param_groups[1]["lr"]})

            print(f"Epoch [{epoch:03d}/{args.epochs_loc}] "
                  f"Train Loss: {tl:.4f} | Val IoU: {val_iou:.4f}"
                  + (" ← best" if val_iou > best_iou else ""))

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), save_dir / "localizer.pth")

        print(f"\nBest Val IoU: {best_iou:.4f}")
        print(f"Checkpoint saved to: {save_dir / 'localizer.pth'}")
        if not args.no_wandb:
            wandb.finish()
        return

    
    model = build_model(args.task, args).to(device)
    print(f"Model: {args.task} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    cls_criterion = nn.CrossEntropyLoss()
    loc_criterion = LocalizationLoss(reduction="mean")
    seg_criterion = CombinedSegLoss(num_classes=args.seg_classes, ce_weight=1.0, dice_weight=1.0)
    criterions = (cls_criterion, loc_criterion, seg_criterion)

    # Encoder gets 10x smaller LR than fresh heads
    if hasattr(model, "encoder") and args.task != "multitask":
        encoder_params = list(model.encoder.parameters())
        head_params = [p for p in model.parameters()
                       if not any(p is ep for ep in encoder_params)]
        param_groups = [
            {"params": encoder_params, "lr": args.lr * 0.1},
            {"params": head_params,    "lr": args.lr},
        ]
    else:
        param_groups = model.parameters()

    optimizer = optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterions, device, args.task, args)
        val_loss, val_acc = evaluate(
            model, val_loader, criterions, device, args.task, args)
        scheduler.step()

        log_dict = {"epoch": epoch, "train/loss": train_loss,
                    "val/loss": val_loss, "lr": scheduler.get_last_lr()[0]}
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
