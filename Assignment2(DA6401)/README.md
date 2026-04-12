
A PyTorch implementation of a shared-backbone multi-task perception model
for the Oxford-IIIT Pet dataset, supporting:

- **Classification** — 37-breed classification
- **Localization** — Bounding box regression (IoU loss)
- **Segmentation** — Pixel-wise foreground/background/boundary segmentation (U-Net decoder)
- **Multi-task** — All three tasks simultaneously with a shared VGG11 encoder

- **GitHub-Link** : 
-**Wandb-Report**: https://api.wandb.ai/links/me25m103-/0jsy18sy

---

## Project Structure

```
da6401_assignment2/
├── data/
│   └── pets_dataset.py       # OxfordIIITPetDataset loader
├── losses/
│   ├── __init__.py
│   └── iou_loss.py           # Custom IoU loss for bounding box regression
├── models/
│   ├── __init__.py
│   ├── layers.py             # CustomDropout (implemented from scratch)
│   ├── vgg11.py              # VGG11Encoder with skip connection support
│   ├── classification.py     # VGG11Classifier
│   ├── localization.py       # VGG11Localizer
│   ├── segmentation.py       # VGG11UNet (U-Net decoder)
│   └── multitask.py          # MultiTaskPerceptionModel
├── train.py                  # Training entrypoint
├── inference.py              # Evaluation & inference entrypoint
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Dataset

Download the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) and place it as:

```
oxford_pets/
  images/
  annotations/
    list.txt
    trimaps/
    xmls/
```

---

## Training

**Multi-task (default):**
```bash
python train.py --task multitask --data_root ./oxford_pets --epochs 30 --batch_size 32 --lr 1e-4
```

**Classification only:**
```bash
python train.py --task classification --data_root ./oxford_pets --epochs 30
```

**Localization only:**
```bash
python train.py --task localization --data_root ./oxford_pets --epochs 30
```

**Segmentation only:**
```bash
python train.py --task segmentation --data_root ./oxford_pets --epochs 30
```

**Disable W&B logging:**
```bash
python train.py --task multitask --no_wandb
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data_root` | `./oxford_pets` | Dataset root |
| `--task` | `multitask` | Task to train |
| `--epochs` | `30` | Training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--weight_decay` | `1e-4` | L2 regularization |
| `--dropout` | `0.5` | Dropout probability |
| `--cls_weight` | `1.0` | Classification loss weight |
| `--loc_weight` | `1.0` | Localization loss weight |
| `--seg_weight` | `1.0` | Segmentation loss weight |
| `--wandb_project` | `da6401-assignment2` | W&B project name |

---

## Inference & Evaluation

```bash
python inference.py --model_path ./checkpoints/best_model.pth --task multitask --split test
```

**Output metrics:**
- Classification: Accuracy, Precision, Recall, F1-Score (macro), Confusion Matrix
- Localization: Mean IoU
- Segmentation: Pixel Accuracy, F1-Score (macro)

Results are saved to `./results/`.

---

## Architecture

### VGG11 Encoder
Five convolutional blocks (1-1-2-2-2 convolutions) with BatchNorm and MaxPool, producing a bottleneck of shape `[B, 512, H/32, W/32]`. Supports `return_features=True` to expose skip connections for the U-Net decoder.

### Segmentation (VGG11UNet)
U-Net-style decoder with four `DecoderBlock` modules (transposed conv + skip concat + 2× conv), upsampling back to the original resolution.

### Custom Layers
`CustomDropout` implements inverted dropout from scratch using `torch.bernoulli`, without relying on `nn.Dropout`.

### IoU Loss
Converts `(cx, cy, w, h)` predictions to corner format, computes intersection over union, and returns `1 - IoU` as the loss.
