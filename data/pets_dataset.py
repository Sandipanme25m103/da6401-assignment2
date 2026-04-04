'''Dataset skeleton for Oxford-IIIT Pet.'''

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Loads images with their corresponding breed labels, bounding box annotations,
    and pixel-wise segmentation masks (trimap: 1=foreground, 2=background, 3=boundary).

    Expected directory layout::

        root/
          images/
            Abyssinian_1.jpg
            ...
          annotations/
            list.txt          # metadata: image_id, class_id, species, breed_id
            trimaps/
              Abyssinian_1.png
              ...
            xmls/
              Abyssinian_1.xml  # bounding box in Pascal VOC format

    Args:
        root: path to the dataset root directory.
        split: one of 'train', 'val', or 'test'.
        transform: optional transforms applied to the PIL image.
        target_transform: optional transforms applied to the segmentation mask.
        image_size: (H, W) to resize images and masks to. Default (224, 224).
        download: placeholder for future auto-download support.

    Returns (per __getitem__):
        image:  FloatTensor [3, H, W] (normalized to [0,1] or by transform).
        labels: dict with keys:
                  'class_id'    int64 scalar (0-indexed breed label).
                  'bbox'        FloatTensor [4] (x_c, y_c, w, h) normalized to [0,1].
                  'mask'        LongTensor  [H, W] with values 0=bg, 1=fg, 2=boundary.
    """

    # Breed name -> 0-indexed class id mapping derived from list.txt ordering
    NUM_CLASSES = 37

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (224, 224),
        download: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size

        self.images_dir = self.root / "images"
        self.trimaps_dir = self.root / "annotations" / "trimaps"
        self.xmls_dir = self.root / "annotations" / "xmls"
        self.list_file = self.root / "annotations" / "list.txt"

        self._samples: List[Dict] = self._load_split()

    # Internal helpers

    def _load_split(self) -> List[Dict]:
        """Parse list.txt and return metadata for the requested split.

        list.txt format (lines starting with '#' are comments):
            image_id  class_id  species  breed_id
        class_id is 1-indexed; we convert to 0-indexed.
        """
        if not self.list_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {self.list_file}. "
                "Please download the Oxford-IIIT Pet dataset."
            )

        samples = []
        with open(self.list_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                image_id  = parts[0]
                class_id  = int(parts[1]) - 1   # 0-indexed

                samples.append({"image_id": image_id, "class_id": class_id})

        # Deterministic stratified-style split: shuffle with fixed seed so all
        # breeds are proportionally represented across train / val / test.

        import random as _random
        rng = _random.Random(42)
        rng.shuffle(samples)
        n = len(samples)
        if self.split == "train":
            return samples[: int(0.8 * n)]
        elif self.split == "val":
            return samples[int(0.8 * n) : int(0.9 * n)]
        elif self.split == "test":
            return samples[int(0.9 * n):]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{self.split}'")

    def _load_image(self, image_id: str) -> Image.Image:
        path = self.images_dir / f"{image_id}.jpg"
        img = Image.open(path).convert("RGB")
        return img.resize(self.image_size, Image.BILINEAR)

    def _load_mask(self, image_id: str) -> torch.Tensor:
        """Load trimap mask and convert to 0-indexed long tensor.

        Trimap values: 1=foreground, 2=background, 3=boundary.
        We subtract 1 to get 0=foreground, 1=background, 2=boundary.
        """
        path = self.trimaps_dir / f"{image_id}.png"
        if not path.exists():
            # Return a dummy all-background mask if file is missing
            return torch.zeros(self.image_size, dtype=torch.long)
        mask = Image.open(path).convert("L")
        mask = mask.resize(self.image_size, Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask)).long() - 1  # 0-indexed
        return mask_tensor

    def _load_bbox(self, image_id: str, img_w: int, img_h: int) -> torch.Tensor:
        """Load bounding box from Pascal VOC XML.

        Returns (x_center, y_center, width, height) normalized to [0, 1]
        relative to the original image dimensions.
        Falls back to a full-image box (0.5, 0.5, 1.0, 1.0) if XML is missing.
        """
        xml_path = self.xmls_dir / f"{image_id}.xml"
        if not xml_path.exists():
            # Default: full-image box in normalized space
            return torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)

        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            orig_w = int(size.find("width").text)
            orig_h = int(size.find("height").text)
            bndbox = root.find(".//bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except Exception:
            return torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)

        # Normalize to [0, 1] using original image dimensions
        scale_x = self.image_size[1] / orig_w
        scale_y = self.image_size[0] / orig_h
        x_c = ((xmin + xmax) / 2.0) * scale_x
        y_c = ((ymin + ymax) / 2.0) * scale_y
        w   = (xmax - xmin) * scale_x
        h   = (ymax - ymin) * scale_y

        return torch.tensor([x_c, y_c, w, h], dtype=torch.float32)

    # Dataset interface

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return image and multi-task labels for sample at index idx.

        Args:
            idx: sample index.

        Returns:
            Tuple of:
              image  - FloatTensor [3, H, W].
              labels - dict with 'class_id' (scalar LongTensor),
                                 'bbox'     ([4] FloatTensor),
                                 'mask'     ([H, W] LongTensor).
        """
        sample = self._samples[idx]
        image_id = sample["image_id"]
        class_id = sample["class_id"]

        # Load image
        img = self._load_image(image_id)
        img_w, img_h = self.image_size[1], self.image_size[0]

        # Apply optional image transform
        if self.transform is not None:
            # albumentations requires named argument: transform(image=np_array)
            img_np = np.array(img)
            result = self.transform(image=img_np)
            img = result["image"]
        else:
            # Default: convert PIL to [0,1] float tensor
            img_np = np.array(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(img_np).permute(2, 0, 1)  # HWC -> CHW

        # Load annotations
        mask = self._load_mask(image_id)
        bbox = self._load_bbox(image_id, img_w=img_w, img_h=img_h)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        labels = {
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "bbox": bbox,
            "mask": mask,
        }

        return img, labels
