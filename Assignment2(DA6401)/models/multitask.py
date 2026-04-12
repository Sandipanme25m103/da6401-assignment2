"""Unified multi-task model - loads pretrained checkpoints for each head."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg11 import VGG11
from .layers import CustomDropout
from .segmentation import DecoderBlock


class MultiTaskPerceptionModel(nn.Module):


    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        # ── Shared encoder ─────────────────────────────────────────────────────
        self.encoder = VGG11(num_classes=1000, in_channels=in_channels, dropout_p=dropout_p)

        # ── Classification head ────────────────────────────────────────────────
        self.cls_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.cls_flatten = nn.Flatten()
        self.cls_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_breeds),
        )

        # ── Localization head ──────────────────────────────────────────────────
        self.loc_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.loc_flatten = nn.Flatten()
        self.loc_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
            nn.Sigmoid(),
        )

        # ── Segmentation decoder ───────────────────────────────────────────────
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=512, out_channels=256)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=256, out_channels=128)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=128, out_channels=64)
        self.dec1 = DecoderBlock(in_channels=64,  skip_channels=64,  out_channels=32)
        self.seg_final = nn.Conv2d(32, seg_classes, kernel_size=1)

        # ── Load pretrained checkpoints if available ───────────────────────────
        self._load_checkpoints()

    def _load_checkpoints(self):
        """Download and load saved checkpoints using gdown."""
        import gdown

        os.makedirs("checkpoints", exist_ok=True)

        cls_ckpt = os.path.join("checkpoints", "classifier.pth")
        loc_ckpt = os.path.join("checkpoints", "localizer.pth")
        seg_ckpt = os.path.join("checkpoints", "unet.pth")

        # Download from Google Drive if not already present
        if not os.path.exists(cls_ckpt):
            gdown.download(id="1KyDuRxgon4J9F4ns_KYJeIGIphuQ75H2", output=cls_ckpt, quiet=False)
        if not os.path.exists(loc_ckpt):
            gdown.download(id="1Ffcb6s8VwUak3ssso6qB8Qxw7YTDG6eJ", output=loc_ckpt, quiet=False)
        if not os.path.exists(seg_ckpt):
            gdown.download(id="1rky74w29AXNsP5MpOOjCu8hvEYKlDKnR", output=seg_ckpt, quiet=False)

        # Load classifier checkpoint 
        if os.path.exists(cls_ckpt):
            state = torch.load(cls_ckpt, map_location="cpu")
            enc_state = {k[len("encoder."):]: v
                         for k, v in state.items() if k.startswith("encoder.")}
            if enc_state:
                self.encoder.load_state_dict(enc_state, strict=False)
            cls_state = {k[len("head.classifier."):]: v
                         for k, v in state.items() if k.startswith("head.classifier.")}
            if cls_state:
                self.cls_head.load_state_dict(cls_state, strict=False)

        # Load localizer checkpoint 
        if os.path.exists(loc_ckpt):
            state = torch.load(loc_ckpt, map_location="cpu")
            # Support two checkpoint formats:
            #   1) keys like "regression_head.*"  (new format, backbone+regression_head)
            #   2) keys like "head.regressor.*"   (old format)
            loc_state = {k[len("regression_head."):]: v
                         for k, v in state.items() if k.startswith("regression_head.")}
            if not loc_state:
                loc_state = {k[len("head.regressor."):]: v
                             for k, v in state.items() if k.startswith("head.regressor.")}
            if loc_state:
                self.loc_head.load_state_dict(loc_state, strict=False)

        # ── Load unet checkpoint ───────────────────────────────────────────────
        if os.path.exists(seg_ckpt):
            state = torch.load(seg_ckpt, map_location="cpu")
            seg_state = {}
            for k, v in state.items():
                if k.startswith("dec") or k.startswith("final_conv"):
                    new_k = k.replace("final_conv.", "seg_final.")
                    seg_state[new_k] = v
            if seg_state:
                self.load_state_dict(seg_state, strict=False)

    def forward(self, x: torch.Tensor):

        bottleneck, features = self.encoder(x, return_features=True)

        # Classification
        cls_feat = self.cls_flatten(self.cls_pool(bottleneck))
        cls_logits = self.cls_head(cls_feat)

        # Localization
        loc_feat = self.loc_flatten(self.loc_pool(bottleneck))
        # Sigmoid [0,1] → pixel space [0, 224] to match dataset bbox labels
        boxes = self.loc_head(loc_feat) * 224.0

        # Segmentation
        d4 = self.dec4(bottleneck, features["block4"])
        d3 = self.dec3(d4, features["block3"])
        d2 = self.dec2(d3, features["block2"])
        d1 = self.dec1(d2, features["block1"])
        seg_logits = self.seg_final(d1)
        if seg_logits.shape[2:] != x.shape[2:]:
            seg_logits = F.interpolate(
                seg_logits, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        return {
            "classification": cls_logits,
            "localization":   boxes,
            "segmentation":   seg_logits,
        }
