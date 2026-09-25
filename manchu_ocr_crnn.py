#!/usr/bin/env python3
"""Standalone Manchu OCR demo using the CRNN-mix model from HuggingFace.

Downloads `mic7ch/manchu-ocr-crnn-mix` (≈145 MB) on first run and runs the
CRNN baseline on a single image. Outputs the recognized Manchu text.

Dependencies: torch, torchvision, pillow, huggingface_hub
Install:      pip install torch torchvision pillow huggingface_hub

Usage:
  python manchu_ocr_crnn.py path/to/image.png
  python manchu_ocr_crnn.py path/to/image.png --device cpu
  python manchu_ocr_crnn.py path/to/image.png --repo mic7ch/manchu-ocr-crnn-step1-real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download


# --- CRNN architecture (copied verbatim from src/CRNN/model.py) -------------

class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=512, dropout=0.1):
        super().__init__()

        def block(ic, oc, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(ic, oc, k, s, p, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout * 0.5),
            )

        self.cnn = nn.Sequential(
            block(3, 64),
            block(64, 64),
            nn.MaxPool2d(2, 2),
            block(64, 128),
            block(128, 128),
            nn.MaxPool2d(2, 2),
            block(128, 256),
            block(256, 256),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.rnn = nn.LSTM(
            512, hidden_size, num_layers=4,
            bidirectional=True, batch_first=True, dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        f = self.cnn(x)
        f = nn.functional.adaptive_avg_pool2d(f, (1, f.shape[-1]))
        f = f.squeeze(2).permute(0, 2, 1)
        f, _ = self.rnn(f)
        f = self.dropout(f)
        return self.fc(f).permute(1, 0, 2)  # [W, B, C] for CTC


def greedy_decode(logits, idx2char):
    seqs = logits.argmax(2).cpu().numpy().T
    out = []
    for s in seqs:
        last = None
        chars = []
        for i in s:
            if i != 0 and i != last:
                chars.append(idx2char.get(int(i), ""))
            last = i
        out.append("".join(chars))
    return out


# --- Demo ------------------------------------------------------------------

def load_model(repo_id: str, device: torch.device):
    print(f"[1/3] Downloading {repo_id}/best_model.pth …", flush=True)
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="best_model.pth")

    print(f"[2/3] Loading checkpoint into memory …", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device)

    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    hidden_size = ckpt["hidden_size"]
    dropout = ckpt["dropout"]

    model = CRNN(num_classes=len(char2idx), hidden_size=hidden_size, dropout=dropout)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, idx2char


def preprocess(image_path: Path) -> torch.Tensor:
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((64, 480)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tfm(img).unsqueeze(0)  # [1, 3, 64, 480]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("image", type=Path, help="Path to a Manchu OCR image (PNG/JPG).")
    ap.add_argument("--repo", default="mic7ch/manchu-ocr-crnn-mix",
                    help="HuggingFace repo to download (default: %(default)s).")
    ap.add_argument("--device", default=None,
                    help="torch device (default: cuda if available, else cpu).")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, idx2char = load_model(args.repo, device)

    print(f"[3/3] Running OCR on {args.image} (device={device}) …", flush=True)
    x = preprocess(args.image).to(device)
    with torch.no_grad():
        logits = model(x)
    text = greedy_decode(logits, idx2char)[0]

    print()
    print(f"Recognized Manchu: {text}")


if __name__ == "__main__":
    main()
