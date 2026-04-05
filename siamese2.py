"""
Siamese U-Net v2 — feature-engineered inputs, base model only
  Encoder 1 (img):  3-ch [hh, hv, hhhv]  from image bands 0,1
  Encoder 2 (aux):  3-ch [f2, f4, f6]    from aux bands 2,4,6
  Loss: IoU + BCE + Edge | pos_weight=6.4 | bs=6
  After training: inference on data/prediction → TIFFs + submission CSV
"""

import json
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
import albumentations as A
from pathlib import Path
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
AUX_DIR = BASE / 'data' / 'aux_data'
IMG_DIR = BASE / 'data' / 'image'
LABEL_DIR = BASE / 'data' / 'label'
SPLIT_DIR = BASE / 'data' / 'split'
PRED_DIR = BASE / 'data' / 'prediction'
PRED_AUX_DIR = PRED_DIR / 'aux_data'
PRED_IMG_DIR = PRED_DIR / 'image'
IMG_SIZE = 512
SEED = 42
POS_WEIGHT = 6.4

# ── Splits ───────────────────────────────────────────────────────────────────
def load_split(name):
    with open(SPLIT_DIR / f"{name}.txt") as f:
        return [l.strip() for l in f if l.strip()]

def filter_ids(ids):
    return [s for s in ids
            if (AUX_DIR / f"{s}_aux.tif").exists()
            and (IMG_DIR / f"{s}_image.tif").exists()
            and (LABEL_DIR / f"{s}_label.tif").exists()]

TRAIN_IDS = filter_ids(load_split('train') + load_split('val'))
VAL_IDS   = filter_ids(load_split('test'))
TEST_IDS  = VAL_IDS

# ── Feature engineering ───────────────────────────────────────────────────────
def encode_band6(arr):
    """80 → 3 | 40 or 60 → 1 | others → 0"""
    out = np.zeros_like(arr, dtype=np.float32)
    out[arr == 80] = 3
    out[(arr == 40) | (arr == 60)] = 1
    return out

def extract_img_features(img_path):
    """Returns [hh, hv, hhhv] as (3, H, W) float32 array."""
    with rasterio.open(img_path) as src:
        img = src.read([1, 2]).astype(np.float32)  # (2, H, W)
    img = np.where(np.isfinite(img), img, 0.0)
    hh   = np.log1p(np.clip(img[0], 0, 2000))
    hv   = np.log1p(np.clip(img[1], 0, 800))
    hhhv = np.log1p((np.clip(img[0], 0, 2000) * np.clip(img[1], 0, 800)).clip(min=0))
    return np.stack([hh, hv, hhhv], axis=0)  # (3, H, W)

def extract_aux_features(aux_path):
    """Returns [f2, f4, f6] as (3, H, W) float32 array."""
    with rasterio.open(aux_path) as src:
        b2 = src.read(2).astype(np.float32)
        b4 = src.read(4).astype(np.float32)
        b6 = src.read(6).astype(np.float32)
    b2 = np.where(np.isneginf(b2), 0.0, b2)
    b4 = np.where(np.isneginf(b4), 0.0, b4)
    f2 = np.clip(b2, 0, 5)
    f4 = np.clip(b4, 0, 80)
    f6 = encode_band6(b6)
    return np.stack([f2, f4, f6], axis=0)  # (3, H, W)

# ── Augmentation ──────────────────────────────────────────────────────────────
TRAIN_AUG = A.Compose([
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
], additional_targets={'image2': 'image'})

# ── Normalization ─────────────────────────────────────────────────────────────
def compute_stats_from_arrays(extractor, ids):
    """Compute per-channel mean/std over a list of tile IDs using a feature extractor."""
    n_ch = sums = sq = cnt = None
    for sid in ids:
        feats = extractor(sid)  # (C, H, W)
        if n_ch is None:
            n_ch = feats.shape[0]
            sums = np.zeros(n_ch, np.float64)
            sq   = np.zeros(n_ch, np.float64)
            cnt  = np.zeros(n_ch, np.float64)
        for c in range(n_ch):
            v = feats[c][np.isfinite(feats[c])]
            sums[c] += v.sum(); sq[c] += (v**2).sum(); cnt[c] += v.size
    means = (sums / cnt).astype(np.float32)
    stds  = np.sqrt(np.maximum(sq / cnt - (sums / cnt)**2, 0)).astype(np.float32)
    return means, np.where(stds < 1e-6, 1.0, stds)

print("Computing stats...")
IMG_MEANS, IMG_STDS = compute_stats_from_arrays(
    lambda sid: extract_img_features(IMG_DIR / f"{sid}_image.tif"), TRAIN_IDS)
AUX_MEANS, AUX_STDS = compute_stats_from_arrays(
    lambda sid: extract_aux_features(AUX_DIR / f"{sid}_aux.tif"), TRAIN_IDS)

# ── Training Dataset ──────────────────────────────────────────────────────────
class SiameseDataset(Dataset):
    def __init__(self, ids, augment=False):
        self.ids = ids
        self.augment = augment

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        img_feats = extract_img_features(IMG_DIR / f"{sid}_image.tif")  # (3, H, W)
        aux_feats = extract_aux_features(AUX_DIR / f"{sid}_aux.tif")    # (3, H, W)
        with rasterio.open(LABEL_DIR / f"{sid}_label.tif") as s:
            mask = (s.read(1).astype(np.float32) == 1).astype(np.float32)

        if self.augment:
            a = TRAIN_AUG(
                image=img_feats.transpose(1, 2, 0),
                image2=aux_feats.transpose(1, 2, 0),
                mask=mask,
            )
            img_feats = a['image'].transpose(2, 0, 1)
            aux_feats = a['image2'].transpose(2, 0, 1)
            mask = a['mask']

        img_feats = (img_feats - IMG_MEANS[:, None, None]) / IMG_STDS[:, None, None]
        aux_feats = (aux_feats - AUX_MEANS[:, None, None]) / AUX_STDS[:, None, None]
        return (torch.from_numpy(img_feats),
                torch.from_numpy(aux_feats),
                torch.from_numpy(mask).unsqueeze(0),
                sid)

# ── Prediction Dataset ────────────────────────────────────────────────────────
class PredictionDataset(Dataset):
    """Loads engineered features from data/prediction for inference (no labels)."""
    def __init__(self):
        self.ids = sorted([
            p.name.replace('_image.tif', '')
            for p in PRED_IMG_DIR.glob('*_image.tif')
        ])
        print(f"Found {len(self.ids)} prediction samples.")

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        img_feats = extract_img_features(PRED_IMG_DIR / f"{sid}_image.tif")
        aux_feats = extract_aux_features(PRED_AUX_DIR / f"{sid}_aux.tif")
        img_feats = (img_feats - IMG_MEANS[:, None, None]) / IMG_STDS[:, None, None]
        aux_feats = (aux_feats - AUX_MEANS[:, None, None]) / AUX_STDS[:, None, None]
        return torch.from_numpy(img_feats), torch.from_numpy(aux_feats), sid

# ── Siamese U-Net ─────────────────────────────────────────────────────────────
class SiameseUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder 1: image features [hh, hv, hhhv] — 3 channels
        self.encoder_img = smp.Unet(
            encoder_name='efficientnet-b4',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
        ).encoder
        # Encoder 2: aux features [f2, f4, f6] — 3 channels
        self.encoder_aux = smp.Unet(
            encoder_name='efficientnet-b4',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
        ).encoder

        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            fi = self.encoder_img(dummy)
            fa = self.encoder_aux(dummy)
        enc_ch = [f.shape[1] for f in fi]

        self.fuse_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(i.shape[1] + a.shape[1], i.shape[1], 1, bias=False),
                nn.BatchNorm2d(i.shape[1]),
                nn.ReLU(inplace=True),
            )
            for i, a in zip(fi, fa)
        ])

        self.decoder = UnetDecoder(
            encoder_channels=enc_ch,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_norm='batchnorm',
            attention_type=None,
        )
        self.seg_head = nn.Conv2d(16, 1, 1)

    def forward(self, img, aux):
        fi = self.encoder_img(img)
        fa = self.encoder_aux(aux)
        fused = [c(torch.cat([i, a], dim=1)) for i, a, c in zip(fi, fa, self.fuse_convs)]
        return self.seg_head(self.decoder(fused))

# ── Losses ────────────────────────────────────────────────────────────────────
class IoUBCELoss(nn.Module):
    def __init__(self, pw=POS_WEIGHT):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits).view(-1)
        t = targets.view(-1)
        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter
        return bce + 1.0 - (inter + 1) / (union + 1)

class EdgeLoss(nn.Module):
    def __init__(self, ew=5.0):
        super().__init__()
        self.ew = ew
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                           dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('laplacian', lap)

    def forward(self, logits, targets):
        lap = self.laplacian  # type: ignore[attr-defined]
        edge = (torch.abs(F.conv2d(targets, lap, padding=1)) > 0.1).float()
        w = 1.0 + edge * (self.ew - 1.0)
        return (F.binary_cross_entropy_with_logits(logits, targets, reduction='none') * w).mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = IoUBCELoss()
        self.edge = EdgeLoss()

    def forward(self, logits, targets):
        return self.main(logits, targets) + 0.5 * self.edge(logits, targets)

# ── Lightning Module ──────────────────────────────────────────────────────────
class SiameseModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.net = SiameseUNet()
        self.criterion = CombinedLoss()
        mk = dict(threshold=0.5)
        self.val_iou  = torchmetrics.JaccardIndex(task='binary', **mk)
        self.val_f1   = torchmetrics.F1Score(task='binary', **mk)
        self.val_prec = torchmetrics.Precision(task='binary', **mk)
        self.val_rec  = torchmetrics.Recall(task='binary', **mk)
        self.test_iou  = torchmetrics.JaccardIndex(task='binary', **mk)
        self.test_f1   = torchmetrics.F1Score(task='binary', **mk)
        self.test_prec = torchmetrics.Precision(task='binary', **mk)
        self.test_rec  = torchmetrics.Recall(task='binary', **mk)

    def forward(self, img, aux):
        return self.net(img, aux)

    def training_step(self, batch, batch_idx):
        img, aux, masks, _ = batch
        loss = self.criterion(self(img, aux), masks)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _eval_step(self, batch, metrics):
        img, aux, masks, _ = batch
        logits = self(img, aux)
        loss = self.criterion(logits, masks)
        probs = torch.sigmoid(logits)
        for m in metrics:
            m.update(probs, masks.int())
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._eval_step(batch, [self.val_iou, self.val_f1, self.val_prec, self.val_rec])
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        self.log('val/iou',  self.val_iou.compute(),  prog_bar=True)
        self.log('val/f1',   self.val_f1.compute())
        self.log('val/prec', self.val_prec.compute())
        self.log('val/rec',  self.val_rec.compute())
        for m in [self.val_iou, self.val_f1, self.val_prec, self.val_rec]:
            m.reset()

    def test_step(self, batch, batch_idx):
        loss = self._eval_step(batch, [self.test_iou, self.test_f1, self.test_prec, self.test_rec])
        self.log('test/loss', loss, on_epoch=True)

    def on_test_epoch_end(self):
        self.log('test/iou',  self.test_iou.compute(),  prog_bar=True)
        self.log('test/f1',   self.test_f1.compute())
        self.log('test/prec', self.test_prec.compute())
        self.log('test/rec',  self.test_rec.compute())
        for m in [self.test_iou, self.test_f1, self.test_prec, self.test_rec]:
            m.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode='max',           # because val/iou
            factor=0.5,           # aggressive drop
            patience=9,           # quick reaction
            threshold=1e-3,
            min_lr=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch", "monitor": "val/iou"}}

# ── Deployment: save predictions as TIFFs ────────────────────────────────────
def deploy(ckpt_path):
    """Load best checkpoint, run inference on data/prediction, save TIFFs."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseModule.load_from_checkpoint(ckpt_path)
    model.eval().to(device)

    pred_ds = PredictionDataset()
    pred_loader = DataLoader(pred_ds, batch_size=1, shuffle=False,
                             num_workers=4, pin_memory=True)

    PRED_DIR.mkdir(exist_ok=True)
    print(f"\nRunning inference on {len(pred_ds)} samples → {PRED_DIR}")

    with torch.no_grad():
        for img, aux, sids in pred_loader:
            img, aux = img.to(device), aux.to(device)
            logits = model(img, aux)
            pred_mask = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)

            sid = sids[0]
            out_path = PRED_DIR / f"{sid}_image.tif"

            # Copy spatial reference from source image
            with rasterio.open(PRED_IMG_DIR / f"{sid}_image.tif") as src:
                profile = src.profile.copy()

            profile.update(dtype=rasterio.uint8, count=1, compress='deflate', nodata=None)
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(pred_mask[np.newaxis, ...])

            print(f"  Saved: {out_path.name}")

    print("Inference complete.")

# ── Submission CSV ────────────────────────────────────────────────────────────
def mask_to_rle(mask):
    """
    Convert binary mask to RLE (Kaggle format).
    Mask must be 2D numpy array with values 0 or 1.
    """
    pixels = mask.flatten(order="F")  # column-major
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

def generate_submission():
    output_csv = PRED_DIR / "submission.csv"
    rows = []

    for tif_path in sorted(PRED_DIR.glob("*.tif")):
        with rasterio.open(tif_path) as src:
            mask = src.read(1)

        mask = (mask == 1).astype(np.uint8)
        rle = mask_to_rle(mask)
        rows.append({
            "id": tif_path.name.replace("_image.tif", ""),
            "rle_mask": rle,
        })

    df = pd.DataFrame(rows)
    df = df.replace("", 0).fillna(0)  # replace null/na with zero — kaggle compatible
    df.to_csv(output_csv, index=False)
    print(f"Saved Kaggle RLE CSV : {output_csv}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    pl.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision('medium')

    train_ds = SiameseDataset(TRAIN_IDS, augment=True)
    val_ds   = SiameseDataset(VAL_IDS)
    test_ds  = SiameseDataset(TEST_IDS)

    gen = torch.Generator().manual_seed(SEED)
    lkw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=6, shuffle=True, generator=gen, **lkw)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, **lkw)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, **lkw)

    model = SiameseModule()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}\n  siamese2  |  params={params:,}\n{'='*60}")

    ckpt_dir = BASE / 'data' / 'ckpt_siamese2'
    ckpt_dir.mkdir(exist_ok=True)
    logger  = TensorBoardLogger(str(BASE / 'lightning_logs'), name='siamese2')
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename='siamese2-{epoch:02d}-{val_iou:.4f}',
        monitor='val/iou', mode='max', save_top_k=1, save_last=True, verbose=True)
    early_cb = EarlyStopping(monitor='val/iou', mode='max', patience=20, verbose=True)

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1, max_epochs=100,
        callbacks=[ckpt_cb, early_cb], logger=logger,
        log_every_n_steps=1, enable_progress_bar=True, enable_model_summary=False)

    trainer.fit(model, train_loader, val_loader)

    best_val = float(ckpt_cb.best_model_score or 0)
    results  = trainer.test(model, test_loader, ckpt_path='best')
    r = results[0]

    summary = {
        'variant': 'siamese2',
        'params': params,
        'best_val_iou': round(best_val, 4),
        'test_iou':  round(r['test/iou'],  4),
        'test_f1':   round(r['test/f1'],   4),
        'test_prec': round(r['test/prec'], 4),
        'test_rec':  round(r['test/rec'],  4),
    }
    with open(BASE / 'results_siamese2.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  siamese2  |  Val IoU: {best_val:.4f}  |  Test IoU: {r['test/iou']:.4f}")
    print(f"{'='*60}")

    # ── Deploy: inference on prediction set ──────────────────────────────────
    deploy(ckpt_cb.best_model_path)

    # ── Generate Kaggle submission CSV ────────────────────────────────────────
    generate_submission()
