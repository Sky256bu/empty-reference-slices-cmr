#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
74_train_seg_students.py -- A/B/C three-arm LV segmentation students (controlled comparison)
================================================================================
Load-bearing experiment: can a student trained on "2 anchor slices per phase + propagated
pseudo-labels" approach a student trained on "dense annotation of every slice"?
  The three arms share **architecture (resnet34 UNet) and recipe**, differing only in the
  label source (read from the manifest exported by 73):
    A  full-dense : GT on every slice                      -> ceiling (annotate everything)
    B  2anchor    : GT on anchor slices + pseudo_fwd on interior slices + empty GT outside the span
                                                           -> our method (2 slices annotated per phase)
    C  sparse     : GT on anchor slices only (is_anchor==1) -> lower bound (2 slices alone are not enough)
  What the headline needs: B~A (approaching full supervision) and B>>C (propagation is not dispensable).

Data comes from 73_export_lv_pseudo.py:
  {mroot}/manifest_train.csv + {mroot}/train/{imgs,gts,pf,pb}/*.png
train/val are split 80/20 at patient level (ed/es of the same patient stay on the same side, to
prevent leakage); val is always evaluated against the true GT.
Final test evaluation lives in 75_eval_seg_and_ef.py (this script only trains, stores weights and
prints val Dice).

The UNet/loss structure is taken over from train_student.py (already working) and kept
self-contained to avoid container churn.
Dependencies: torch/torchvision/timm/opencv/numpy/pandas. torch 2.12 new AMP API.

Usage (run once per arm, with the same fixed seed):
    cd /root/autodl-tmp/sam3-main
    for ARM in A B C; do
      python 74_train_seg_students.py --mroot /root/autodl-tmp/acdc/seg_lv \
          --arm $ARM --out_dir /root/autodl-tmp/acdc/seg_lv/ckpt \
          --encoder resnet34 --epochs 120 --patience 15 --bs 16 --seed 0
    done
Products: {out_dir}/student_{A,B,C}.pt  + the best val Dice (LV-present) per arm printed at the end.
================================================================================
"""
import argparse, os, random, json, hashlib
import cv2, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm


def log(m): print(m, flush=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- model (taken over from train_student.py) ----------
class ConvBNAct(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False),
            nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNAct(out_ch, out_ch)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], 1)
        return self.conv2(self.conv1(x))


class UNet(nn.Module):
    def __init__(self, encoder_name="resnet34", in_chans=1, num_classes=1, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name, features_only=True, pretrained=pretrained, in_chans=in_chans)
        enc_chs = self.encoder.feature_info.channels()
        n_skip = len(enc_chs) - 1
        dec_chs = [256, 128, 64, 32, 16][:n_skip]
        skips = enc_chs[:-1][::-1]
        self.blocks = nn.ModuleList()
        in_ch = enc_chs[-1]
        for skip_ch, out_ch in zip(skips, dec_chs):
            self.blocks.append(DecoderBlock(in_ch, skip_ch, out_ch)); in_ch = out_ch
        self.head = nn.Conv2d(in_ch, num_classes, 1)
    def forward(self, x):
        H, W = x.shape[-2:]
        feats = self.encoder(x)
        f = feats[-1]; skips = feats[:-1][::-1]
        for blk, skip in zip(self.blocks, skips): f = blk(f, skip)
        f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
        return self.head(f)


def soft_dice_loss(logits, t, eps=1e-6):
    p = torch.sigmoid(logits); d = (1, 2, 3)
    inter = (p * t).sum(d); den = p.sum(d) + t.sum(d)
    return 1.0 - (2 * inter + eps) / (den + eps)


def seg_loss(logits, t):
    dice = soft_dice_loss(logits, t)
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none").mean((1, 2, 3))
    return (dice + bce).mean()


def dice_np(pred, gt):
    pred = (pred > 0).astype(np.float32); gt = (gt > 0).astype(np.float32)
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else float(2.0 * (pred * gt).sum() / s)


# ---------- data ----------
def label_path_for_arm(row, arm):
    """Return the relative png path of the label for this row (slice) under the given arm, or None (the arm does not use this row)."""
    if arm == "A":
        return row["gt"]
    if arm == "C":
        return row["gt"] if int(row["is_anchor"]) == 1 else None
    if arm == "B":
        if int(row["is_anchor"]) == 1 or int(row["in_lvspan"]) == 0:
            return row["gt"]                       # anchors, or outside the LV span (empty): use GT
        pf = str(row.get("pf", "") or "")
        return pf if pf else row["gt"]             # interior slices use the fwd pseudo-label (falling back to GT when missing)
    if arm == "I":                                 # SDF interpolation densification control
        if int(row["is_anchor"]) == 1 or int(row["in_lvspan"]) == 0:
            return row["gt"]
        return str(row["img"]).replace("imgs/", "pi/", 1)   # same naming convention as 73b
    raise ValueError(arm)


class SliceDS(Dataset):
    def __init__(self, rows, root_split, aug=False):
        self.rows = rows; self.root = root_split; self.aug = aug
    def __len__(self): return len(self.rows)
    def _load(self, rel):
        p = os.path.join(self.root, rel)
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(f"cannot read label/image: {p}")
        return im
    def __getitem__(self, i):
        img_rel, lab_rel = self.rows[i]
        img = self._load(img_rel).astype(np.float32) / 255.0
        lab = (self._load(lab_rel) > 127).astype(np.float32)
        if self.aug:
            if random.random() < 0.5:
                img = img[:, ::-1].copy(); lab = lab[:, ::-1].copy()
            if random.random() < 0.7:
                H, W = img.shape
                ang = random.uniform(-10, 10); sc = random.uniform(0.9, 1.1)
                M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, sc)
                M[0, 2] += random.uniform(-8, 8); M[1, 2] += random.uniform(-8, 8)
                img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
                lab = cv2.warpAffine(lab, M, (W, H), flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT)
        return (torch.from_numpy(img)[None], torch.from_numpy(lab)[None])


def build_rows(df, root_split, arm):
    out = []
    for _, r in df.iterrows():
        lab = label_path_for_arm(r, arm)
        if lab is None:
            continue
        out.append((r["img"], lab))
    return out


@torch.no_grad()
def val_dice(model, df_val, root_split, device):
    """val always uses the true GT; Dice is computed only on LV-present (gt_px>0) slices (the meaningful figure)."""
    model.eval()
    ds, ns = [], []
    for _, r in df_val.iterrows():
        img = cv2.imread(os.path.join(root_split, r["img"]), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(root_split, r["gt"]), cv2.IMREAD_GRAYSCALE) > 127
        x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].to(device)
        p = torch.sigmoid(model(x))[0, 0].cpu().numpy() > 0.5
        d = dice_np(p, gt)
        if gt.sum() > 0:
            ds.append(d)
        else:
            ns.append(1.0 if p.sum() == 0 else 0.0)
    return float(np.mean(ds)) if ds else 0.0, float(np.mean(ns)) if ns else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mroot", default="/root/autodl-tmp/acdc/seg_lv")
    ap.add_argument("--arm", required=True, choices=["A", "B", "C", "I"])
    ap.add_argument("--out_dir", default="/root/autodl-tmp/acdc/seg_lv/ckpt")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--n_train_pat", type=int, default=0,
                    help="Number of training patients to subsample (0=all); val/test unchanged, for the low-budget crossover sweep")
    ap.add_argument(
        "--data_seed", type=int, default=0,
        help="Controls the patient split only; must be fixed to 0 for high-budget top-up runs")
    ap.add_argument(
        "--train_seed", type=int, required=True,
        help="Controls initialization, augmentation, DataLoader and optimization randomness only")
    ap.add_argument(
        "--split_json",
        default="/root/autodl-tmp/sam3-main/highN_fixed_split_seed0.json",
        help="Frozen patient-split JSON; required for high-budget top-up runs")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--no_aug", action="store_true", help="Disable training augmentation (ablation: test whether the sparse advantage comes from augmentation)")
    args = ap.parse_args()
    set_seed(args.train_seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_split = os.path.join(args.mroot, "train")

    df = pd.read_csv(os.path.join(args.mroot, "manifest_train.csv"))
    df["pf"] = df["pf"].fillna(""); df["gt"] = df["gt"].astype(str)
    # fixed patient-level split: the data split is strictly decoupled from training randomness
    split_path = os.path.abspath(args.split_json)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"fixed split JSON does not exist: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        split_record = json.load(f)

    if int(split_record.get("data_seed", -1)) != int(args.data_seed):
        raise RuntimeError(
            f"data_seed mismatch: args={args.data_seed}, "
            f"split_json={split_record.get('data_seed')}"
        )

    manifest_path = os.path.join(args.mroot, "manifest_train.csv")
    with open(manifest_path, "rb") as f:
        manifest_sha256 = hashlib.sha256(f.read()).hexdigest()

    expected_manifest_sha256 = split_record.get("manifest_sha256")
    if manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            "manifest_train.csv does not match the frozen split:\n"
            f"expected={expected_manifest_sha256}\n"
            f"actual  ={manifest_sha256}"
        )

    budget_key = str(int(args.n_train_pat))
    if budget_key not in split_record.get("budgets", {}):
        raise RuntimeError(
            f"budget N={budget_key} is not in the split JSON; "
            f"available budgets={list(split_record.get('budgets', {}).keys())}"
        )

    val_list = [str(x) for x in split_record["val_keys"]]
    tr_list = [
        str(x)
        for x in split_record["budgets"][budget_key]["keys"]
    ]

    val_keys = set(val_list)
    tr_keys = set(tr_list)

    if tr_keys & val_keys:
        raise RuntimeError(
            f"train/val patient overlap: {sorted(tr_keys & val_keys)}"
        )

    available_keys = set(df["key"].astype(str).unique())
    missing_train = sorted(tr_keys - available_keys)
    missing_val = sorted(val_keys - available_keys)

    if missing_train or missing_val:
        raise RuntimeError(
            f"the frozen split contains patients missing from the manifest: "
            f"missing_train={missing_train}, missing_val={missing_val}"
        )

    def key_digest(xs):
        payload = "\n".join(xs) + "\n"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    actual_train_sha256 = key_digest(tr_list)
    expected_train_sha256 = (
        split_record["budgets"][budget_key].get("sha256")
    )

    if actual_train_sha256 != expected_train_sha256:
        raise RuntimeError(
            "training patient list hash mismatch:\n"
            f"expected={expected_train_sha256}\n"
            f"actual  ={actual_train_sha256}"
        )

    df_tr = df[df["key"].astype(str).isin(tr_keys)].copy()
    df_val = df[df["key"].astype(str).isin(val_keys)].copy()

    run_manifest = {
        "script": "74_train_seg_students_fixedsplit.py",
        "arm": args.arm,
        "nominal_n_train_pat": int(args.n_train_pat),
        "actual_n_train_pat": len(tr_list),
        "n_val_pat": len(val_list),
        "data_seed": int(args.data_seed),
        "train_seed": int(args.train_seed),
        "val_frac_historical": float(args.val_frac),
        "encoder": args.encoder,
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.bs),
        "learning_rate": float(args.lr),
        "pretrained": not args.no_pretrained,
        "augmentation": not args.no_aug,
        "manifest_train_sha256": manifest_sha256,
        "split_json": split_path,
        "train_keys_sha256": actual_train_sha256,
        "val_keys_sha256": split_record.get("val_sha256"),
        "train_keys": tr_list,
        "val_keys": val_list,
    }

    with open(
        os.path.join(args.out_dir, f"run_manifest_{args.arm}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(run_manifest, f, indent=2, ensure_ascii=False)

    log(
        f"[split] nominal_N={args.n_train_pat} "
        f"actual_train={len(tr_keys)} val={len(val_keys)} "
        f"data_seed={args.data_seed} train_seed={args.train_seed} "
        f"(rows train_all={len(df_tr)} val={len(df_val)})"
    )
    log(f"[split] train_keys_sha256={actual_train_sha256}")
    log(f"[split] val_keys_sha256={split_record.get('val_sha256')}")

    rows_tr = build_rows(df_tr, root_split, args.arm)
    log(f"[arm {args.arm}] training label rows={len(rows_tr)}  "
        f"(A=all slices / B=anchor GT + interior fwd pseudo-label / C=anchors only)")
    if len(rows_tr) < 10:
        log("[!] too few training samples; check arm/manifest."); return

    ds_tr = SliceDS(rows_tr, root_split, aug=not args.no_aug)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=4,
                       drop_last=True, pin_memory=True)

    model = UNet(args.encoder, 1, 1, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    amp_enabled = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best, best_ep, wait, hist = -1.0, -1, 0, []

    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0.0; nb = 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = seg_loss(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += float(loss); nb += 1
        vd, vn = val_dice(model, df_val, root_split, device)
        hist.append(dict(epoch=ep, train_loss=tl / max(nb, 1), val_dice=vd, val_neg=vn))
        log(f"  ep{ep:3d} loss={tl / max(nb,1):.4f}  valDice(LV)={vd:.4f}  valNeg={vn:.3f}")
        if vd > best:
            best, best_ep, wait = vd, ep, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "encoder": args.encoder,
                    "arm": args.arm,
                    "val_dice": vd,
                    "epoch": ep,
                    "data_seed": int(args.data_seed),
                    "train_seed": int(args.train_seed),
                    "nominal_n_train_pat": int(args.n_train_pat),
                    "actual_n_train_pat": len(tr_list),
                    "train_keys_sha256": actual_train_sha256,
                    "val_keys_sha256": split_record.get("val_sha256"),
                    "manifest_train_sha256": manifest_sha256,
                    "split_json": split_path,
                },
                os.path.join(args.out_dir, f"student_{args.arm}.pt"),
            )
        else:
            wait += 1
            if args.patience and wait >= args.patience:
                log(f"  early stop @ep{ep} (best valDice={best:.4f} @ep{best_ep})"); break

    json.dump(hist, open(os.path.join(args.out_dir, f"hist_{args.arm}.json"), "w"))
    log(f"\n[done] arm {args.arm}  best val Dice(LV-present)={best:.4f} @ep{best_ep}  "
        f"-> student_{args.arm}.pt")


if __name__ == "__main__":
    main()
