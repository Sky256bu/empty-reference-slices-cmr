#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
74_train_seg_students.py —— A/B/C 三臂 LV 分割学生(受控对照)
================================================================================
承重实验: "只标 2 锚点层/相位 + 传播伪标签" 训出的学生, 能否逼近 "全层稠密标注" 的学生?
  三臂**同架构(resnet34 UNet)同recipe**, 只换标签来源(读 73 导出的 manifest):
    A  full-dense : 所有层用 GT             -> 天花板(标全部)
    B  2anchor    : 锚点层GT + 内部层 pseudo_fwd + 跨度外空GT  -> 我们的方法(标2层/相位)
    C  sparse     : 只用锚点层(is_anchor==1)的 GT            -> 下界(2层本身不够)
  头条要的: B≈A (逼近全监督) 且 B≫C (传播不可省)。

数据来自 73_export_lv_pseudo.py:
  {mroot}/manifest_train.csv + {mroot}/train/{imgs,gts,pf,pb}/*.png
patient-level 80/20 切 train/val(同患者 ed/es 同侧, 防泄漏); val 一律用真 GT 评估。
最终 test 评估在 75_eval_seg_and_ef.py(此脚本只训 + 存权重 + 打印 val Dice)。

UNet/loss 结构照搬 train_student.py(已跑通), 自包含以免容器 churn。
依赖: torch/torchvision/timm/opencv/numpy/pandas。torch2.12 新 AMP API。

用法(三臂各跑一次, 固定同seed):
    cd /root/autodl-tmp/sam3-main
    for ARM in A B C; do
      python 74_train_seg_students.py --mroot /root/autodl-tmp/acdc/seg_lv \
          --arm $ARM --out_dir /root/autodl-tmp/acdc/seg_lv/ckpt \
          --encoder resnet34 --epochs 120 --patience 15 --bs 16 --seed 0
    done
产物: {out_dir}/student_{A,B,C}.pt  + 结尾打印各臂 best val Dice(LV-present)。
================================================================================
"""
import argparse, os, random, json
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


# ---------- 模型(照搬 train_student.py) ----------
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


# ---------- 数据 ----------
def label_path_for_arm(row, arm):
    """返回该行(切片)在指定臂下的标签 png 相对路径, 或 None(该臂不用此行)。"""
    if arm == "A":
        return row["gt"]
    if arm == "C":
        return row["gt"] if int(row["is_anchor"]) == 1 else None
    if arm == "B":
        if int(row["is_anchor"]) == 1 or int(row["in_lvspan"]) == 0:
            return row["gt"]                       # 锚点 or LV跨度外(空) 用 GT
        pf = str(row.get("pf", "") or "")
        return pf if pf else row["gt"]             # 内部层用 fwd 伪标签(缺则退GT)
    if arm == "I":                                 # SDF插值densification对照
        if int(row["is_anchor"]) == 1 or int(row["in_lvspan"]) == 0:
            return row["gt"]
        return str(row["img"]).replace("imgs/", "pi/", 1)   # 与73b同名约定
    raise ValueError(arm)


class SliceDS(Dataset):
    def __init__(self, rows, root_split, aug=False):
        self.rows = rows; self.root = root_split; self.aug = aug
    def __len__(self): return len(self.rows)
    def _load(self, rel):
        p = os.path.join(self.root, rel)
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(f"读不到标签/图像: {p}")
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
    """val 一律用真 GT; 只在 LV-present(gt_px>0) 层上算 Dice(有意义的数)。"""
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
                    help="子采样训练患者数(0=全部); val/test不变, 供低预算crossover扫描")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--no_aug", action="store_true", help="关训练增强(消融: 检验sparse优势是否来自增强)")
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_split = os.path.join(args.mroot, "train")

    df = pd.read_csv(os.path.join(args.mroot, "manifest_train.csv"))
    df["pf"] = df["pf"].fillna(""); df["gt"] = df["gt"].astype(str)
    # patient-level 划分
    keys = sorted(df["key"].unique())
    rng = random.Random(args.seed); rng.shuffle(keys)
    n_val = max(1, int(len(keys) * args.val_frac))
    val_keys = set(keys[:n_val])
    tr_pool = keys[n_val:]                         # 已按seed shuffle
    if args.n_train_pat and args.n_train_pat < len(tr_pool):
        tr_pool = tr_pool[:args.n_train_pat]       # 子采样(低预算扫描)
    tr_keys = set(tr_pool)
    df_tr = df[df["key"].isin(tr_keys)].copy()
    df_val = df[df["key"].isin(val_keys)].copy()
    log(f"[split] patients train={len(tr_keys)} val={len(val_keys)}  "
        f"(rows train_all={len(df_tr)} val={len(df_val)})")

    rows_tr = build_rows(df_tr, root_split, args.arm)
    log(f"[arm {args.arm}] 训练标签行数={len(rows_tr)}  "
        f"(A=全层 / B=锚点GT+内部fwd伪标 / C=仅锚点)")
    if len(rows_tr) < 10:
        log("[!] 训练样本过少, 检查 arm/manifest。"); return

    ds_tr = SliceDS(rows_tr, root_split, aug=not args.no_aug)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=4,
                       drop_last=True, pin_memory=True)

    model = UNet(args.encoder, 1, 1, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    best, best_ep, wait, hist = -1.0, -1, 0, []

    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0.0; nb = 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = seg_loss(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += float(loss); nb += 1
        vd, vn = val_dice(model, df_val, root_split, device)
        hist.append(dict(epoch=ep, train_loss=tl / max(nb, 1), val_dice=vd, val_neg=vn))
        log(f"  ep{ep:3d} loss={tl / max(nb,1):.4f}  valDice(LV)={vd:.4f}  valNeg={vn:.3f}")
        if vd > best:
            best, best_ep, wait = vd, ep, 0
            torch.save({"model": model.state_dict(), "encoder": args.encoder,
                        "arm": args.arm, "val_dice": vd, "epoch": ep},
                       os.path.join(args.out_dir, f"student_{args.arm}.pt"))
        else:
            wait += 1
            if args.patience and wait >= args.patience:
                log(f"  早停 @ep{ep} (best valDice={best:.4f} @ep{best_ep})"); break

    json.dump(hist, open(os.path.join(args.out_dir, f"hist_{args.arm}.json"), "w"))
    log(f"\n[done] arm {args.arm}  best val Dice(LV-present)={best:.4f} @ep{best_ep}  "
        f"-> student_{args.arm}.pt")


if __name__ == "__main__":
    main()
