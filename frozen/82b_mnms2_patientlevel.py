#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
82b_mnms2_patientlevel.py —— M&Ms-2 病例级跨厂商重跑
================================================================================
承重命题(跨中心泛化 punch):
  同一低预算学生, 测到别的厂商(OOD)时, sparse-only(C) 的 Dice/EF 掉得比 in-dist 更狠,
  SAM 传播(B) 守住 -> B−C 差距在 OOD 放大。若成立, in-hand 的"预算×难度相图"升级为
  "跨厂商泛化"。训练域固定 SIEMENS(唯一覆盖全8疾病, 隔离厂商漂移, 减疾病 confound)。

复用: 74_train_seg_students(UNet/seg_loss/SliceDS/label_path_for_arm/val_dice/set_seed)
      75_eval_seg_and_ef(dice_np/icc_2_1/predict_all) —— 不重造模型/EF 原语。
数据: 81 导出的 {mroot}/all/{imgs,gts,pf,pb} + manifest_mnms2.csv(带 vendor/disease)。
  三臂读法同 ACDC: A=全层GT / B=锚点GT+内部fwd伪标 / C=仅锚点GT。
  (I=SDF插值 需先跑 81b 出 pi/, 本脚本 --arms 默认 A,B,C; 有 pi 再加 I。)

EF: method-of-disks(逐层预测LV像素×体素体积), 体素体积从 M&Ms GT nii 的 affine 取。
  仅 LV(label=1) 有临床意义。ICC(2,1)/MAE/bias/HFrEF(GT EF<40) 亚组。

一次跑一个 (N, seed), 全部结果 append 到 results_mnms2.csv。外层循环 N/seed:
    cd /root/autodl-tmp/sam3-main
    for N in 10 25 50 0; do for S in 0 1 2; do
      python 82_mnms2_crossvendor.py --mroot /root/autodl-tmp/mnms/seg_lv \
          --root /root/autodl-tmp/mnms/MnM2 --n_train_pat $N --seed $S --arms A,B,C
    done; done
产物: {mroot}/results_mnms2.csv (逐 arm×domain 一行) + 结尾打印本次对照表。
判读: domain=Philips/GE 上 C 相对 SIEMENS_val 的跌幅 > B 的跌幅 => punch 成立。
================================================================================
"""
import argparse, importlib, os, sys, random, copy
import numpy as np, pandas as pd, cv2, nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _fit(t, S=256):
    """[1,H,W] -> [1,S,S], 居中 pad(补零)/crop, 不 resize(保物理尺度)。"""
    _, H, W = t.shape
    if H > S:
        top = (H - S) // 2; t = t[:, top:top + S, :]; H = S
    if W > S:
        left = (W - S) // 2; t = t[:, :, left:left + S]; W = S
    ph, pw = S - H, S - W
    if ph > 0 or pw > 0:
        t = F.pad(t, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=0.0)
    return t


class Fit256(Dataset):
    """包住 74.SliceDS(保留其增强), 把每个样本统一到 256x256, 使 default_collate 可堆叠。"""
    def __init__(self, base, size=256):
        self.base = base; self.size = size
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        img, lab = self.base[i]
        return _fit(img, self.size), _fit(lab, self.size)


def log(m): print(m, flush=True)


def get_primitives(code_dir):
    code_dir = os.path.abspath(code_dir)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    T = importlib.import_module("74_train_seg_students")
    E = importlib.import_module("75_eval_seg_and_ef")
    return T, E


def train_arm(rows, df_val, root_split, T, args, device):
    """训一个臂, 返回 (best_model, best_val_dice)。early-stop on SIEMENS-val LV Dice。"""
    if len(rows) < 10:
        return None, -1.0
    ds = Fit256(T.SliceDS(rows, root_split, aug=not args.no_aug), 256)
    dl = DataLoader(
        ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    model = T.UNet(args.encoder, 1, 1, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    best, best_state, wait = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = T.seg_loss(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        vd, _ = T.val_dice(model, df_val, root_split, device)
        if vd > best:
            best, best_state, wait = vd, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if args.patience and wait >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def eval_domain(model, df_dom, root_split, E, device):
    """返回 per-slice dice 数组(对齐 df_dom 行) + pred_px dict[(key,phase,z)]。"""
    pred_px, dice_row = E.predict_all(model, df_dom, root_split, device)
    return np.asarray(dice_row), pred_px


def ef_table(df_dom, pred_px, root, device_keys=None):
    """method-of-disks EF: 每 key 用 M&Ms GT nii 的 affine 取 vox_ml。返回 DataFrame。"""
    rows = []
    for key in sorted(df_dom["key"].unique()):
        vol_gt, vol_pr, ok = {}, {}, True
        for PH, phase in (("ED", "ed"), ("ES", "es")):
            sub = df_dom[(df_dom["key"] == key) & (df_dom["phase"] == phase)]
            if sub.empty:
                ok = False; break
            gp = os.path.join(root, "dataset", key, f"{key}_SA_{PH}_gt.nii.gz")
            if not os.path.exists(gp):
                ok = False; break
            vox_ml = abs(np.linalg.det(nib.load(gp).affine[:3, :3])) / 1000.0
            vol_gt[phase] = float(sub["gt_px"].sum()) * vox_ml
            pxs = 0
            for z in sub["z"]:
                pxs += pred_px.get((key, phase, int(z)), 0)
            vol_pr[phase] = float(pxs) * vox_ml
        if not ok:
            continue
        def ef(v):
            return 100.0 * (v["ed"] - v["es"]) / v["ed"] if v["ed"] > 1e-6 else np.nan
        ef_gt = ef(vol_gt)
        ef_pred = ef(vol_pr)
        ef_error = ef_pred - ef_gt
        rows.append(dict(
            key=key,
            edv_gt=vol_gt["ed"],
            esv_gt=vol_gt["es"],
            ef_gt=ef_gt,
            edv_pred=vol_pr["ed"],
            esv_pred=vol_pr["es"],
            ef_pred=ef_pred,
            ef_error=ef_error,
            ef_abs_error=abs(ef_error),
        ))
    return pd.DataFrame(rows)


def summarize_ef(ef_df, E):
    if ef_df.empty:
        return dict(ef_n=0, ef_bias=np.nan, ef_mae=np.nan, ef_loa=np.nan,
                    ef_r=np.nan, ef_icc=np.nan, hf_n=0, hf_mae=np.nan)
    g = ef_df["ef_gt"].values; p = ef_df["ef_pred"].values
    d = p - g; m = np.isfinite(d)
    r = np.corrcoef(p[m], g[m])[0, 1] if m.sum() > 3 else np.nan
    icc = E.icc_2_1(g[m], p[m]) if m.sum() > 3 else np.nan
    hf = ef_df[ef_df["ef_gt"] < 40]
    hfd = (hf["ef_pred"].values - hf["ef_gt"].values)
    hfm = np.isfinite(hfd)
    return dict(ef_n=int(m.sum()), ef_bias=float(d[m].mean()),
                ef_mae=float(np.abs(d[m]).mean()), ef_loa=float(2 * 1.96 * d[m].std()),
                ef_r=float(r), ef_icc=float(icc),
                hf_n=int(hfm.sum()),
                hf_mae=float(np.abs(hfd[hfm]).mean()) if hfm.sum() > 0 else np.nan)


def stratified_dice(df_dom, dice_arr):
    """整体(gt_px>0) / near(in_span,dist1-2) / far(in_span,dist>=3) 的 LV Dice。"""
    d = df_dom.copy(); d["_dice"] = dice_arr
    pres = d[d["gt_px"] > 0]
    ov = float(pres["_dice"].mean()) if len(pres) else np.nan
    near = pres[(pres["in_lvspan"] == 1) & (pres["dist"].between(1, 2))]
    far = pres[(pres["in_lvspan"] == 1) & (pres["dist"] >= 3)]
    return (ov, float(near["_dice"].mean()) if len(near) else np.nan,
            float(far["_dice"].mean()) if len(far) else np.nan, len(pres))


def vmask(df, name):
    return df["vendor"].astype(str).str.lower().str.contains(name.lower().strip())


def atomic_csv(df, path):
    """先写临时文件再替换，避免中断留下半个 CSV。"""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_torch_save(obj, path):
    """原子保存 checkpoint。"""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def upsert_csv(new_df, path, identity):
    """
    按 identity 去重更新。
    重跑同一 N/seed/arm/domain/key 时，用新结果覆盖旧结果。
    """
    path = os.path.abspath(path)
    if os.path.exists(path):
        old = pd.read_csv(path)
        both = pd.concat([old, new_df], ignore_index=True, sort=False)
    else:
        both = new_df.copy()

    missing = [c for c in identity if c not in both.columns]
    if missing:
        raise ValueError(f"upsert identity 缺列: {missing}")

    both = both.drop_duplicates(identity, keep="last")
    sort_cols = [c for c in identity if c in both.columns]
    both = both.sort_values(sort_cols).reset_index(drop=True)
    atomic_csv(both, path)


def cpu_state_dict(model):
    """checkpoint 一律保存 CPU tensor，便于无卡加载和归档。"""
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
    }


def load_saved_model(ckpt, T, args, device):
    st = torch.load(ckpt, map_location="cpu")
    encoder = st.get("encoder", args.encoder)
    model = T.UNet(
        encoder,
        1,
        1,
        pretrained=False,
    ).to(device)
    model.load_state_dict(st["model"])
    return model, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mroot", default="/root/autodl-tmp/mnms/seg_lv")
    ap.add_argument("--root", default="/root/autodl-tmp/mnms/MnM2")
    ap.add_argument("--code_dir", default="/root/autodl-tmp/sam3-main")
    ap.add_argument("--train_vendor", default="SIEMENS")
    ap.add_argument("--test_vendors", default="Philips,GE MEDICAL")
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--n_train_pat", type=int, default=0, help="0=全部SIEMENS训练池")
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="仅控制模型初始化、增强和优化随机性",
    )
    ap.add_argument(
        "--data_seed",
        type=int,
        default=20260714,
        help="固定患者 train/val 划分和训练子集；三个模型 seed 必须相同",
    )
    ap.add_argument(
        "--ckpt_root",
        default="",
        help="默认 {mroot}/ckpt_mnms2_patientlevel",
    )
    ap.add_argument(
        "--per_case_root",
        default="",
        help="默认 {mroot}/per_case_mnms2_patientlevel",
    )
    ap.add_argument(
        "--aggregate_out",
        default="",
        help="默认 {mroot}/results_mnms2_patientlevel.csv；不污染旧 results_mnms2.csv",
    )
    ap.add_argument(
        "--per_case_out",
        default="",
        help="默认 {mroot}/results_mnms2_per_case.csv",
    )
    ap.add_argument(
        "--eval_only",
        action="store_true",
        help="只加载已有 checkpoint 并评估；缺 checkpoint 时直接报错",
    )
    ap.add_argument(
        "--overwrite_ckpt",
        action="store_true",
        help="即使 checkpoint 已存在也重新训练并覆盖",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="只检查划分、域和输出路径，不训练不推理",
    )
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--no_aug", action="store_true")
    args = ap.parse_args()

    T, E = get_primitives(args.code_dir)
    T.set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_split = os.path.join(args.mroot, "all")
    arms = [a for a in args.arms.split(",") if a]

    df = pd.read_csv(os.path.join(args.mroot, "manifest_mnms2.csv"))
    df["pf"] = df["pf"].fillna(""); df["gt"] = df["gt"].astype(str)

    def norm_key(k):
        s = str(k).strip()
        try:
            return str(int(float(s))).zfill(3)   # int64 1 / "1" / "1.0" -> "001"
        except Exception:
            return s
    df["key"] = df["key"].map(norm_key)           # pandas 把 "001" 读成了 int, 修回

    # ---- 训练域(SIEMENS) 患者级 train/val ----
    tv = df[vmask(df, args.train_vendor)]
    tr_keys_all = sorted(tv["key"].unique())
    rng = random.Random(args.data_seed)
    rng.shuffle(tr_keys_all)
    n_val = max(1, int(len(tr_keys_all) * args.val_frac))
    val_keys = set(tr_keys_all[:n_val])
    tr_pool = tr_keys_all[n_val:]
    if args.n_train_pat and args.n_train_pat < len(tr_pool):
        tr_pool = tr_pool[:args.n_train_pat]
    tr_keys = set(tr_pool)
    df_tr = df[df["key"].isin(tr_keys)].copy()
    df_val = df[df["key"].isin(val_keys)].copy()
    N = len(tr_keys)
    log(f"[split] train_vendor={args.train_vendor}  train_pat={N}  val_pat={len(val_keys)}")

    # ---- 测试域集合: in-dist SIEMENS-val + 每个 OOD 厂商 ----
    domains = {"SIEMENS_val": df_val}
    for v in args.test_vendors.split(","):
        v = v.strip()
        if not v:
            continue
        sub = df[vmask(df, v)].copy()
        label = "Philips" if "philip" in v.lower() else ("GE" if "ge" in v.lower() else v)
        domains[label] = sub
        log(f"[domain] {label}: {sub['key'].nunique()} 例")

    # ---- P0 输出路径 ----
    ckpt_root = (
        args.ckpt_root
        if args.ckpt_root
        else os.path.join(args.mroot, "ckpt_mnms2_patientlevel")
    )
    per_case_root = (
        args.per_case_root
        if args.per_case_root
        else os.path.join(args.mroot, "per_case_mnms2_patientlevel")
    )
    aggregate_out = (
        args.aggregate_out
        if args.aggregate_out
        else os.path.join(args.mroot, "results_mnms2_patientlevel.csv")
    )
    per_case_out = (
        args.per_case_out
        if args.per_case_out
        else os.path.join(args.mroot, "results_mnms2_per_case.csv")
    )

    run_ckpt_dir = os.path.join(ckpt_root, f"N{N}_s{args.seed}")
    run_case_dir = os.path.join(per_case_root, f"N{N}_s{args.seed}")
    os.makedirs(run_ckpt_dir, exist_ok=True)
    os.makedirs(run_case_dir, exist_ok=True)

    log(
        f"[randomness] data_seed={args.data_seed} "
        f"model_seed={args.seed}"
    )
    log(f"[path] ckpt_dir={run_ckpt_dir}")
    log(f"[path] per_case_dir={run_case_dir}")
    log(f"[path] aggregate={aggregate_out}")
    log(f"[path] per_case_all={per_case_out}")

    if args.dry_run:
        log("[dry-run] train keys:")
        log(",".join(sorted(tr_keys)))
        log("[dry-run] val keys:")
        log(",".join(sorted(val_keys)))
        for arm in arms:
            ckpt = os.path.join(run_ckpt_dir, f"student_{arm}.pt")
            log(f"[dry-run] arm={arm} ckpt={ckpt}")
        log("[dry-run] 完成；未训练、未推理、未写结果。")
        return

    # ---- 逐臂 train/load + 逐域 eval + 病例级落盘 ----
    out_rows = []
    all_case_rows = []
    printed = {}

    for arm in arms:
        # 每个 arm 使用相同 model seed，避免 arm 顺序改变初始化。
        T.set_seed(args.seed)

        ckpt = os.path.join(run_ckpt_dir, f"student_{arm}.pt")
        model = None
        vbest = np.nan

        if os.path.exists(ckpt) and not args.overwrite_ckpt:
            model, st = load_saved_model(ckpt, T, args, device)

            if int(st.get("N", N)) != int(N):
                raise RuntimeError(
                    f"checkpoint N 不匹配: {ckpt}; "
                    f"ckpt={st.get('N')} current={N}"
                )
            if int(st.get("data_seed", args.data_seed)) != int(args.data_seed):
                raise RuntimeError(
                    f"checkpoint data_seed 不匹配: {ckpt}; "
                    f"ckpt={st.get('data_seed')} "
                    f"current={args.data_seed}"
                )

            vbest = float(st.get("val_dice", np.nan))
            log(
                f"[arm {arm}] 加载 checkpoint: {ckpt} "
                f"(val={vbest:.4f})"
            )
        else:
            if args.eval_only:
                raise FileNotFoundError(
                    f"--eval_only 但 checkpoint 不存在: {ckpt}"
                )

            rows = T.build_rows(df_tr, root_split, arm)
            model, vbest = train_arm(
                rows,
                df_val,
                root_split,
                T,
                args,
                device,
            )
            if model is None:
                log(f"[arm {arm}] 训练样本过少，跳过")
                continue

            payload = {
                "model": cpu_state_dict(model),
                "encoder": args.encoder,
                "val_dice": float(vbest),
                "train_vendor": args.train_vendor,
                "N": int(N),
                "requested_n_train_pat": int(args.n_train_pat),
                "seed": int(args.seed),
                "data_seed": int(args.data_seed),
                "arm": arm,
                "train_keys": sorted(tr_keys),
                "val_keys": sorted(val_keys),
                "epochs": int(args.epochs),
                "patience": int(args.patience),
                "lr": float(args.lr),
                "batch_size": int(args.bs),
            }
            atomic_torch_save(payload, ckpt)
            log(
                f"[arm {arm}] train_slices={len(rows)} "
                f"best val Dice={vbest:.4f}"
            )
            log(f"[arm {arm}] checkpoint -> {ckpt}")

        for dom, df_dom in domains.items():
            dice_arr, pred_px = eval_domain(
                model,
                df_dom,
                root_split,
                E,
                device,
            )
            ov, near, far, npres = stratified_dice(
                df_dom,
                dice_arr,
            )

            ef_df = ef_table(
                df_dom,
                pred_px,
                args.root,
            )
            efs = summarize_ef(ef_df, E)

            # 病例级长表。
            case_df = ef_df.copy()
            case_df.insert(0, "domain", dom)
            case_df.insert(0, "arm", arm)
            case_df.insert(0, "seed", int(args.seed))
            case_df.insert(0, "data_seed", int(args.data_seed))
            case_df.insert(0, "N", int(N))
            case_df.insert(0, "train_vendor", args.train_vendor)
            case_df["val_best"] = float(vbest)
            case_df["checkpoint"] = ckpt

            condition_csv = os.path.join(
                run_case_dir,
                f"{arm}_{dom}.csv",
            )
            atomic_csv(case_df, condition_csv)
            all_case_rows.append(case_df)

            row = dict(
                train_vendor=args.train_vendor,
                data_seed=int(args.data_seed),
                N=int(N),
                seed=int(args.seed),
                arm=arm,
                domain=dom,
                val_best=round(float(vbest), 4),
                dice_overall=round(ov, 4),
                dice_near=round(near, 4) if near == near else "",
                dice_far=round(far, 4) if far == far else "",
                n_pres=npres,
                checkpoint=ckpt,
                **{
                    k: (
                        round(v, 4)
                        if isinstance(v, float) and v == v
                        else v
                    )
                    for k, v in efs.items()
                },
            )
            out_rows.append(row)
            printed[(arm, dom)] = (
                ov,
                far,
                efs["ef_icc"],
                efs["ef_mae"],
            )

            log(
                f"[arm {arm}][{dom}] "
                f"cases={len(case_df)} "
                f"Dice={ov:.4f} "
                f"EF-MAE={efs['ef_mae']:.4f} "
                f"-> {condition_csv}"
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 本次 run 病例级文件 ----
    if all_case_rows:
        run_case_df = pd.concat(
            all_case_rows,
            ignore_index=True,
            sort=False,
        )
        run_case_csv = os.path.join(
            run_case_dir,
            "per_case_all.csv",
        )
        atomic_csv(run_case_df, run_case_csv)

        upsert_csv(
            run_case_df,
            per_case_out,
            identity=[
                "train_vendor",
                "data_seed",
                "N",
                "seed",
                "arm",
                "domain",
                "key",
            ],
        )
        log(
            f"[out] 病例级 {len(run_case_df)} 行 "
            f"-> {run_case_csv}"
        )
        log(f"[out] 病例级总表 upsert -> {per_case_out}")

    # ---- 聚合文件：单独写 P0 文件，不污染旧结果 ----
    dfo = pd.DataFrame(out_rows)
    if not dfo.empty:
        upsert_csv(
            dfo,
            aggregate_out,
            identity=[
                "train_vendor",
                "data_seed",
                "N",
                "seed",
                "arm",
                "domain",
            ],
        )
    log(f"\n[out] 聚合 {len(out_rows)} 行 -> {aggregate_out}")

    # ---- 本次对照表 ----
    log("\n" + "=" * 78)
    log(f"  N={N} seed={args.seed}  跨厂商对照 (Dice整体 / Dice远端 / EF-ICC / EF-MAE)")
    log("=" * 78)
    doms = list(domains.keys())
    log(f"  {'arm':>3s} | " + " | ".join(f"{d:^22s}" for d in doms))
    for arm in arms:
        cells = []
        for d in doms:
            if (arm, d) in printed:
                ov, far, icc, mae = printed[(arm, d)]
                cells.append(f"{ov:.3f}/{far if far==far else 0:.3f}/"
                             f"{icc if icc==icc else 0:.2f}/{mae if mae==mae else 0:.1f}")
            else:
                cells.append("-")
        log(f"  {arm:>3s} | " + " | ".join(f"{c:^22s}" for c in cells))
    log("\n[判读] 本脚本仅生成统计所需原始结果。")
    log("       是否成立以病例级配对效应、95%CI、Wilcoxon 和受益病例比例为准。")
    log("       不再使用‘OOD 必然放大 B−C 差距’作为预设命题。")


if __name__ == "__main__":
    main()
