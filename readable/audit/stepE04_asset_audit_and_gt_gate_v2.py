#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stepE04_asset_audit_and_gt_gate.py
==================================
Purpose: given the established fact that the prediction masks of the 63 formal runs were never
      written to disk, provide an **executable substitute gate** for the empty-slice
      false-positive taxonomy of stepE02, and establish the survival state of the assets
      (which decides whether the situation is "irreversible" or merely "expensive").

Upstream: EVAL_VALIDITY_EXTENSION_v0_3_DRAFT_FP_TAXONOMY.md 7
      (the original first gate, "check empty_fp case by case against real ACDC masks", is no
      longer executable)

Four modes, mutually independent, all **read-only and pure CPU**:

  --audit    Asset survival audit. Answers three questions:
             (1) whether the prediction masks really do not exist (evidence harder than an
                 argument from file counts)
             (2) whether the checkpoints are still present and whether their hashes still equal
                 the values declared in the per-case table
                 -> if so, the masks are "regenerable but require a GPU", not "permanently lost"
             (3) whether any array was written to disk in a non-NIfTI form (.npy/.npz/.h5/.mha/...)

  --discover Probe the ACDC GT layout. **Produces no ruling**; it only prints the file naming,
             axis order, affine spacing and label values, so that --gate can be parameterised.

  --gate     GT-side gate (**real masks, real gate**). Recompute, from the public ACDC GT, the
             foreground voxel count and empty-slice structure of every (patient, phase), and
             compare case by case against the `gt` column of the frozen stack_dice_acdc_clean.csv.
             What this validates is **the half of stepE02 that depends on GT alone**: whether
             axis 0 really is the slice axis, whether spacing is taken from the affine, and the
             definition of Z_present.

  --perturb  Real-anatomy perturbation gate. Construct pseudo-predictions with known answers from
             the real GT (out-of-span fusion / isolated blocks / interior holes / anisotropy) and
             validate decompose_empty_slice_fp of stepE02 against them.
             This is stronger than the T1-T9 synthetic phantoms (real shapes, real anisotropy,
             real Z), but it **still cannot substitute for validation against real model output**;
             under the protocol that gap must be written up explicitly as a known limitation and
             must not be treated as closed.

Identity: design-phase (the ACDC discovery set, authorised by amendment v0.2 1.1).
      Any output of this script **may not enter the current manuscript** and may not be compared
      statistically with any frozen result.
      --gate is a self-proof of code correctness (of the same nature as S1/S2 of stepE01), not a
      new statistic.

Dependencies: numpy scipy pandas nibabel

Usage
----
  python3 stepE04_asset_audit_and_gt_gate.py --audit \
      --roots /root/autodl-tmp /root \
      --percase-csv /path/mnms2_scope_control_v3_R_aggregate_averaged_per_case.csv \
      --out-dir /root/autodl-tmp/design_phase_quarantine

  python3 stepE04_asset_audit_and_gt_gate.py --discover --acdc-root /root/autodl-tmp/acdc

  python3 stepE04_asset_audit_and_gt_gate.py --gate \
      --acdc-root /root/autodl-tmp/acdc --label 3 \
      --frozen-csv /path/stack_dice_acdc_clean.csv \
      --out-dir /root/autodl-tmp/design_phase_quarantine

  python3 stepE04_asset_audit_and_gt_gate.py --perturb \
      --acdc-root /root/autodl-tmp/acdc --label 3 --taxonomy /path/stepE02_fp_taxonomy.py \
      --out-dir /root/autodl-tmp/design_phase_quarantine
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import tarfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError:
    nib = None

ARRAY_EXT = (".npy", ".npz", ".h5", ".hdf5", ".mha", ".mhd", ".nrrd",
             ".nii", ".nii.gz", ".pkl", ".pickle", ".raw")
CKPT_EXT = (".pt", ".pth", ".ckpt", ".safetensors")
PRED_HINT = re.compile(r"(pred|prob|logit|infer|seg_out|output|mask)", re.I)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def ensure_out(d):
    os.makedirs(d, exist_ok=True)
    return d


# ==================================================================== AUDIT
def cmd_audit(a):
    out = ensure_out(a.out_dir)
    report = {"mode": "audit", "roots": a.roots}

    ext_counter = Counter()
    pred_hits, ckpt_files, tars = [], [], []
    for root in a.roots:
        if not os.path.isdir(root):
            print(f"[WARN] does not exist: {root}")
            continue
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in (".git", "__pycache__")]
            for fn in fns:
                p = os.path.join(dp, fn)
                low = fn.lower()
                for e in ARRAY_EXT:
                    if low.endswith(e):
                        ext_counter[e] += 1
                        if PRED_HINT.search(p):
                            pred_hits.append(p)
                        break
                if low.endswith(CKPT_EXT):
                    ckpt_files.append(p)
                if low.endswith((".tar", ".tar.gz", ".tgz")):
                    tars.append(p)

    report["array_file_counts"] = dict(ext_counter)
    report["pred_like_array_files_n"] = len(pred_hits)
    report["pred_like_array_files_sample"] = pred_hits[:50]
    report["checkpoint_files_n"] = len(ckpt_files)

    # ---- tar inventory (listing only, no extraction)
    tar_inventory = {}
    for t in tars:
        try:
            with tarfile.open(t, "r:*") as tf:
                names = tf.getnames()
            c = Counter()
            for n in names:
                nl = n.lower()
                for e in ARRAY_EXT + CKPT_EXT:
                    if nl.endswith(e):
                        c[e] += 1
                        break
            tar_inventory[t] = {"n_members": len(names), "by_ext": dict(c),
                                "pred_like": sum(1 for n in names if PRED_HINT.search(n))}
        except Exception as ex:
            tar_inventory[t] = {"error": repr(ex)}
    report["tar_inventory"] = tar_inventory

    # ---- checkpoint survival + hash check (decides whether this is "irreversible" or "merely expensive")
    ck = {"declared": 0, "exists": 0, "hash_match": 0, "hash_mismatch": [], "missing": []}
    if a.percase_csv and os.path.exists(a.percase_csv):
        df = pd.read_csv(a.percase_csv)
        if {"checkpoint", "checkpoint_sha256"} <= set(df.columns):
            decl = df[["checkpoint", "checkpoint_sha256"]].drop_duplicates()
            ck["declared"] = len(decl)
            for p, h in decl.itertuples(index=False):
                if os.path.exists(p):
                    ck["exists"] += 1
                    got = sha256(p)
                    if got == h:
                        ck["hash_match"] += 1
                    else:
                        ck["hash_mismatch"].append({"path": p, "declared": h, "actual": got})
                else:
                    ck["missing"].append(p)
    report["checkpoint_survival"] = ck

    with open(os.path.join(out, "stepE04_audit.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n===== asset survival audit =====")
    print("array-like file count:", dict(ext_counter))
    print("suspected prediction array files:", len(pred_hits))
    print("checkpoint files:", len(ckpt_files))
    print(f"the per-case table declares {ck['declared']} checkpoints:"
          f"present {ck['exists']}, hash matching {ck['hash_match']}, "
          f"mismatching {len(ck['hash_mismatch'])}, missing {len(ck['missing'])}")
    print("\nReading rules (written here to preclude explanation after the fact):")
    print("  A. suspected prediction arrays = 0 and pred_like inside the tar = 0  -> the masks really were not written to disk (evidence stronger than a file-count comparison)")
    print("  B. all checkpoint hashes match                                       -> the masks are regenerable but require a GPU, not permanently lost")
    print("  C. a checkpoint is missing or its hash does not match                -> the masks of that run are permanently lost and must be recorded in the ledger")
    print(f"\nwritten: {os.path.join(out, 'stepE04_audit.json')}")


# ================================================================= DISCOVER
def _nifti_list(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith((".nii", ".nii.gz")):
                out.append(os.path.join(dp, fn))
    return sorted(out)


PAT_KEY = re.compile(r"(patient\d{3})", re.I)
PAT_FRAME = re.compile(r"frame(\d+)", re.I)


def cmd_discover(a):
    if nib is None:
        sys.exit("[ABORT] nibabel required")
    files = _nifti_list(a.acdc_root)
    print(f"total NIfTI: {len(files)}")
    gts = [f for f in files if "_gt" in os.path.basename(f).lower()]
    print(f"files containing _gt: {len(gts)}")
    keys = sorted({m.group(1).lower() for f in files if (m := PAT_KEY.search(f))})
    print(f"patient keys: {len(keys)}, e.g. {keys[:5]} ... {keys[-3:] if keys else ''}")
    cfgs = []
    for dp, _, fns in os.walk(a.acdc_root):
        for fn in fns:
            if fn.lower() == "info.cfg":
                cfgs.append(os.path.join(dp, fn))
    print(f"Info.cfg: {len(cfgs)} -- the source of the ED/ES frame numbers; without it phase alignment is impossible")
    for f in (gts or files)[:5]:
        img = nib.load(f)
        zoom = img.header.get_zooms()
        data = np.asanyarray(img.dataobj)
        vals = np.unique(data)
        print(f"\n{f}\n  shape={data.shape} zooms={zoom}")
        print(f"  affine diagonal={np.round(np.abs(np.diag(img.affine))[:3], 4)}")
        print(f"  label values={vals[:10]}")
        if data.ndim == 3:
            nz = [int(data[..., z].any()) for z in range(data.shape[2])]
            print(f"  non-empty slices on the last axis {sum(nz)}/{data.shape[2]} (if the last axis is the slice axis, this should be ~the number of slices from apex to base)")
    print("\nWarning: native ACDC NIfTI is usually (X, Y, Z), so the slice axis is the **last** axis;"
          "stepE02 assumes axis 0 is the slice axis. --gate/--perturb transpose according to --slice-axis."
          "This convention must be confirmed separately for each dataset and must not be assumed to be the same (spec 7).")


# ===================================================================== GATE
PAT_PHASE_NAME = re.compile(r"[_\-](ed|es)[_\-]", re.I)


def _phase_from_name(base, key, ed_es):
    """Two naming conventions:
       (a) official ACDC: patientXXX_frameNN_gt.nii.gz + Info.cfg giving the ED/ES frame numbers
       (b) this project's preprocessed copy: patientXXX_sax_ed_gt.nii.gz (the phase is in the file name, no Info.cfg)
       Only these two are recognised; anything else returns None -- no guessing."""
    m = PAT_PHASE_NAME.search(base)
    if m:
        return m.group(1).lower()
    fr = PAT_FRAME.search(base)
    if fr and key in ed_es:
        ed, es = ed_es[key]
        f = int(fr.group(1))
        return "ed" if f == ed else ("es" if f == es else None)
    return None


def _load_acdc_pairs(root, label, slice_axis):
    """Return {(key, phase): (mask3d[Z,Y,X], zooms_zyx)}."""
    if nib is None:
        sys.exit("[ABORT] nibabel required")
    # 1) Info.cfg -> ED/ES frame numbers (needed only for naming convention (a); left empty if absent)
    ed_es = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower() == "info.cfg":
                k = PAT_KEY.search(dp)
                if not k:
                    continue
                cfg = {}
                for line in open(os.path.join(dp, fn)):
                    if ":" in line:
                        a_, b_ = line.split(":", 1)
                        cfg[a_.strip().upper()] = b_.strip()
                try:
                    ed_es[k.group(1).lower()] = (int(cfg["ED"]), int(cfg["ES"]))
                except Exception:
                    pass
    out = {}
    for f in _nifti_list(root):
        base = os.path.basename(f).lower()
        if "_gt" not in base:
            continue
        k = PAT_KEY.search(base)
        if not k:
            continue
        key = k.group(1).lower()
        phase = _phase_from_name(base, key, ed_es)
        if phase is None:
            continue
        img = nib.load(f)
        d = np.asanyarray(img.dataobj)
        if d.ndim != 3:
            continue
        m = (d == label)
        z = list(img.header.get_zooms()[:3])
        m = np.moveaxis(m, slice_axis, 0)
        z = [z[slice_axis]] + [z[i] for i in range(3) if i != slice_axis]
        out[(key, phase)] = (m, tuple(float(x) for x in z))
    return out


def cmd_gate(a):
    out = ensure_out(a.out_dir)
    fro = pd.read_csv(a.frozen_csv)
    # the gt column for the same (key, phase) in the frozen table should be constant across the 63 runs -- check that first
    const = fro.groupby(["key", "phase"])["gt"].nunique()
    n_nonconst = int((const > 1).sum())
    print(f"[pre-check] number of (key,phase) whose gt column is not constant across runs in the frozen table: {n_nonconst} (should be 0)")

    ref = fro.groupby(["key", "phase"])["gt"].first().to_dict()
    pairs = _load_acdc_pairs(a.acdc_root, a.label, a.slice_axis)
    print(f"[pre-check] number of (key,phase) parsed from the GT: {len(pairs)}; in the frozen table: {len(ref)}")
    if not pairs:
        sys.exit("[ABORT] no GT unit was parsed -- this is a parsing failure, not a gate FAIL."
                 "Run --discover first to check the naming and directories. 0/0 must not be recorded as any ruling.")

    rows, miss = [], []
    for (k, ph), gtv in ref.items():
        if (k, ph) not in pairs:
            miss.append((k, ph))
            continue
        m, zoom = pairs[(k, ph)]
        zpres = np.array([m[z].any() for z in range(m.shape[0])])
        rows.append({
            "key": k, "phase": ph,
            "gt_frozen": int(gtv), "gt_recomputed": int(m.sum()),
            "diff": int(m.sum()) - int(gtv),
            "n_slices": int(m.shape[0]),
            "n_gt_present": int(zpres.sum()),
            "n_gt_empty": int((~zpres).sum()),
            "gt_support_contiguous": bool(
                zpres.sum() == 0 or
                (np.flatnonzero(zpres)[-1] - np.flatnonzero(zpres)[0] + 1) == zpres.sum()),
            "dz_mm": zoom[0], "dy_mm": zoom[1], "dx_mm": zoom[2],
        })
    df = pd.DataFrame(rows)
    p = os.path.join(out, "stepE04_gt_gate.csv")
    df.to_csv(p, index=False)

    n = len(df)
    exact = int((df["diff"] == 0).sum()) if n else 0
    print("\n===== GT-side gate =====")
    print(f"comparable units: {n}; missing: {len(miss)}")
    print(f"`gt` column exactly equal case by case: {exact}/{n}")
    if n and exact < n:
        print("mismatching samples:")
        print(df[df["diff"] != 0].head(10).to_string(index=False))
    if n:
        print(f"empty-slice count distribution: median {df['n_gt_empty'].median():.1f}, "
              f"range {df['n_gt_empty'].min()}-{df['n_gt_empty'].max()}")
        print(f"units whose GT axial support is contiguous: {int(df['gt_support_contiguous'].sum())}/{n} "
              "(a non-contiguous support makes the beyond / interior split ambiguous and must be handled in the protocol)")
    print("\nReading (fixed in advance; no explanation after the fact):")
    print("  PASS  = exact/n is 1.00 -> axis order, labels, spacing and phase mapping agree with the frozen pipeline")
    print("  FAIL  = any mismatch -> check --slice-axis and --label first, then the GT version;"
          "while in the FAIL state no readout other than --perturb may be run")
    print(f"\nwritten: {p}")


# ================================================================== PERTURB
def _load_taxonomy(path):
    spec = importlib.util.spec_from_file_location("taxo", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, sha256(path)


def _make_cases(gt):
    """Construct pseudo-predictions with known answers from the real GT. Returns {name: (pred, expected property)}."""
    from scipy import ndimage
    Z = gt.shape[0]
    zp = np.array([gt[z].any() for z in range(Z)])
    idx = np.flatnonzero(zp)
    cases = {}
    # P0 perfect prediction: empty-slice false positives should be 0
    cases["P0_perfect"] = (gt.copy(), {"empty_fp_total_mm3": 0.0})
    if idx.size == 0:
        return cases
    lo, hi = int(idx[0]), int(idx[-1])
    # P1 one slice beyond the span and connected to the main body (LCC cannot remove it) -> connected + beyond
    p1 = gt.copy()
    if hi + 1 < Z:
        p1[hi + 1] = ndimage.binary_dilation(gt[hi], iterations=1)
        cases["P1_beyond_connected"] = (p1, {"expect": "connected>0, lcc_removable_frac==0"})
    # P2 an isolated block on an empty slice (LCC can remove it) -> isolated
    p2 = gt.copy()
    tgt = None
    for z in range(Z):
        if not zp[z] and (z == 0 or z >= hi + 2):
            tgt = z
            break
    if tgt is not None:
        yy, xx = gt.shape[1] // 4, gt.shape[2] // 4
        p2[tgt, yy:yy + 3, xx:xx + 3] = True
        cases["P2_isolated"] = (p2, {"expect": "isolated>0, lcc_removable_frac==1"})
    # P3 both present -> LCC removes only the isolated share
    if "P1_beyond_connected" in cases and "P2_isolated" in cases:
        cases["P3_mixed"] = (p1 | p2, {"expect": "0 < lcc_removable_frac < 1"})
    # P4 add a spurious block on present slices only -> must not be counted as empty-slice FP (the taxonomy stays in scope)
    p4 = gt.copy()
    yy, xx = gt.shape[1] // 4, gt.shape[2] // 4
    p4[lo, yy:yy + 3, xx:xx + 3] = True
    cases["P4_present_only_blob"] = (p4, {"empty_fp_total_mm3": 0.0})
    # P5 corner contact (the real-anatomy version of T8): connected under 26, isolated under 6
    #    this case decides whether the 26/6 sensitivity of spec 4 is **substantive** on real data
    if hi + 1 < Z:
        ys, xs = np.nonzero(gt[hi])
        if ys.size:
            j = int(np.argmax(ys + xs))
            y0, x0 = int(ys[j]) + 1, int(xs[j]) + 1
            if y0 + 2 <= gt.shape[1] and x0 + 2 <= gt.shape[2]:
                p5 = gt.copy()
                blob = np.zeros_like(gt)
                blob[hi + 1, y0:y0 + 2, x0:x0 + 2] = True
                # guard: must not intersect the GT and must have no 6-adjacency (otherwise the case does not hold and is skipped)
                from scipy import ndimage as _nd
                face = _nd.binary_dilation(blob, structure=_nd.generate_binary_structure(3, 1))
                if not (blob & gt).any() and not (face & gt).any():
                    p5 |= blob
                    cases["P5_corner_contact"] = (
                        p5, {"expect": "26: connected(lcc_removable=0) / 6: isolated(=1)"})
    return cases


def cmd_perturb(a):
    out = ensure_out(a.out_dir)
    taxo, taxo_hash = _load_taxonomy(a.taxonomy)
    print(f"stepE02 SHA256 = {taxo_hash}")
    pairs = _load_acdc_pairs(a.acdc_root, a.label, a.slice_axis)
    if not pairs:
        sys.exit("[ABORT] no (key, phase) GT unit was parsed. Run --discover first to check the naming and directories,"
                 "do not produce any readout on empty input.")
    keys = sorted(pairs.keys())[: a.limit] if a.limit else sorted(pairs.keys())
    rows = []
    for (k, ph) in keys:
        gt, zoom = pairs[(k, ph)]
        for cname, (pred, exp) in _make_cases(gt).items():
            for conn in (26, 6):
                r = taxo.decompose_empty_slice_fp(pred, gt, spacing=zoom, connectivity=conn)
                r.update({"key": k, "phase": ph, "case": cname,
                          "expect": json.dumps(exp, ensure_ascii=False)})
                rows.append(r)
    df = pd.DataFrame(rows)
    p = os.path.join(out, "stepE04_perturb.csv")
    df.to_csv(p, index=False)

    print("\n===== real-anatomy perturbation gate =====")
    ok = True
    for cname, expected_zero in (("P0_perfect", True), ("P4_present_only_blob", True)):
        sub = df[(df["case"] == cname)]
        if len(sub):
            bad = int((sub["empty_fp_total_mm3"].fillna(0) > 0).sum())
            print(f"{cname}: empty-slice false positives should be 0 -> violations {bad}/{len(sub)}")
            ok &= (bad == 0)
    for cname, col, want in (("P1_beyond_connected", "empty_fp_connected_mm3", ">0"),
                             ("P2_isolated", "empty_fp_isolated_mm3", ">0")):
        sub = df[(df["case"] == cname) & (df["connectivity"] == 26)]
        if len(sub) and col in sub:
            good = int((sub[col].fillna(0) > 0).sum())
            print(f"{cname}: {col} {want} -> satisfied {good}/{len(sub)}")
            ok &= (good == len(sub))
    sub5 = df[df["case"] == "P5_corner_contact"]
    if len(sub5):
        a26 = sub5[sub5["connectivity"] == 26]["lcc_removable_frac"].fillna(-1)
        a6 = sub5[sub5["connectivity"] == 6]["lcc_removable_frac"].fillna(-1)
        print(f"P5_corner_contact: 26-neighbourhood median {a26.median():.2f} (should be 0), "
              f"6-neighbourhood median {a6.median():.2f} (should be 1) -- {len(a26)} cases constructed successfully")
    if "lcc_removable_frac" in df:
        s26 = df[df["connectivity"] == 26].set_index(["key", "phase", "case"])["lcc_removable_frac"]
        s6 = df[df["connectivity"] == 6].set_index(["key", "phase", "case"])["lcc_removable_frac"]
        j = pd.concat([s26.rename("c26"), s6.rename("c6")], axis=1).dropna()
        if len(j):
            flip = float((np.abs(j["c26"] - j["c6"]) > 0.5).mean())
            print(f"fraction of large divergence in lcc_removable_frac between the 26- and 6-neighbourhood: {flip:.3f}")
            print("  -> this fraction is the substantive evidence for the sensitivity analysis of spec 4;"
                  "if it is close to 0 under real anisotropy, record honestly that the sensitivity is not substantive on this data")
    print(f"\noverall: {'PASS' if ok else 'FAIL'}")
    print("Warning: this gate uses real anatomy and real anisotropic spacing, which is stronger than the T1-T9 synthetic phantoms,"
          "but it contains **no real model output**. That gap must not be described as closed.")
    print(f"\nwritten: {p}")


# ====================================================================== CLI
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audit", action="store_true")
    g.add_argument("--discover", action="store_true")
    g.add_argument("--gate", action="store_true")
    g.add_argument("--perturb", action="store_true")
    ap.add_argument("--roots", nargs="*", default=["/root/autodl-tmp"])
    ap.add_argument("--percase-csv", default=None)
    ap.add_argument("--acdc-root", default=None)
    ap.add_argument("--frozen-csv", default=None)
    ap.add_argument("--taxonomy", default="stepE02_fp_taxonomy.py")
    ap.add_argument("--label", type=int, default=3, help="ACDC: 1=RV 2=MYO 3=LV")
    ap.add_argument("--slice-axis", type=int, default=2, help="the slice axis in the native NIfTI; defaults to the last axis")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="./design_phase_quarantine")
    a = ap.parse_args()

    if a.audit:
        cmd_audit(a)
    elif a.discover:
        if not a.acdc_root:
            sys.exit("[ABORT] --discover requires --acdc-root")
        cmd_discover(a)
    elif a.gate:
        for req in ("acdc_root", "frozen_csv"):
            if not getattr(a, req):
                sys.exit(f"[ABORT] --gate requires --{req.replace('_', '-')}")
        cmd_gate(a)
    else:
        if not a.acdc_root:
            sys.exit("[ABORT] --perturb requires --acdc-root")
        cmd_perturb(a)


if __name__ == "__main__":
    main()
