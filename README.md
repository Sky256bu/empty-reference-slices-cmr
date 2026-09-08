# Empty-reference slices in cardiac MR segmentation evaluation

Code accompanying the manuscript *Empty-reference slices escape present-slice overlap
metrics yet enter the ejection fraction: an evaluation blind spot in cardiac MR
segmentation*.

## What this repository contains

| Directory | Contents |
|---|---|
| `frozen/` | The 17 scripts and configuration files as they were run. Byte-exact. The SHA256 values declared in the Methods and in Supplementary Table S7 refer to **these** files. |
| `audit/` | 10 post-hoc audit and figure scripts, written after the frozen analysis. None produced a confirmatory result. |
| `readable/` | An English mirror of `frozen/` and `audit/`, produced by translating comments, docstrings and log messages. Provided for readability; see the caveat below. |
| `MANIFEST.csv` | Computed and declared SHA256 for every file. |

## Verifying the frozen scripts

```
sha256sum frozen/*
```

Every value should match the corresponding row of Supplementary Table S7 and of
`MANIFEST.csv`. The frozen scripts carry Chinese comments; this is deliberate. Translating
them would change their bytes and therefore their hashes, which would break the chain
between the manuscript's declarations and the files released here.

`readable/` exists for that reason. It is a **derived artefact, not the code that produced
the reported results**, and its files have different hashes by construction. The translation
touched only comments, docstrings and printed messages; every rewritten file was checked to
have an abstract syntax tree identical to its original once string constants are normalised.

## Pipeline

```
73_export_lv_pseudo.py          ACDC slice export + two-anchor propagation
73b_export_interp_pseudo.py     SDF interpolation control (arm I)
81_export_mnms2_pseudo.py       M&Ms-2 slice export + propagation
81b_export_mnms2_interp.py      SDF interpolation control, M&Ms-2
        |
74_train_seg_students.py        student training, arms A/B/C/I
74_train_seg_students_fixedsplit.py
mnms2_scope_training_engine_v3.py
mnms2_scope_control_config_v3.json
        |
75_eval_seg_and_ef.py           per-slice Dice + method-of-disks EF
81_eval_ssl_kill_test_v1.py     volume rule, voxel-volume and EDV>0 guards
82b_mnms2_patientlevel.py       cross-vendor case-level EF
84_eval_mnms2_scope_control_v3.py
        |
85_analyze_mnms2_scope_control_v3.py   crossed bootstrap, unit-only BCa, sign test, forest
87_aggregate_repeats_v3_1.py           R = 3 repeat aggregation
        |
step06_fix.py                   clean ACDC stack-Dice table
step07_extract_a5_a6.py         the three Dice definitions, empty_fp_penalty
stack_dice_check.py             stack-Dice self-consistency check
```

## Model

The propagation backbone is **SAM 3.1** with Object Multiplex
(`sam3.1_multiplex.pt`, built via `build_sam3_multiplex_video_predictor`).

A note on naming: `29_baseline_50.py` contains an output-key alias `sam2_backbone_out`, and
the checkpoint loader remaps a legacy `sam2_predictor.` prefix. These are internal key names
inherited from the SAM 2 tracker lineage and are not a statement about the model version.
Grepping this repository for `sam2` will find those key names and nothing else.

## Data availability

The M&Ms-2 checkpoints carry SHA256 anchors that have not changed since the run date, and
the corresponding masks are GPU-regenerable.

The ACDC prediction masks were not written to disk, and no checkpoint hash anchors were
retained for the 63 formal ACDC runs. That material is technically regenerable but
evidentially unrecoverable.

Neither dataset is redistributed here. ACDC and M&Ms-2 must be obtained from their
respective challenge organisers under their own terms.

## Not included

Abandoned experimental lines (2-D echocardiography, CAMUS, TED, cell tracking) and
superseded script versions are excluded. `87_aggregate_repeats_v3.py`, which exists in the
internal bundle, produced no reported result and is not shipped.

## Reproducing figures and tables

The scripts in `audit/` take their input directory from the `FROZEN_DATA_DIR` environment
variable, defaulting to the directory of the script itself. Place the frozen result tables
alongside them, or point the variable at wherever they are kept.

## Citation

Citation information will be added once the manuscript is published.

## License

Released under the MIT License. See `LICENSE`.
