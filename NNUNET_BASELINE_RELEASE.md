# Additional nnU-Net baseline archive (rep00)

This repository addition accompanies the empty-reference-slice cardiac MRI study. The unchanged `nnUNet_public_code_release_20261020_v2.zip` contains the reviewed `nnunet_baseline/` directory (72 files), including source code, the exact shared reference plan, anonymous aggregate results, provenance records, and offline checks.

## Download and verify

Download [the version 2 archive](nnUNet_public_code_release_20261020_v2.zip) and [its SHA-256 record](nnUNet_public_code_release_20261020_v2.sha256), then extract the ZIP. From its `nnunet_baseline/` directory run:

```bash
python tools/verify_release.py
python tools/verify_release.py --self-test
```

The first command uses the Python standard library. The synthetic tests also require NumPy. These checks do not train a model or require patient data or CUDA.

Archive SHA-256: `d7a2cc1a4671a1dc7babbb83c0aef142317fb68e87a834db625b3681b52e78d5` (232,788 bytes).

The archive was rechecked before upload on 28 October 2026: 71 file hashes, 35 Python files compiled in memory, and 51 synthetic tests passed. The ZIP is retained as an exact byte-preserving archive rather than converting or rewriting its historical source files in the web editor. Extract it to obtain the directory structure documented in its README.

## Scope and limitations

- Ten training units, two supervision arms (B and Cneg), one training run per unit/arm: 20 nnU-Net models. `rep00` is a run index, not a paired-random-seed guarantee.
- The same 141 unique Philips/GE patients were evaluated for every model. Repeated model-patient evaluations are not additional unique patients.
- Segmentation, failure-rate and empty-reference-slice results are exploratory. The original complete-cohort EF-MAE comparison is undefined for some predictions; finite extremes and undefined values were not clipped or replaced.
- NIfTI spatial-unit declarations were absent in the examined headers. Absolute-volume outputs remain conditional on the archived conversion convention and do not establish physical mL accuracy.
- Exact controlled inputs are needed for the original workflow. The release is not a turnkey reproduction from a fresh dataset download. Historical execution and weight provenance are limited to the matches explicitly documented inside the archive.

Patient images, masks, identifiers, private case mappings, case-level tables, fixed splits, model weights and the reference fingerprint's training-derived arrays are excluded. Dataset access remains governed by the original organisers' terms. The author information/declaration form is not part of this code release.

The original `frozen/`, `audit/`, `readable/`, `MANIFEST.csv` and `v1.0-submission` tag are retained unchanged. The old tag covers the original code only; this additional archive must be cited using its own version or commit. Read the archive README, PRIVATE_INPUT_SPEC.md, DEPENDENCIES.md and provenance files before attempting the controlled-input workflow.
