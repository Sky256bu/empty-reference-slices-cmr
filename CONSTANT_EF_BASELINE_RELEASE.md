# Constant EF reference source and results

This separate archive supports the post hoc descriptive constant-predictor
reference in Supplementary Table S12 of the cardiac MRI empty-reference-slice
study. It does not replace the original frozen analysis or the
`v1.1-nnunet-r1` source archive.

Download `Constant_EF_baseline_source_and_results_20261105.zip` and its adjacent
`.sha256` checksum from this repository or the `v1.2-constant-ef` release.

ZIP SHA-256:
`2ac249b098574370cb33eb18a18b1efd1e71c912fbfd88b4c8d27cfce9f35543`.

The ZIP contains 11 files: the byte-preserved cloud runtime, retained anonymous
summaries and three CSVs, the frozen original model aggregate source, provenance,
an English README, a standard-library offline verifier, and member checksums.
No patient-level records, medical images, masks, weights, screenshots or
credentials are distributed.

After extracting, run `python verify_public_results.py` inside the extracted
directory. It verifies source/member hashes and 430 numeric relationships at
absolute tolerance 1e-10. The new verifier reconstructs public aggregates; it
does not independently recompute patient EF or training medians. Repeating the
retained cloud runtime requires the original controlled study inputs and the
dependencies described in the archive README.

Each of ten units fitted its own median reference EF using ten SIEMENS training
patients, then applied the constant to 141 held-out patients (88 Philips, 53 GE).
The overall constant MAE was 12.22 percentage points, versus 10.74 for original
lightweight-model arm B and 18.23 for Cneg. B was 1.48 points lower overall, 2.55
lower on Philips, and 0.29 higher on GE. This uses full training-reference EF,
exceeds the matched sparse annotation budget, and is descriptive without a new
confidence interval or significance test. It does not resolve undefined EF in
the separate nnU-Net experiment.

The five file identities declared in Table S12 are unchanged. The existing
`v1.1-nnunet-r1` tag remains at
`8632cb76dc47cc29133f7537399422a0e57e21f8`; the new archive and documentation
are additional files, without modifying existing frozen scripts or archives.
