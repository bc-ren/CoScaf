# CoScaf-final-best-H configurations

The four [dataset configurations](../configs/README.md) under `configs/` retain
the existing JSON structure and record the parameters of the paper's highest-H
single-seed CoScaf models **with TTA**.
These are retrospective test-set maxima, not validation-selected seed averages.
All metrics in each row below belong to the same H-selected run; CZSL is not
replaced by a separate CZSL maximum. RetiRare-74 uses the code identifier
`RetiRareV2`.

| Dataset | Candidate | Final seed | K | Batch | Final epochs | H (%) | Same-run CZSL (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| AWA2 | b128_p20 | 7033 | 3 | 128 | 6 | 92.7208 | 94.2731 |
| CUB | b128_p20 | 7033 | 2 | 128 | 11 | 86.7560 | 93.6216 |
| SUN | b032_p08 | 7033 | 8 | 32 | 8 | 65.2385 | 85.8333 |
| RetiRare-74 | b032_p22 | 7031 | 8 | 32 | 4 | 30.3343 | 34.0510 |

[CoScaf-final-best-H.json](CoScaf-final-best-H.json) records full-precision
S/U/H/CZSL/AUSUC, run identities, calibration, training protocol and artifact
hashes. It is a provenance document, not a CLI configuration or checkpoint.
Model hashes were read from the audited inference receipts; checkpoint bytes
were not downloaded or rehashed for this configuration update.

## Loss coefficients

| Dataset | `source.anchor_weight` | `adaptation.anchor_weight` | `adaptation.topology_weight` |
|---|---:|---:|---:|
| AWA2 | 0.1 | 0.9 | 0.25 |
| CUB | 0.1 | 0.9 | 0.0 |
| SUN | 0.7 | 0.3 | 0.0 |
| RetiRare-74 | 0.2 | 1.0 | 0.0 |

The two anchor losses are active in the archived best-H runs and remain
unchanged. A zero penalty at initialization does not make an anchor inactive
after parameter updates. The topology coefficient is made explicitly zero
where the original configuration relied on its zero default. AWA2 retains
its active coefficient of 0.25.

The legacy `order_weight` is not consumed by the active adaptation code.
It was already zero in the archived AWA2 configuration and absent in the other
three. Its effective zero is recorded in the manifest; an unsupported option
is not added to the runnable JSON files.

## Training protocol and current code support

These JSON files map the archived model parameters to the public interface.
**Replacing the configuration alone does not reproduce the complete archived
experiment with the current public pipeline.** This update changes configuration
and documentation only; it does not add a scheduler, checkpoint selection or
a calibration-import path to the code.

The archived protocol used development seed 7011, a 100-epoch cap, and patience
5 on TUNE H after fitting that epoch's seen-class offset on CAL. Improvement
required more than 1e-12. The final model was initialized afresh using the
listed final seed and `ceil(median(best_epoch))` across the registered folds.
The fold-best epochs were AWA2 `[9, 3]`, CUB `[18, 3]`, SUN `[5, 11]`, and
RetiRare-74 `[6, 4, 3, 4, 1]`. Official test metrics did not determine these
training horizons or the calibration offsets.

AWA2, CUB and RetiRare-74 used `initial_lr * 0.95 ** (epoch - 1)`, with
one-based epoch indexing. SUN retained its original constant learning rates.
The JSON learning rates are the initial values. Features and model computation
were FP32, with AMP and TF32 disabled; no batch-size LR scaling was applied.

The current public pipeline uses one `seed` and fixed `source.epochs` for both
development and final training, and does not implement the archived epoch LR
decay or per-epoch development checkpoint selection. It also checks that the
configuration is unchanged between development and final training. Editing the
seed or epoch count between those commands is therefore not a replay solution.

The manifest separately records the CAL-fitted source offset used inside TTA
and adapted offset used for final GZSL evaluation. The public pipeline computes
its own calibration; it does not load these manifest values. Original folds,
training exclusions, class/sample ordering, preprocessing, backbone weights
and FP32 feature identities must also match. Archived and public data/cache
containers differ and require identity-preserving conversion.

For RetiRare-74, `source.prior_temperature=0.01` is intentional: the archived
ORIX `PortableData` execution path used the readout value 0.01. Its legacy
`source_recipe.prior_temperature=0.25` field was not the effective override.

No new training, inference or benchmark measurement was performed for this
update. End-to-end numerical parity with the archived implementation remains
unverified. Earlier public configurations remain available in Git history at
commit `024bc4b29c96b8ec350589b57c223ffb7894a3f9`.
