# CoScaf-final-best-H configurations

The original configurations are preserved under
`configs/<dataset>/archive/CoScaf-final-best-H.json`. They record the parameters
of the paper's highest-H single-seed CoScaf models **with TTA**.
These are retrospective test-set maxima, not validation-selected seed averages.
All metrics in each row below belong to the same H-selected run; CZSL is not
replaced by a separate CZSL maximum. RetiRare-74 uses the code identifier
`RetiRareV2`.

The active [dataset configurations](../configs/README.md) are now named
`CoScaf-no-anchor.json`. Source-training and TTA parameter anchor losses
have been removed from the code and active configs.
All other configured values are retained. This changes the optimization
objective: the archived metrics below are not results for the modified recipe,
which has not been evaluated on the benchmarks.

| Dataset | Candidate | Final seed | K | Batch | Final epochs | H (%) | Same-run CZSL (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| AWA2 | b128_p20 | 7033 | 3 | 128 | 6 | 92.7208 | 94.2731 |
| CUB | b128_p20 | 7033 | 2 | 128 | 11 | 86.7560 | 93.6216 |
| SUN | b032_p08 | 7033 | 8 | 32 | 8 | 65.2385 | 85.8333 |
| RetiRare-74 | b032_p22 | 7031 | 8 | 32 | 4 | 30.3343 | 34.0510 |

[CoScaf-final-best-H.json](CoScaf-final-best-H.json) records full-precision
S/U/H/CZSL/AUSUC, run identities, calibration, training protocol and artifact
hashes. It is a provenance document, not a CLI configuration or checkpoint.
CUB and RetiRare-74 checkpoint bytes, saved score banks, original splits and
backbone assets are supplied separately. Their hashes and frozen calibrations
are listed in `configs/<dataset>/historical-replay.json`. The current validation
scope is recorded in [release-validation.json](release-validation.json).

## Archived loss coefficients

| Dataset | `source.anchor_weight` | `adaptation.anchor_weight` | `adaptation.topology_weight` |
|---|---:|---:|---:|
| AWA2 | 0.1 | 0.9 | 0.25 |
| CUB | 0.1 | 0.9 | 0.0 |
| SUN | 0.7 | 0.3 | 0.0 |
| RetiRare-74 | 0.2 | 1.0 | 0.0 |

The two anchor losses were active in the archived best-H runs. Their original
values remain in the archived records only. The current source objective is
class-balanced cross-entropy; TTA retains consistency, ranking, class/mode
alignment, semantic relations and the configured topology term. AWA2's topology
coefficient remains 0.25, and the other datasets retain 0.0.

New checkpoints omit the four source-initialization buffers used by the deleted
loss. The loader accepts old checkpoints by discarding exactly those four
obsolete buffers while strictly validating the remaining model state. This
preserves source-score restoration; retraining or rerunning TTA with the current
code uses the modified objective, not the historical optimization trajectory.

The legacy `order_weight` is not consumed by the active adaptation code.
It was already zero in the archived AWA2 configuration and absent in the other
three. Its effective zero is recorded in the manifest; an unsupported option
is not added to the runnable JSON files.

## Training protocol and replay

The archived runs used DEV seed 7011, a 100-epoch cap, patience 5 on TUNE H
after fitting that epoch's offset on CAL, and an improvement tolerance of 1e-12.
The current no-anchor pipeline implements those selection mechanics. Final
training starts afresh at the configured FINAL seed and uses
`ceil(median(best_epoch))` from the **current** development run. Historical
horizons are records, not substitutes for the new objective's selection.

AWA2, CUB and RetiRare-74 use `initial_lr * 0.95 ** (epoch - 1)`; SUN retains
constant LR. FP32, disabled AMP/TF32 and no batch-size LR scaling are retained.
Original NPZ folds and cache row orders are read directly, with train/test and
CAL/TUNE role checks. New runs calibrate only on their own development CAL scores.

Historical replay is a separate path: it loads the original source checkpoint,
the already-adapted generator and frozen source/adapted offsets. It never runs
an optimizer, refits gamma, or modifies weights. CZSL uses unseen-only scores;
GZSL uses each endpoint's own stored CAL offset. This preserves the original
inference recipe without restoring deleted anchor-loss code.

For RetiRare-74, `source.prior_temperature=0.01` is the effective original readout
value; the legacy recipe's `.25` field was not used on that execution path.
The RetiRare paper TTA endpoint is H30.3343%; its matched source endpoint is
H30.4989%. Neither is the separate search-wide no-TTA maximum (31.9573%).

The historical configurations were selected retrospectively by official-test H.
They are not validation-selected primary estimates. No new benchmark training
has been performed for the no-anchor objective. Earlier public configurations
remain in Git history at `024bc4b29c96b8ec350589b57c223ffb7894a3f9`.
