# Dataset hyperparameters

Each dataset has a separate directory containing its active `CoScaf-no-anchor.json`
configuration in the existing public JSON format. Both source-training and TTA
parameter anchor losses have been removed from the code and these configurations.

| Dataset | Hyperparameter configuration | Final seed | K | Batch size | Final epochs |
|---|---|---:|---:|---:|---:|
| AWA2 | [CoScaf-no-anchor.json](AWA2/CoScaf-no-anchor.json) | 7033 | 3 | 128 | 6 |
| CUB | [CoScaf-no-anchor.json](CUB/CoScaf-no-anchor.json) | 7033 | 2 | 128 | 11 |
| SUN | [CoScaf-no-anchor.json](SUN/CoScaf-no-anchor.json) | 7033 | 8 | 32 | 8 |
| RetiRare-74 | [CoScaf-no-anchor.json](RetiRare-74/CoScaf-no-anchor.json) | 7031 | 8 | 32 | 4 |

Each file provides source-training parameters under `source`, TTA parameters
and loss coefficients under `adaptation`, target-setting parameters under
`adaptation_setting`, and the seed, folds and adaptation batch size. RetiRare-74
retains the internal dataset identifier `RetiRareV2`.

For example, use `--config configs/AWA2/CoScaf-no-anchor.json` in the
repository-root commands shown in the [main README](../README.md).

All remaining values are inherited from the corresponding archived highest-H
single-seed model with TTA. The original configurations, including their original
nonzero anchor weights, are preserved unchanged as
`<dataset>/archive/CoScaf-final-best-H.json`. They are historical records, not
configurations for the current loss function; use commit `32b1950` for the
previous public implementation.

No benchmark results have been measured for the modified recipe. See the
[protocol and code-support notes](../docs/paper-best-h-configs.md) and
[model records](../docs/CoScaf-final-best-H.json) for the original learning-rate
schedules, development selection, calibration and provenance. The current
public pipeline does not implement the complete archived training protocol.
