# Dataset hyperparameters

Each dataset has a separate directory containing its `CoScaf-final-best-H.json`
configuration in the existing public JSON format.

| Dataset | Hyperparameter configuration | Final seed | K | Batch size | Final epochs |
|---|---|---:|---:|---:|---:|
| AWA2 | [CoScaf-final-best-H.json](AWA2/CoScaf-final-best-H.json) | 7033 | 3 | 128 | 6 |
| CUB | [CoScaf-final-best-H.json](CUB/CoScaf-final-best-H.json) | 7033 | 2 | 128 | 11 |
| SUN | [CoScaf-final-best-H.json](SUN/CoScaf-final-best-H.json) | 7033 | 8 | 32 | 8 |
| RetiRare-74 | [CoScaf-final-best-H.json](RetiRare-74/CoScaf-final-best-H.json) | 7031 | 8 | 32 | 4 |

Each file provides source-training parameters under `source`, TTA parameters
and loss coefficients under `adaptation`, target-setting parameters under
`adaptation_setting`, and the seed, folds and adaptation batch size. RetiRare-74
retains the internal dataset identifier `RetiRareV2`.

For example, use `--config configs/AWA2/CoScaf-final-best-H.json` in the
repository-root commands shown in the [main README](../README.md).

These configurations describe the archived highest-H single-seed models with
TTA. See the [protocol and code-support notes](../docs/paper-best-h-configs.md)
and [model records](../docs/CoScaf-final-best-H.json) for the original learning-rate
schedules, development selection, calibration and provenance. The current
public pipeline does not implement the complete archived training protocol.
