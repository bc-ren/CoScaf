# Dataset configurations

Each dataset has an active `CoScaf-no-anchor.json` in the public JSON format.
Architecture, batch sizes, initial learning rates and non-anchor coefficients
come from the corresponding historical highest-H single-seed TTA model.

| Dataset | Final seed | K | Batch | Historical final epochs |
|---|---:|---:|---:|---:|
| AWA2 | 7033 | 3 | 128 | 6 |
| CUB | 7033 | 2 | 128 | 11 |
| SUN | 7033 | 8 | 32 | 8 |
| RetiRare-74 | 7031 | 8 | 32 | 4 |

`source.epoch_decay` controls the actual scheduler. `development` specifies the
separate DEV seed, epoch cap, patience and improvement tolerance. With that
section present, `source.epochs` records the historical horizon; new final
training derives its horizon from the new development run. Removing anchor losses
changes the objective, so the historical metrics are not claimed for new training.

The unchanged `archive/CoScaf-final-best-H.json` files retain the original loss
coefficients as records, not active recipes. CUB and RetiRare-74 additionally have
`historical-replay.json` manifests for saved-model inference. They load the saved
adapted generator rather than executing the current TTA objective.

RetiRare-74 is called `RetiRareV2` in model configurations and `RetiRare74` in the
original asset paths. These names refer to the same 74-class protocol.
