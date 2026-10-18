# CoScaf

**Adapting in Concert: Recognizing the Unseen via Semantic Scaffolding**

This release includes the CoScaf model, source training, generator-only test-time
adaptation, original dataset splits and per-dataset configurations. CUB and
RetiRare-74 cached features, pretrained backbones and historical checkpoints are
distributed through a separate `CoScaf-assets` package; the download link will
be added after the author uploads it. Raw images are not included.

The active training code omits the source and TTA parameter-anchor losses.
Historical checkpoints retain their original learned parameters and can be used
for inference without rerunning TTA. Historical scores are not measurements of
the modified training objective. See [model records](docs/paper-best-h-configs.md).

## Install

Use Python 3.12 and a PyTorch installation appropriate for your device:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,retina]'
```

Transformers is pinned to 5.13.1. FP32 is used throughout the model; AMP and TF32
are disabled. Cached features avoid differences from image decoding. ORIX used
CUDA; local CPU checks and their exact scope are recorded in
[release validation](docs/release-validation.json).

## Historical checkpoint inference

Place the separately downloaded `CoScaf-assets` directory alongside this repository.
The following commands use the original CUB source checkpoint, saved adapted
generator and frozen calibration offsets. They do not train or rerun adaptation.

```bash
python -m coscaf.historical --mode verify \
  --manifest configs/CUB/historical-replay.json --assets ../CoScaf-assets

python -m coscaf.historical --mode replay \
  --manifest configs/CUB/historical-replay.json --assets ../CoScaf-assets \
  --output outputs/cub-replay --device cuda:0
```

For RetiRare-74, replace `CUB` in the manifest path with `RetiRare-74`.
Use `--limit 16 --device cpu` for a small forward check. A subset check does not
establish full-test replay. `--mode audit` recomputes metrics from the saved score
banks without loading the backbone. `--mode verify` checks all declared asset
hashes and creates a local receipt; rerun it after moving or extracting assets.

| Historical TTA checkpoint | Candidate / seed | H (%) | Same-model CZSL (%) |
|---|---|---:|---:|
| CUB | b128_p20 / 7033 | 86.7560 | 93.6216 |
| RetiRare-74 | b032_p22 / 7031 | 30.3343 | 34.0510 |

These are the paper's H-selected TTA models. The replay command also reports the
matched source-model metrics. It does not substitute a CZSL maximum from another
run. Full-precision S/U/H/CZSL/AUSUC and file hashes are in each replay manifest.

## Train the no-anchor model

Use the original splits in `data_splits/`; do not regenerate them for these
configurations. All four dataset configurations are under [`configs/`](configs/README.md).
CUB and RetiRare-74 can use the supplied train caches directly. For CUB:

```bash
coscaf develop --config configs/CUB/CoScaf-no-anchor.json \
  --data data_splits/CUB --cache ../CoScaf-assets/cache/CUB/train-portable \
  --vit-snapshot ../CoScaf-assets/models/backbones/vit_base_in21k \
  --output runs/cub/dev --device cuda:0

coscaf train --config configs/CUB/CoScaf-no-anchor.json \
  --data data_splits/CUB --cache ../CoScaf-assets/cache/CUB/train-portable \
  --vit-snapshot ../CoScaf-assets/models/backbones/vit_base_in21k \
  --development runs/cub/dev --output runs/cub/model --device cuda:0

coscaf evaluate --run runs/cub/model --data data_splits/CUB \
  --cache ../CoScaf-assets/cache/CUB/test \
  --vit-snapshot ../CoScaf-assets/models/backbones/vit_base_in21k \
  --output runs/cub/test --device cuda:0
```

Development uses seed 7011, up to 100 epochs, and patience 5 on TUNE H after CAL
calibration at each epoch. Final training starts afresh with the configured final
seed and the ceiling of the median best development epoch from this new run.
AWA2/CUB/RetiRare-74 use `initial_lr * 0.95 ** (epoch - 1)`; SUN uses constant LR.
Test labels enter only the final metric calculation. New no-anchor runs must use
their own development calibration, not historical offsets.

[Data setup](docs/data.md) lists retinal backbone arguments and instructions for
extracting AWA2/SUN features. [Asset contents](docs/assets.md) describes the
separate download. Use a new output directory for every run.

## Checks

```bash
pytest
ruff check src tests
```
