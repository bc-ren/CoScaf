# CoScaf

**Adapting in Concert: Recognizing the Unseen via Semantic Scaffolding**

CoScaf jointly learns visual evidence and a shared semantic prototype generator,
then adapts the generator using an unlabeled target set. It supports generalized
zero-shot learning (GZSL) and conventional unseen-only evaluation (CZSL).

```text
Images → Frozen encoder + learned prompts → Global / local / regional evidence
                                                     ↓
Fixed class semantics → Shared prototype generator → Class scores
                                  ↑
                 Reliable cross-layer target assignments
                      (generator-only adaptation)
```

This repository provides the model, data preparation, development calibration,
training, adaptation, evaluation, and tests. It contains no dataset images,
sample annotations, semantic matrices, feature caches, or model weights.

## Installation

Use Python 3.10 or newer. Install a PyTorch build appropriate for your CUDA version
first, then install CoScaf:

```bash
git clone git@github.com:bc-ren/CoScaf.git
cd CoScaf
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Verify the installation without downloading data or pretrained weights:

```bash
pytest -q
coscaf smoke --output /tmp/coscaf-smoke
```

The smoke command exercises development, fresh source training, checkpoint
restoration, calibration, and adaptation on generated tensors. Its numbers are
not benchmark results. [Runtime details](docs/environment.md) distinguish the
archived environments from release-validation environments.

## Data and encoders

| Dataset | Fixed semantics | Frozen visual encoder | Input |
|---|---|---|---|
| AWA2 | 85D attributes | Google ViT-Base/16, ImageNet-21K | 384 × 384 |
| CUB | 1024D sentence vectors (`sent_splits.mat`) | Same ViT-Base | 384 × 384 |
| SUN | 102D attributes | Same ViT-Base | 384 × 384 |
| RetiRareV2 | 512D RetiZero disease-name vectors | Official RetiZero | 224 × 224 |

The natural-image backbone is pinned to `google/vit-base-patch16-224-in21k`,
revision `b4569560a39a0f1af58e3ddaf17facf20ab919b0`, with a checkpoint checksum
check. It downloads on first use, or can be supplied via `--vit-snapshot`.
Natural-image experiments do not use CLIP. RetiZero is obtained separately;
see [retinal setup](docs/retinal.md).

[Data preparation](docs/data.md) describes the portable bundle, official MATLAB
import, class ordering, one-based indices, and training-side quarantine. Official
test membership and row order are preserved. For example:

```bash
coscaf prepare-natural --dataset AWA2 \
  --labels /path/to/AWA2/res101.mat \
  --splits /path/to/AWA2/att_splits.mat \
  --images /path/to/awa2_image_paths.json --output data/awa2
```

`--labels` reads only labels, not ResNet features. `images.json` is an ordered
list of image paths relative to the image root. For CUB, also pass
`--semantics /path/to/CUB/sent_splits.mat`. An existing audited zero-based
training quarantine can be supplied with `--exclude-train`; tests are unchanged.

## Run an experiment

Run commands from the repository root. Use new output directories; existing runs
are never silently overwritten. Substitute the configuration and paths for each
dataset. The following example uses AWA2.

**1. Extract the frozen training prefix.**

```bash
coscaf cache --config configs/awa2.json --data data/awa2 \
  --image-root /path/to/AWA2/JPEGImages --role train \
  --output cache/awa2/train --device cuda:0
```

Natural-image prefixes are unprompted block-10 tokens, including CLS. Prompts
are trained in blocks 11–12. Existing 768D image vectors cannot replace these
patch tokens. FP32 prefix storage needs approximately 1.7 MiB per natural image;
local evidence can need substantially more disk space, especially for SUN.

**2. Train seen-only development replicas and calibrate.**

```bash
coscaf develop --config configs/awa2.json --data data/awa2 \
  --cache cache/awa2/train --output runs/awa2/development --device cuda:0
```

Each development model starts fresh. Its source partition alone determines
whitening and ridge initialization. Pseudo-unseen classes come from original
seen classes. CAL fits the seen-class offset and probability temperature; TUNE
provides development metrics. Official test data enter neither operation.
Both source and adapted calibration are frozen in `calibration.json`.

**3. Train the final source model from scratch.**

```bash
coscaf train --config configs/awa2.json --data data/awa2 \
  --cache cache/awa2/train --development runs/awa2/development \
  --output runs/awa2/final --device cuda:0
```

All task parameters are initialized anew. Prompts, generator, queries, mode
priors where enabled, and regional readout share one optimizer trajectory.
Only the pretrained encoder is reused. `freeze.json` records the model,
configuration, calibration, and dataset identity before official inference.

**4. Extract the test prefix and evaluate.**

```bash
coscaf cache --config configs/awa2.json --data data/awa2 \
  --image-root /path/to/AWA2/JPEGImages --role test \
  --run runs/awa2/final --output cache/awa2/test --device cuda:0

coscaf evaluate --run runs/awa2/final --data data/awa2 \
  --cache cache/awa2/test --output runs/awa2/evaluation --device cuda:0
```

Evaluation reports the source model and **transductive generator-only TTA**.
Adaptation updates a copy of the same run's generator using the whole unlabeled
target set, processed in memory-sized chunks. The backbone, prompts, semantics,
queries, and regional readout remain fixed. Official labels enter metric
computation only after predictions are fixed.

Outputs include `metrics.json`, complete S–U curves, scores, and adaptation
diagnostics. S/U/H/CZSL/AUSUC/ECE/Brier are stored as fractions. See
[method and metrics](docs/method.md) for the definitions.

For multiple seeds, copy a configuration, change `seed`, and repeat the entire
workflow in separate output directories. Independent runs can use `cuda:0` and
`cuda:1`; scripts do not reserve or interrupt other GPU jobs.

## Registered reference results

These are archived **CoScaf-Best single-seed registrations**, selected
retrospectively by test H. They are not validation-selected seed averages or
new measurements from packaging this repository. The recipes support
reproduction; select new settings using development data.

| Dataset | Seed | K | S (%) | U (%) | H (%) | CZSL (%) | AUSUC (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AWA2 | 3408 | 3 | 93.11 | 89.87 | 91.46 | 93.31 | 89.42 |
| CUB | 3409 | 2 | 79.12 | 92.72 | 85.38 | 94.74 | 83.50 |
| SUN | 3407 | 8 | 57.79 | 75.21 | 65.36 | 86.18 | 52.44 |
| RetiRareV2 | 3407 | 8 | 30.04 | 26.17 | 27.97 | 29.72 | 12.89 |

Full-precision values are in [reference_results.json](docs/reference_results.json).
Core computations were compared against the archived implementation.
Dataset-scale retraining was not performed for this release. Exact replay also
requires the original training exclusions, folds, preprocessing, and environment.

## Code map

```text
configs/                 Dataset-specific recipes
src/coscaf/
  backbones.py           Frozen encoders and late visual prompts
  initialization.py      Fit-only whitening and ridge initialization
  model.py               Shared generator and evidence readouts
  training.py            Joint source training
  transport.py           Cross-layer reliability and partial transport
  adaptation.py          Generator-only target adaptation
  metrics.py             Calibration, GZSL/CZSL, and S–U curves
  data.py                Class/index/split/cache validation
  preparation.py         Metadata import
  pipeline.py            Development, training, and evaluation
  cli.py                 Command-line entry points
  smoke.py               Synthetic end-to-end example
tests/                   Numerical, protocol, and integration tests
docs/                    Data, method, runtime, and retinal setup
```

## Attribution

Dependencies, datasets, and pretrained models retain their respective licenses
and citation requirements. RetiZero code and weights are not redistributed;
see [third-party notices](NOTICE.md).

The manuscript is titled **Adapting in Concert: Recognizing the Unseen via
Semantic Scaffolding**. No publication venue or DOI is claimed here. Cite the
repository URL for the software until bibliographic details are available.
