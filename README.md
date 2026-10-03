# CoScaf

Adapting in Concert: Recognizing the Unseen via Semantic Scaffolding

## Environment

- Linux, Python 3.12, NVIDIA GPU. Experiments used RTX 5000 Ada (32 GB).
- Training environment: PyTorch 2.5.1 with CUDA 12.1; Transformers 5.13.1.
- NumPy, SciPy and Pillow are installed with the package.
- AWA2/CUB/SUN use frozen `google/vit-base-patch16-224-in21k` at 384 × 384.
  RetiRare-74 uses the official RetiZero backbone at 224 × 224.

Create an environment and install the package (CUDA 12.1 example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e '.[dev]'
pytest -q
```

Data and pretrained weights are obtained separately. See [data setup](docs/data.md)
for the input format and RetiZero installation. Keep the supplied configurations
in `configs/`; CUB requires 1024D semantics from `sent_splits.mat`.

## Training

Run from the repository root. The example below uses AWA2. For other datasets,
follow [data setup](docs/data.md), then substitute their configuration and paths.

```bash
coscaf prepare-natural --dataset AWA2 \
  --labels /data/AWA2/res101.mat --splits /data/AWA2/att_splits.mat \
  --images /data/AWA2/images.json --output data/awa2

coscaf cache --config configs/awa2.json --data data/awa2 \
  --image-root /data/AWA2/JPEGImages --role train \
  --output cache/awa2/train --device cuda:0

coscaf develop --config configs/awa2.json --data data/awa2 \
  --cache cache/awa2/train --output runs/awa2/validation --device cuda:0

coscaf train --config configs/awa2.json --data data/awa2 \
  --cache cache/awa2/train --development runs/awa2/validation \
  --output runs/awa2/model --device cuda:0
```

`develop` fits calibration offsets on held-out seen-class folds. `train` then
initializes all task parameters afresh and jointly trains prompts, evidence
modules and the generator. The pretrained backbone stays frozen.
Prefix caches contain patch tokens, not the 768D vectors in `vit_features.mat`.

## Testing

```bash
coscaf cache --config configs/awa2.json --data data/awa2 \
  --image-root /data/AWA2/JPEGImages --role test --run runs/awa2/model \
  --output cache/awa2/test --device cuda:0

coscaf evaluate --run runs/awa2/model --data data/awa2 \
  --cache cache/awa2/test --output runs/awa2/test --device cuda:0
```

`metrics.json` contains S, U, H, CZSL and AUSUC for the model before and after
TTA. Values are fractions (multiply by 100 for percentages). Scores and S–U
curves are saved alongside it. TTA updates only the generator using the
unlabeled test set; test labels are used only to calculate metrics.

Use a new output directory for each run. For parallel runs, select `cuda:0` or
`cuda:1`. A data-free workflow check is available with
`coscaf smoke --output /tmp/coscaf-smoke`.
