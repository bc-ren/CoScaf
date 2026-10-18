# CoScaf

## Installation

Python 3.12; use a CUDA-compatible PyTorch installation for GPU runs.

```bash
git clone https://github.com/bc-ren/CoScaf.git
cd CoScaf
python -m venv .venv
source .venv/bin/activate
pip install -e '.[retina]'
```

## Data

Google Drive: **[link to be added]**

Extract `CoScaf-assets/` alongside this repository. Use the included
`data_splits/` and the CUB/RetiRare-74 caches. AWA2 and SUN require their original
images to generate caches.

## Run

### Saved checkpoint inference

Set `MODEL=CUB` or `MODEL=RetiRare-74`.

```bash
MODEL=CUB
python -m coscaf.historical --mode verify \
  --manifest manifests/$MODEL.json --assets ../CoScaf-assets
python -m coscaf.historical --mode replay \
  --manifest manifests/$MODEL.json --assets ../CoScaf-assets \
  --output outputs/$MODEL-replay --device cuda:0
```

### Training and evaluation

The current training configuration omits anchor losses; saved checkpoints use
the inference command above. Run the following in Bash, choosing `AWA2`, `CUB`,
`SUN`, or `RetiRare74`:

```bash
DATASET=CUB
ASSETS=../CoScaf-assets
DATA=data_splits/$DATASET
CONFIG=configs/$DATASET-best.json
CACHE=$ASSETS/cache/$DATASET
RUN=runs/$DATASET
BACKBONE=(--vit-snapshot "$ASSETS/models/backbones/vit_base_in21k")
if [ "$DATASET" = RetiRare74 ]; then
  CONFIG=configs/RetiRare-74-best.json
  BACKBONE=(--retizero-source "$ASSETS/code/retinal/third_party/RetiZero_d72aadc"
    --retizero-checkpoint "$ASSETS/models/backbones/retizero/RetiZero.pth"
    --bert-path "$ASSETS/models/backbones/clinicalbert")
fi

coscaf develop --config "$CONFIG" --data "$DATA" --cache "$CACHE/train-portable" \
  "${BACKBONE[@]}" --output "$RUN/dev" --device cuda:0
coscaf train --config "$CONFIG" --data "$DATA" --cache "$CACHE/train-portable" \
  "${BACKBONE[@]}" --development "$RUN/dev" --output "$RUN/model" --device cuda:0
coscaf evaluate --run "$RUN/model" --data "$DATA" --cache "$CACHE/test" \
  "${BACKBONE[@]}" --output "$RUN/test" --device cuda:0
```

For AWA2/SUN, generate the train cache **before** `develop` and the test cache
**after** `train`, using the variables above. `IMAGE_ROOT` must resolve the paths
in `data_splits/$DATASET/images.jsonl`.

```bash
IMAGE_ROOT=/path/to/dataset
coscaf cache --role train --config "$CONFIG" --data "$DATA" \
  --image-root "$IMAGE_ROOT" "${BACKBONE[@]}" --output "$CACHE/train-portable"
coscaf cache --role test --config "$CONFIG" --data "$DATA" --run "$RUN/model" \
  --image-root "$IMAGE_ROOT" "${BACKBONE[@]}" --output "$CACHE/test"
```
