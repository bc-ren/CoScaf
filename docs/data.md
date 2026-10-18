# Data and original splits

`data_splits/` includes the exact ORIX `metadata.npz`, all 43 development folds,
and the four-field image manifest for each dataset. SHA-256 checksums are in
`data_splits/SHA256.json`. Image pixels are not distributed.

| Dataset | Seen / unseen classes | Train | Test seen / unseen | Folds | Semantics |
|---|---:|---:|---:|---:|---:|
| AWA2 | 40 / 10 | 23,467 | 5,882 / 7,913 | 10 | 85D |
| CUB | 150 / 50 | 7,055 | 1,764 / 2,967 | 8 | 1024D |
| SUN | 645 / 72 | 10,290 | 2,580 / 1,440 | 20 | 102D |
| RetiRare-74 | 59 / 15 | 3,428 | 862 / 1,110 | 5 | 512D |

Metadata stores zero-based labels, class semantics, original train/test indices
and seen/unseen class IDs. Fold NPZ files store `fit`, `cal`, `tune`,
`pseudo_seen` and `pseudo_unseen`. Original row numbers and test ordering are
preserved, including RetiRare-74's interleaved test order. Excluded images remain
in the index space and do not enter training or evaluation. Do not compact the
RetiRare-74 metadata from 5,607 rows to its 5,400 selected samples.

Each `images.jsonl` row contains only `index`, `image_path`, `class_id` and
`sha256`. RetiRare-74 uses anonymous `RRV2_######` filenames. Original case/source
filenames and private patient mappings are not included. These splits establish
image/content separation; no patient-disjoint claim is made.

## Cached features

The separate asset package contains all train/test FP32 prefix features for CUB
and RetiRare-74, original initializer means, indices and role-aware manifests.
Prefixes still require the last two frozen backbone blocks during inference.

| Dataset | Train rows | Test rows | Prefix shape per image | Prompt blocks |
|---|---:|---:|---|---|
| CUB | 7,055 | 4,731 | 577 x 768, block 10 | 11–12 |
| RetiRare-74 | 3,428 | 1,972 | 197 x 1024, block 22 | 23–24 |

The manifest's `array` field identifies the original NPY filename. The loader
checks exact row order, metadata identity, dtype, shape and checksums. Training
can read only the train cache; development folds contain no official test rows.
Initializer whitening and ridge fitting use only each current fit partition.

## RetiRare-74 commands

Use the README's training commands with the retinal configuration and
`--data data_splits/RetiRare74`, then append these arguments instead of
`--vit-snapshot`:

```bash
--retizero-source ../CoScaf-assets/code/retinal/third_party/RetiZero_d72aadc \
--retizero-checkpoint ../CoScaf-assets/models/backbones/retizero/RetiZero.pth \
--bert-path ../CoScaf-assets/models/backbones/clinicalbert
```

The supplied backbone includes the official pretrained LoRA weights, which
remain frozen. Its text encoder produced the fixed 512D disease-name semantics;
standalone ClinicalBERT vectors are not interchangeable.

## AWA2 and SUN

Obtain the original images separately, matching the supplied image manifest.
Use the existing `coscaf cache` command and the original `data_splits/<dataset>`
directory. Natural images use the pinned Google ViT at 384 x 384; RetiZero uses
224 x 224. Supply an image root under which each relative `image_path` resolves.
Do not run `prepare-natural` to replace the included registered splits. That
command remains available for new datasets/protocols.

Test cache extraction requires a completed run via `--run`, after development
and source training have been frozen. Library/JPEG versions can affect freshly
extracted features; use published cache hashes to identify the archived features.
