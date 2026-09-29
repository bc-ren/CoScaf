# Data preparation

This repository contains code and synthetic tests, not images, pretrained
weights, per-image records, semantic matrices, or feature caches. Obtain each
dataset and backbone from its provider, observing its applicable terms. Keep
all real data outside the Git checkout or in ignored directories.

## Portable bundle

Each dataset uses one directory:

```text
dataset/
  metadata.npz
  provenance.json
  images.json               # optional: needed to extract prefixes from images
  folds/
    fold_0.json
    ...
  cache/
    train/{raw.npy,indices.npy,manifest.json}
    test/{raw.npy,indices.npy,manifest.json}
```

`metadata.npz` contains these arrays, loaded with `allow_pickle=False`:

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `labels` | `(N,)`, integer | Zero-based class index for each image |
| `semantics` | `(C,A)`, floating point | Fixed class vectors, ordered by class index |
| `train` | `(N_train,)`, integer | Original seen training rows |
| `test_seen` | `(N_seen,)`, integer | Fixed official seen-test rows |
| `test_unseen` | `(N_unseen,)`, integer | Fixed official unseen-test rows |
| `class_names` | `(C,)`, Unicode, optional | Unique class names in semantic-row order |
| `groups` | `(N,)`, Unicode/integer, optional | Anonymous leakage-group identifiers |

All image indices address the same full image table. Do not renumber test rows
after filtering training data. Seen/unseen class lists are inferred from the
declared membership. Every semantic row must occur in this class partition.
`images.json`, when used, is a JSON list of relative image paths in the same
row order; the extraction command takes the image root separately.

```python
from coscaf.data import DatasetBundle, PrefixCache, load_fold

data = DatasetBundle.load("/path/to/dataset")
fold = load_fold(data, data.root / "folds/fold_0.json")
cache = PrefixCache(data.root / "cache/train", data, "train")
tokens = cache.take(fold["fit"][:8])
```

The loader rejects duplicate/out-of-range indices, overlapping roles, unseen
training classes, invalid vectors, inconsistent class sets, and group leakage.
When group IDs are unavailable, row separation alone does **not** establish
patient-level independence.

## AWA2, CUB, and SUN

Use the official proposed image split in `att_splits.mat`. The label source may
be `res101.mat` or another original MATLAB file with the same ordered `labels`;
its visual features are not consumed. AWA2 uses 85D attributes, CUB uses the
1024D `att` matrix in `sent_splits.mat`, and SUN uses 102D attributes.

```python
from coscaf.preparation import import_natural, make_development_folds
import json

data = import_natural(
    "/data/CUB/res101.mat",
    "/data/CUB/att_splits.mat",
    "/data/prepared/CUB",
    dataset="CUB",
    semantics_file="/data/CUB/sent_splits.mat",
    images_file="/data/CUB/images.json",
)
folds = make_development_folds(data, nfolds=4)
(data.root / "folds").mkdir(exist_ok=True)
for index, fold in enumerate(folds):
    (data.root / "folds" / f"fold_{index}.json").write_text(json.dumps(fold))
```

The importer converts MATLAB one-based labels and image indices **once**, and
transposes semantics from `(A,C)` to `(C,A)`. For AWA2/SUN, entries marked
negative in `original_att` are set to zero. CUB's names and all five split
arrays must match `att_splits.mat` exactly, including order; there is no silent
312D fallback. Official test arrays are copied without sorting or resampling.

An optional `exclude_indices` argument accepts a `.npy` integer vector or JSON
integer list of **zero-based original training rows** quarantined by a duplicate
audit. It cannot exclude test rows. The output records every exclusion in
`excluded_train` and `provenance.json`. Use the exact documented quarantine
list to reproduce a run that excluded duplicates; an arbitrary list changes
the training protocol. The import does not invent or infer exclusions.

Development folds use only original seen training images. The fixed natural
protocol has two class-partition families: random classes and contiguous
classes along the leading semantic principal component. Set `nfolds` to
5/4/10 for AWA2/CUB/SUN, producing 10/8/20 folds. The supplied best recipes use
fold IDs `[0,5]`, `[0,4]`, and `[0,10]`, respectively: one fold from each family.
Class and image randomization seeds are fixed independently of training seed.
Alternatively import the exact original fold JSON files.

Each fold contains global image-index lists `fit`, `cal`, `tune`, optionally
`confirm`, plus class-index lists `pseudo_seen` and `pseudo_unseen`. These two
class lists partition the original seen classes. `fit` contains only
pseudo-seen classes. No development role may contain official test images.
Calibration uses `cal`; model/configuration selection uses `tune`. Keep these
roles unchanged across seeds.

## RetiRareV2

Provide a local, de-identified export with the same NPZ schema and the fixed
74-class split (59 seen, 15 unseen; 3,428 training and 1,972 test images for
the reported dataset). The general importer does not hard-code dataset counts,
so a separately documented subset can use the same interface.

```python
from coscaf.preparation import import_retinal

data = import_retinal(
    "/data/retinal_export.npz",
    "/data/prepared/RetiRareV2",
    images_file="/data/retinal_images.json",
)
```

Supply the original five development folds as `folds/fold_0.json` through
`fold_4.json`. Grouped retinal records require supplied group-disjoint folds;
the natural-image fold generator deliberately rejects grouped data. Preserve
anonymous leakage groups when exporting. Do not publish patient identifiers,
clinical records, image filenames, or image-level metadata in this repository.

The command-line importer validates and copies the supplied fold JSON files:

```bash
coscaf prepare-retinal \
  --metadata /data/retinal_export.npz \
  --images /data/retinal_images.json \
  --folds /data/fixed_retinal_folds \
  --output /data/prepared/RetiRareV2
```

The reported configuration uses fixed **512D disease-name embeddings** from
the complete official RetiZero text encoder and its trained projection head.
Standalone ClinicalBERT vectors and clinical-description embeddings are not
equivalent. Names use the template `a fundus image of {standardized_candidate_name}.` Preserve the
ordered, reviewed name inputs used to make the original embeddings. Encode
one name at a time with the official tokenizer/pooling, respecting its 77-token
limit, then L2-normalize each projected vector. No image labels enter text
encoding. RetiZero's native image-text similarity is not added to CoScaf scores.

## Frozen-prefix caches

Joint visual prompts require patch tokens, not a final CLS feature vector.
Existing `vit_features.mat` 768D vectors cannot replace these caches.

| Dataset | Backbone | Resolution | Frozen prefix | Raw token shape |
| --- | --- | ---: | --- | --- |
| AWA2/CUB/SUN | `google/vit-base-patch16-224-in21k` | 384 | After block 10, before terminal LayerNorm | `(N,577,768)` |
| RetiRareV2 | Official RetiZero | 224 | After block 22, before terminal normalization | `(N,197,1024)` |

Keep the pretrained backbone frozen. Google ViT is pinned to revision
`b4569560a39a0f1af58e3ddaf17facf20ab919b0`; images are converted to RGB,
resized bilinearly, and normalized with mean/std 0.5. Positional embeddings
are interpolated for 384px inputs. The original cache used Pillow 10.4.0 with
IJG JPEG 9; different JPEG decoders can change pixels and final results.
Record the decoder, library versions, checkpoint hash, and preprocessing in
the cache manifest when regenerating features.

RetiZero uses its official checkpoint, including its pretrained LoRA weights;
no new backbone fine-tuning is performed. ImageNet mean/std preprocessing is
used at 224px. Obtain the RetiZero source/checkpoint separately from its
official provider; no third-party model weights or source are bundled here.

`raw.npy` must be float32, and `indices.npy` must exactly equal `data.train`
for the train cache or `data.test` (seen followed by unseen) for the test cache.
Prefer the cache command, which records the actual backbone checkpoint,
preprocessing, and environment in the manifest:

```bash
coscaf cache \
  --config configs/cub.json \
  --data /data/prepared/CUB \
  --image-root /data/CUB/images \
  --role train \
  --output /data/prepared/CUB/cache/train \
  --device cuda:0
```

This checks row membership, shape, finite values, and hashes. The cache reader
checks these hashes by default and refuses requests outside its role. Fit the
whitening/ridge initializer only on the current fold's `fit` rows or on the
final seen training set. Test features are available only to evaluation and
unlabeled transductive adaptation after configuration/calibration are frozen;
their labels must never enter the adaptation objective or model selection.
