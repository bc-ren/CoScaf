# RetiRareV2 with RetiZero

The registered dataset has 74 classes (59 seen, 15 unseen), 3,428 training
images, and 1,972 fixed test images. The clinical dataset and sample metadata
are not distributed. Use an authorized export with its original class order,
splits, and anonymous leakage groups. Groups must not cross training/test or
development roles. Missing patient mappings do not establish patient-level
exclusivity.

Obtain the official source and checkpoint from
[RetiZero](https://github.com/LooKing9218/RetiZero) separately, following its
dependency and model-access instructions:

```bash
git clone https://github.com/LooKing9218/RetiZero.git third_party/RetiZero
git -C third_party/RetiZero checkout d72aadc692fbe33b182c79711bccb397edffb419
python -m pip install -e '.[retina]'
```

The loader initializes the official architecture from the local text-model
configuration, then loads the complete checkpoint strictly. It does not
substitute standalone language-model weights. Pretrained visual LoRA stays
frozen. The CoScaf-Best readout uses 1024D patches and fixed 512D semantics,
without adding RetiZero's native global image–text score.

## Class semantics

The registered semantics are disease-name vectors from the template
`a fundus image of {standardized_candidate_name}.`, using the complete official
text encoder and learned 512D projection. Preserve existing vectors when
available. Name vectors and clinical-description vectors are distinct.

If recreating vectors, use official tokenization/pooling at batch size one,
without artificial padding, and respect the 77-token limit. Preserve class
row order. The release imports fixed vectors; it does not generate diagnoses
or descriptions.

## Workflow

```bash
coscaf prepare-retinal --metadata /path/to/authorized_metadata.npz \
  --images /path/to/image_paths.json --folds /path/to/fixed_folds \
  --output data/retirarev2
```

Follow the README workflow with `configs/retirarev2.json`. Add these arguments
to each `cache`, `develop`, `train`, and `evaluate` command:

```bash
--retizero-source third_party/RetiZero \
--retizero-checkpoint /path/to/official_checkpoint.pth \
--bert-path /path/to/official_text_tokenizer_and_config
```

Caching uses the official image preprocessor and block-22 tokens at 224 pixels;
prompts train in blocks 23–24. Projected 512D image embeddings are insufficient
for joint prompt training. Images, metadata, names, and feature caches stay local.
