# Companion assets

The `CoScaf-assets` folder is prepared separately for manual upload to Google
Drive. A download URL can be added after that upload. The Git repository stores
code, configurations, numeric splits and manifests, not the large model/cache
files.

| Contents | Scope |
|---|---|
| FP32 train/test prefix caches | All registered CUB and RetiRare-74 rows |
| Checkpoints | Original source model and saved adapted generator for each paper H-selected TTA run |
| Prediction banks | Original source/adapted test scores and row indices for verification |
| Backbones | Pinned Google ViT and complete RetiZero, with required configurations and source dependencies |
| Setup data | Four original metadata files, all 43 folds, initializer means and anonymous image manifests |
| Integrity | File SHA-256 inventory and per-model replay manifests |

CUB prefixes occupy 20,891,203,840 bytes; RetiRare-74 prefixes occupy
4,357,325,056 bytes. The complete original-asset inventory is approximately
27.6 GB. The final delivery inventory records exact sizes and checksums.

Keep all asset paths intact when extracting or moving the folder. Each
`configs/<dataset>/historical-replay.json` resolves paths relative to `--assets`.
Run `--mode verify` once after placement. Do not use a partial download for
inference: every large file must match its SHA-256.

Raw image pixels, private retinal source filenames and patient mappings are not
included. Retinal row IDs remain in the original 5,607-row index space; 207
excluded rows are not used by the train/test caches.

Saved checkpoint replay reads source/adapted weights directly and returns both
endpoints. New no-anchor training uses separate outputs and its own DEV-derived
calibration and epoch horizon.
