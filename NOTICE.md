# Assets and attribution

CoScaf source code is distributed separately from model weights and image caches.
The release's numeric split/semantic metadata retains the original dataset class
and sample ordering. Original dataset images are not included. Dataset access
and use remain subject to the respective providers' terms.

The companion asset package uses these upstream components:

- Google [ViT-Base/16](https://huggingface.co/google/vit-base-patch16-224-in21k),
  revision `b4569560a39a0f1af58e3ddaf17facf20ab919b0`.
- [RetiZero](https://github.com/LooKing9218/RetiZero), revision
  `d72aadc692fbe33b182c79711bccb397edffb419`, with its official checkpoint and
  ClinicalBERT configuration/tokenizer. Preserve the upstream notices included
  with that source. CoScaf does not grant additional rights to upstream assets.

PyTorch, Transformers, NumPy, SciPy and Pillow are installed dependencies and
retain their respective licenses. Cached retinal features use anonymous sample
indices; private source-file and patient mappings are excluded.
