# Runtime and numerical reproducibility

CoScaf uses a pinned pretrained backbone and fixed dataset semantics. The public
code retains the source implementation's readout equations, initialization,
class weighting, optimizer groups, minibatch ordering, and generator-only
adaptation. Checkpoints and datasets are obtained separately.

## Archived experiment environments

The archived records describe distinct preprocessing and training environments;
they should not be treated as one interchangeable package set.

| Component | Recorded version / configuration |
|---|---|
| Source-training Python | 3.12.3 |
| Source-training PyTorch | 2.5.1+cu121 |
| Source-training NumPy | 2.4.4 |
| Source-training SciPy | 1.18.0 |
| Transformers | 5.13.1 |
| Pixel preparation PyTorch / torchvision | 2.13.0+cpu / 0.28.0+cpu |
| Frozen-prefix extraction PyTorch | 2.13.0+cu132 |
| Pillow | 10.4.0; AWA2 pixel records specify the IJG JPEG 9.0 decoder |
| Hardware | NVIDIA RTX 5000 Ada, 32 GB |

These version strings are reported verbatim from archived environment and cache
manifests. The prefix cache records specify FP32 inference with TF32 disabled,
with positional interpolation enabled at 384 pixels. Recomputing image caches
with another JPEG decoder, resizing implementation, tensor-library version, or
attention backend can change floating-point results. Preserve cache hashes when
replaying archived checkpoints.

## Public-release validation environment

The source-only test suite was run on CPU with Python 3.12.14, PyTorch 2.14.0,
NumPy 2.5.3, SciPy 1.18.1, Transformers 5.13.1, Pillow 12.3.0, and pytest 9.1.1.
The backbone wrapper accepts both the older `encoder.layer` interface and the
newer `layers` interface. Actual Hugging Face ViT prefix extraction, late-block
prompt gradients, and unprompted suffix replay were tested with randomly
initialized architecture-compatible weights under Transformers 5.13.1, without
model downloads. Earlier module checks also ran under 4.44.2; this is not a
claim of bitwise equality between different Transformers versions.

The release audit compared the original and independent mode-prior readouts
against the archived equations for K = 1, 2, 3, and 8, including parameter
initialization, scores, evidence, and gradients. These comparisons were exact in
the same CPU environment. Preprocessing and ridge initialization were also exact
for normalization and two whitening configurations. Dataset-scale training and
image-cache extraction were not rerun as part of packaging the release; the
reported benchmark results remain the archived experimental measurements.

## Backbone integrity

Natural-image experiments use `google/vit-base-patch16-224-in21k` at revision
`b4569560a39a0f1af58e3ddaf17facf20ab919b0`. The loader checks the safetensors SHA-256:

```text
fd4e1169c7aa6c2dbfa8a6448be13b35abc0ee256190857c90009d12c094619b
```

The retinal loader expects the separate official RetiZero source at commit
`d72aadc692fbe33b182c79711bccb397edffb419`. If Git metadata are present, it verifies
the commit. It strictly loads the complete checkpoint and records its SHA-256.
Its pretrained LoRA weights remain frozen. They are not newly fitted task weights.

For each new run, record the actual Python/package versions, backbone and cache
hashes, GPU model, precision settings, effective/microbatch sizes, and random seed.
Source task parameters are initialized afresh; adaptation starts from that run's
own source generator.
