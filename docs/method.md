# Method and evaluation

## Joint source learning

Sixteen prompt tokens are inserted after CLS in each of the frozen encoder's
final two blocks and removed after each block. Earlier and final normalized
patch tokens are averaged. Prompts change the evidence, not its dimensionality.

Unprompted source-training features determine a fixed coordinate transform and
analytic ridge initialization. Whitening uses only the current training
partition. Class means initialize a semantic center map; residual targets
initialize shared mode maps. These computations use no prior task checkpoint
and are not a separate optimizer stage.

Every seen and unseen class uses the same affine semantic generator. Residuals
are centered within each class, and prototypes are unit normalized. No free
class-specific prototypes are learned.

Three evidence terms form the score:

1. **Global:** patch mean, fixed coordinate transform, cosine similarities to
   prototypes, and normalized log-sum-exp over modes.
2. **Local:** low-rank semantic class/mode queries attend to image patches.
   Weighted log-sum-exp combines similarities. AWA2 uses compatibility-based
   mode weights; other datasets use an independent patch-evidence prior.
3. **Regional:** first and second moments of a fixed rank-16 projection over
   the whole grid and four quadrants produce a 760D descriptor. A small semantic
   readout gives a bounded score correction.

The global/local mixture weight is dataset specific. Source learning minimizes
class-weighted cross-entropy plus a generator-initialization anchor. One AdamW
optimizer has component-specific learning rates. Dropout is disabled, gradients
are enabled, and every source image is visited exactly once per epoch.

## Generator-only adaptation

Earlier and final layer predictions supply deterministic views of each image.
Cross-layer agreement, disagreement, and margins determine reliability; SUN
uses semantic-neighborhood agreement. Reliability is rescaled to the feasible
partial-mass budget and capped at one. The implementation falls back to equal
reliability if all raw reliabilities vanish.

Partial transport preserves sample mass while softly regulating class mass
within the seen and unseen groups. Class and mode targets are computed once.
Some configurations combine global and local evidence in mode targets.
A copy of the source generator is then updated using class/mode alignment,
teacher consistency, semantic-neighbor ranking, generator anchoring,
semantic-relation preservation, and source-topology preservation. The exact
coefficients are in each configuration.

All other parameters and buffers stay fixed. Local scores and routing are
recomputed after each update. Every optimizer step aggregates the whole target
set; `adaptation_batch` determines memory use, not the statistical protocol.
The original source model is retained. No target sample labels or test-fitted
calibration enter the adaptation API.

## Calibration and metrics

A scalar offset is subtracted from seen-class logits. Original-seen development
CAL data fit the source and adapted offsets separately, with equal fold and
class weights. TUNE provides development metrics. Choices are frozen before
official evaluation.

`S` and `U` are average per-class seen and unseen accuracies when all classes
compete. `H = 2SU / (S + U)`. `CZSL` restricts both images and candidate classes
to unseen classes; it is distinct from GZSL `U`.

The S–U curve enumerates margin thresholds where the winning group changes.
AUSUC integrates S against increasing U by trapezoids. Global class-index order
determines ties. Test-curve maximum H is a diagnostic, never the reported
calibrated working point. ECE uses 15 bins. ECE and Brier use equal seen/unseen
group mass and equal class weights within groups. Probability temperature is
also fitted using CAL only. JSON stores all metrics as fractions.
