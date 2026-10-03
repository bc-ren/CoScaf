"""CoScaf evidence encoder and class-shared prototype scaffold."""

import math

import torch
from torch import nn
from torch.nn import functional as F

GENERATOR = ("center_weight", "center_bias", "mode_weight", "mode_bias")


def prototypes(parameters, semantics):
    a = F.normalize(semantics, dim=-1)
    center = a @ parameters["center_weight"] + parameters["center_bias"]
    residual = torch.einsum("ca,akd->ckd", a, parameters["mode_weight"]) + parameters["mode_bias"]
    residual = residual - residual.mean(1, keepdim=True)
    return center, F.normalize(center[:, None] + residual, dim=-1)


def _branch_scores(global_similarity, local_similarity, modes, prior=None):
    route = (local_similarity / 0.5).softmax(-1) if prior is None else None
    if prior is None:
        prior = route.clamp_min(1e-12).log()
    local = 0.1 * torch.logsumexp(local_similarity / 0.1 + prior, -1)
    global_ = 0.1 * (torch.logsumexp(global_similarity / 0.1, -1) - math.log(modes))
    return global_, local, prior, route


def regional_moments(patch, projection):
    b, n, d = patch.shape
    side = math.isqrt(n)
    assert side * side == n and side % 2 == 0
    grid = patch.reshape(b, side, side, d)
    parts = [patch] + [
        grid[:, i : i + side // 2, j : j + side // 2].reshape(b, -1, d)
        for i in (0, side // 2)
        for j in (0, side // 2)
    ]
    projected = [p @ projection for p in parts]
    means = torch.stack([p.mean(1) for p in projected], 1)
    centered = [p - p.mean(1, keepdim=True) for p in projected]
    cov = torch.stack([p.transpose(1, 2) @ p / p.shape[1] for p in centered], 1)
    i, j = torch.triu_indices(projection.shape[1], projection.shape[1], device=patch.device)
    cc = cov[..., i, j] * torch.where(i == j, 1.0, math.sqrt(2.0)).to(patch)
    # Smooth signed square root has a finite derivative at zero.
    cc = cc / (cc.abs() + 1e-6).sqrt()
    return torch.cat(
        [F.normalize(means.flatten(1), dim=-1), F.normalize(cc.flatten(1), dim=-1)], -1
    )


class _BaseReadout(nn.Module):
    def __init__(
        self,
        semantics,
        initializer,
        visual_mean,
        visual_matrix,
        query_rank=8,
        moment_rank=16,
        local_weight=0.25,
        native=False,
    ):
        super().__init__()
        a, d = initializer["weight"].shape
        self.native, self.local_weight = bool(native), float(local_weight)
        self.register_buffer("semantics", semantics.detach().clone())
        self.register_buffer("visual_mean", visual_mean.detach().clone())
        self.register_buffer("visual_matrix", visual_matrix.detach().clone())
        self.center_weight = nn.Parameter(initializer["weight"].clone())
        self.center_bias = nn.Parameter(initializer["bias"].clone())
        self.mode_weight = nn.Parameter(initializer["residual_weight"].clone())
        self.mode_bias = nn.Parameter(initializer["residual_bias"].clone())
        self.modes = self.mode_weight.shape[1]
        self.query_low = nn.Linear(a, query_rank, bias=False)
        self.query_high = nn.Linear(query_rank, self.modes * d, bias=False)
        nn.init.normal_(self.query_high.weight, std=0.02)
        # Fixed, label-independent projection: no PCA or extra learned stage.
        gen = torch.Generator(device="cpu").manual_seed(1947)
        projection = torch.linalg.qr(torch.randn(d, moment_rank, generator=gen), mode="reduced").Q
        self.register_buffer("projection", projection.to(semantics))
        size = 5 * (moment_rank + moment_rank * (moment_rank + 1) // 2)
        self.moment = nn.Linear(size, a, bias=False)
        nn.init.zeros_(self.moment.weight)
        self._anchor_names = GENERATOR
        for name in self._anchor_names:
            self.register_buffer("initial_" + name, getattr(self, name).detach().clone())

    def visual(self, value):
        z = F.normalize(value, dim=-1) - self.visual_mean
        if self.visual_matrix.numel():
            z = z @ self.visual_matrix
        return F.normalize(z, dim=-1)

    def prototypes(self, ids):
        return prototypes({k: getattr(self, k) for k in GENERATOR}, self.semantics[ids])[1]

    def _local_attention(self, patch, a):
        q = self.query_high(self.query_low(a)).reshape(len(a), self.modes, patch.shape[-1])
        return (
            torch.einsum("cmd,bnd->bcmn", q, patch) / (math.sqrt(patch.shape[-1]) * 0.2)
        ).softmax(-1)

    def _moment_evidence(self, patch, a):
        return 0.1 * torch.tanh(self.moment(regional_moments(patch, self.projection)) @ a.T)

    def forward(self, patch, ids, native_embedding=None, return_usage=False):
        # Preserve the direct path's operation order for exact gradient replay.
        a = F.normalize(self.semantics[ids], dim=-1)
        p = self.prototypes(ids)
        global_z = torch.einsum("bd,ckd->bck", self.visual(patch.mean(1)), p)
        att = self._local_attention(patch, a)
        local = self.visual(torch.einsum("bcmn,bnd->bcmd", att, patch))
        local_z = (local * p[None]).sum(-1)
        # Input-conditioned, differentiable mode routing, within this same model.
        global_score, local_score, _, route = _branch_scores(global_z, local_z, self.modes)
        moments = self._moment_evidence(patch, a)
        if self.native:
            assert native_embedding is not None and native_embedding.shape[-1] == a.shape[-1]
            score = F.normalize(native_embedding, dim=-1) @ a.T
            score = score + self.local_weight * (local_score - global_score) + moments
        else:
            score = (
                (1 - self.local_weight) * global_score + self.local_weight * local_score + moments
            )
        return (score, route) if return_usage else score

    def anchor_loss(self):
        terms = []
        for name in self._anchor_names:
            start = getattr(self, "initial_" + name)
            terms.append(
                (getattr(self, name) - start).square().mean()
                / start.square().mean().clamp_min(1e-6)
            )
        return torch.stack(terms).mean()


class CoScafModel(nn.Module):
    """A prompted frozen suffix and a shared semantic readout."""

    def __init__(self, suffix, head):
        super().__init__()
        self.suffix, self.head = suffix, head

    def forward(self, raw, ids):
        features = self.suffix(raw)
        patch, native = features if isinstance(features, tuple) else (features, None)
        return self.head(patch, ids, native)

    def trainable_state(self):
        # Only task state is saved; pretrained backbone weights are loaded separately.
        return {"head": self.head.state_dict(), "prompt": self.suffix.adapter_state()}

    def load_trainable_state(self, state):
        self.head.load_state_dict(state["head"], strict=True)
        self.suffix.load_adapter(state["prompt"])


class CoScafReadout(_BaseReadout):
    """Three-branch matching with shared semantic-conditioned prototypes.

    ``original`` routes modes using local prototype compatibility; ``latent``
    uses a separately learned semantic-query prior from the same image patches.
    Global and local branches share prototypes. The regional branch supplies a
    bounded residual score. All semantic rows remain fixed buffers.
    """

    def __init__(self, *args, variant="original", prior_temperature=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        assert variant in ("original", "latent")
        self.variant = variant
        self.prior_temperature = prior_temperature
        if variant == "latent":
            self.prior_low = nn.Linear(self.semantics.shape[1], 8, bias=False)
            self.prior_high = nn.Linear(8, self.modes * self.center_weight.shape[1], bias=False)
            nn.init.normal_(self.prior_high.weight, std=0.02)

    def features(self, patch, ids):
        a = F.normalize(self.semantics[ids], dim=-1)
        d = patch.shape[-1]
        att = self._local_attention(patch, a)
        out = {
            "global": self.visual(patch.mean(1)),
            "local": self.visual(torch.einsum("bcmn,bnd->bcmd", att, patch)),
            "moment": self._moment_evidence(patch, a),
        }
        if self.variant == "latent":
            pq = self.prior_high(self.prior_low(a)).reshape(len(ids), self.modes, d)
            energy = torch.einsum("cmd,bnd->bcmn", pq, patch) / (math.sqrt(d) * 0.2)
            out["prior"] = (
                torch.logsumexp(energy, -1) - math.log(patch.shape[1])
            ) / self.prior_temperature
        return out

    def score_features(self, feat, ids, parameters=None, diagnostics=False):
        pars = parameters or {k: getattr(self, k) for k in GENERATOR}
        (_, p) = prototypes(pars, self.semantics[ids])
        global_similarity = torch.einsum("bd,ckd->bck", feat["global"], p)
        local_similarity = (feat["local"] * p[None]).sum(-1)
        prior = feat["prior"].log_softmax(-1) if self.variant == "latent" else None
        gs, ls, prior, _ = _branch_scores(global_similarity, local_similarity, self.modes, prior)
        z = (1 - self.local_weight) * gs + self.local_weight * ls + feat["moment"]
        if diagnostics:
            return z, {
                "local_responsibility": (local_similarity / 0.1 + prior).softmax(-1),
                "global_responsibility": (global_similarity / 0.1).softmax(-1),
            }
        return z

    def forward(self, patch, ids, native_embedding=None, return_usage=False):
        if self.variant == "original":
            return super().forward(patch, ids, native_embedding, return_usage)
        (z, extra) = self.score_features(self.features(patch, ids), ids, diagnostics=True)
        return (z, extra["local_responsibility"]) if return_usage else z
