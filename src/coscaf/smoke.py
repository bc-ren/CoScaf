"""Small synthetic end-to-end exercise, separate from reported experiments."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import DatasetBundle
from .io import write_json
from .pipeline import develop, evaluate_run, train
from .preparation import write_prefix_cache_manifest


class SyntheticSuffix(nn.Module):
    """Tiny differentiable stand-in for testing orchestration without downloads."""

    def __init__(self, device):
        super().__init__()
        self.prompts = nn.Parameter(torch.randn(2, 2, 32, device=device) * 0.01)
        self.enabled = True

    def layer_views(self, raw):
        early = F.normalize(raw, dim=-1)
        late = F.normalize(raw + self.prompts.mean((0, 1)), dim=-1) if self.enabled else early
        return (early + late) * 0.5, early, late

    def forward(self, raw):
        return self.layer_views(raw)[0]

    def adapter_state(self):
        return {"prompts": self.prompts.detach().cpu().clone()}

    def load_adapter(self, state):
        with torch.no_grad():
            self.prompts.copy_(state["prompts"].to(self.prompts))


def run_smoke(output):
    """Exercise split checks, fresh fits, calibration freeze, restore, and TTA."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rng = np.random.default_rng(91)
    labels = np.repeat(np.arange(6), 20)
    semantics = rng.normal(size=(6, 7))
    image_features = rng.normal(size=(6, 32))
    raw = (image_features[labels, None] + rng.normal(scale=0.2, size=(120, 16, 32))).astype(
        np.float32
    )
    fit = np.concatenate([np.arange(c * 20, c * 20 + 16) for c in range(4)])
    seen_test = np.concatenate([np.arange(c * 20 + 16, c * 20 + 20) for c in range(4)])
    unseen_test = np.arange(80, 120)
    data = output / "data"
    (data / "folds").mkdir(parents=True)
    np.savez_compressed(
        data / "metadata.npz",
        labels=labels,
        semantics=semantics,
        train=fit,
        test_seen=seen_test,
        test_unseen=unseen_test,
    )
    fold = {
        "fit": np.r_[0:8, 20:28].tolist(),
        "cal": np.r_[8:12, 28:32, 40:48, 60:68].tolist(),
        "tune": np.r_[12:16, 32:36, 48:56, 68:76].tolist(),
        "pseudo_seen": [0, 1],
        "pseudo_unseen": [2, 3],
    }
    write_json(data / "folds/fold_0.json", fold)
    bundle = DatasetBundle.load(data)
    for role, indices in (("train", fit), ("test", bundle.test)):
        folder = output / ("cache_" + role)
        folder.mkdir()
        np.save(folder / "raw.npy", raw[indices])
        np.save(folder / "indices.npy", indices)
        write_prefix_cache_manifest(folder, bundle, role, {"id": "synthetic"})
    config = {
        "dataset": "synthetic",
        "seed": 7,
        "semantics_dim": 7,
        "visual_dim": 32,
        "backbone": "synthetic",
        "folds": [0],
        "source": {
            "routing": "latent",
            "prior_temperature": 0.5,
            "visual_norm": "l2",
            "ridge": 0.01,
            "mode_scale": 0.2,
            "modes": 2,
            "local_weight": 0.25,
            "prompt_lr": 0.001,
            "generator_lr": 0.001,
            "query_lr": 0.001,
            "moment_lr": 0.001,
            "weight_decay": 0.0001,
            "loss_temperature": 0.05,
            "anchor_weight": 0.1,
            "epochs": 1,
            "batch_size": 8,
            "micro_batch_size": 4,
            "score_batch_size": 4,
        },
        "adaptation": {
            "engine": "geometry_ot",
            "lr": 0.0001,
            "steps": 2,
            "assignment_temperature": 0.07,
            "consistency_temperature": 0.07,
            "partial_mass": 0.65,
            "view_js_temperature": 0.02,
            "view_margin": 0.0,
            "view_margin_temperature": 0.02,
            "ot_epsilon": 0.15,
            "ot_prior_shrink": 0.75,
            "ot_relaxation": 0.8,
            "ot_iterations": 10,
            "mode_temperature": 0.05,
            "min_mode_mass": 0.01,
            "class_alignment_weight": 1.0,
            "mode_alignment_weight": 1.0,
            "consistency_weight": 1.0,
            "ranking_weight": 0.5,
            "ranking_margin": 0.03,
            "hard_negative_neighbors": 2,
            "anchor_weight": 0.1,
            "relation_weight": 0.125,
            "gradient_clip": 5.0,
            "topology_weight": 0.25,
        },
        "adaptation_setting": {"lr_factor": 1.0, "local_target_mix": 1.0, "group_weight": 1.0},
        "adaptation_batch": 8,
    }
    develop(config, data, output / "cache_train", output / "development", SyntheticSuffix)
    train(
        config,
        data,
        output / "cache_train",
        output / "development",
        output / "model",
        SyntheticSuffix,
    )
    metrics = evaluate_run(
        output / "model", data, output / "cache_test", output / "evaluation", SyntheticSuffix
    )
    report = {
        "status": "PASS",
        "data": "synthetic only",
        "reported_benchmark_results": False,
        "checks": [
            "development",
            "fresh source fit",
            "calibration freeze",
            "restore",
            "TTA",
            "metrics",
        ],
        "metrics": metrics,
    }
    write_json(output / "smoke.json", report)
    return report
