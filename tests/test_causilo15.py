"""Exercise the Causilo 1.5 backend and full refit with local CPU test weights."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("autogluon.tabular")
pytest.importorskip("causilo.v1_5.nn.model")

import torch
from causilo.model_selection import RUNTIME15_FORMAT
from causilo.v1_5 import checkpoints
from causilo.v1_5.nn.model import Model, ModelConfig
from safetensors.torch import save_file

from tabbench_bio.dataset import TaskType
from tabbench_bio.model import AutoGluonModel
from tabbench_bio.models.causilo import Causilo15Model


@pytest.mark.parametrize(
    ("task_type", "task", "max_classes", "prediction_shape"),
    [
        (TaskType.Classification, "classification", 10, (4, 2)),
        (TaskType.Regression, "regression", 0, (4,)),
    ],
)
def test_refits_on_all_training_rows(
    tmp_path, monkeypatch, task_type, task, max_classes, prediction_shape
):
    config = ModelConfig(max_classes=max_classes, width=16, column_depth=1, prediction_depth=1)
    config_bytes = json.dumps(
        {
            "architecture": RUNTIME15_FORMAT,
            "format_version": 1,
            "config_version": 1,
            "task": task,
            "model": config.to_dict(),
        }
    ).encode("utf-8")
    (tmp_path / "config.json").write_bytes(config_bytes)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        weights = Model(config).state_dict()
    save_file(
        weights,
        tmp_path / "model.safetensors",
        metadata={
            "architecture": RUNTIME15_FORMAT,
            "format_version": "1",
            "task": task,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
    )
    monkeypatch.setattr(checkpoints, "download_checkpoint", lambda source: tmp_path)
    monkeypatch.setenv("TABBENCH_MODEL_CPUS", "1")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    fit_rows = []
    original_fit = Causilo15Model._fit

    def record_fit(self, X, y, **kwargs):
        fit_rows.append(set(X.index))
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(Causilo15Model, "_fit", record_fit)

    rng = np.random.default_rng(42)
    features = pd.DataFrame(rng.normal(size=(104, 3)), columns=list("abc"))
    train = features.iloc[:100].copy()
    train["target"] = np.arange(100) % 2 if max_classes else rng.normal(size=100)
    test = features.iloc[100:]
    model = AutoGluonModel(
        ["CAUSILO-1.5"],
        ensemble=False,
        optimize=False,
        task_type=task_type,
        autogluon_presets="medium_quality",
        autogluon_path=str(tmp_path / "autogluon"),
    )
    predictor = model.fit(train)
    holdout_fit, full_fit = fit_rows
    assert holdout_fit < full_fit
    assert full_fit == set(train.index)
    assert predictor.model_best == predictor.model_refit_map()["Causilo15"]

    predictions = model.predict_proba(test) if max_classes else model.predict(test)
    assert predictions.shape == prediction_shape
    assert np.isfinite(np.asarray(predictions)).all()
