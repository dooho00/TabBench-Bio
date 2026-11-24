"""AutoGluon adapter for the released Lattice checkpoints."""

import numpy as np
import torch
from autogluon.core.models import AbstractModel
from huggingface_hub import hf_hub_download
from lattice import LatticeClassifier, LatticeRegressor


class LatticeModel(AbstractModel):
    ag_key = "LATTICE"
    ag_name = "Lattice"
    model_size = "default"
    checkpoint_repo = "hachure-ai/Lattice"
    checkpoint_revision = "38d704ff3495ed36d97a473f9f321a791f7d9eea"  # v1.0.1

    def _fit(self, X, y, num_cpus=1, num_gpus=0, **kwargs):
        assert num_cpus >= 1
        if num_gpus and not torch.cuda.is_available():
            raise RuntimeError(f"{self.ag_key} requested a GPU, but CUDA is unavailable.")
        torch.set_num_threads(num_cpus)
        regression = self.problem_type == "regression"
        estimator = LatticeRegressor if regression else LatticeClassifier
        checkpoint = hf_hub_download(
            repo_id=self.checkpoint_repo,
            filename="reg.ckpt" if regression else "cls.ckpt",
            revision=self.checkpoint_revision,
        )
        self.model = estimator(
            model_size=self.model_size,
            model_path=checkpoint,
            device="cuda:0" if num_gpus else "cpu",
            random_state=42,
        )
        self.model.fit(self.preprocess(X), y)
        if not regression:
            assert np.array_equal(self.model.classes_, np.arange(self.num_classes)), (
                f"{self.ag_key} probabilities must follow AutoGluon's class order."
            )

    def _predict_proba(self, X, **kwargs):
        regression = self.problem_type == "regression"
        predict = self.model.predict if regression else self.model.predict_proba
        prediction = np.asarray(predict(self.preprocess(X)))
        expected = (len(X),) if regression else (len(X), self.num_classes)
        assert prediction.shape == expected, f"Unexpected {self.ag_key} prediction shape."
        assert np.isfinite(prediction).all(), f"{self.ag_key} returned non-finite predictions."
        return self._convert_proba_to_unified_form(prediction)

    def _get_default_resources(self):
        return torch.get_num_threads(), int(torch.cuda.is_available())

    def _more_tags(self):
        return {"can_refit_full": True}

    @classmethod
    def supported_problem_types(cls):
        return ["binary", "multiclass", "regression"]
