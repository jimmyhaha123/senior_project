"""
XGBoost classification objective for MIVABO using a REAL dataset (Covertype subset ~10,000 samples), with local caching.

- Dataset: UCI Covertype (Forest Cover Type) via sklearn.datasets.fetch_covtype (multi-class, 54 features).
    We derive a binary classification target by thresholding: y_bin = 1 if original class == 2 else 0.
    Then we draw a stratified subset of 10,000 samples (fixed random_state) for manageable per-evaluation training.
- On first use, creates a stratified train/valid split from that subset and saves it to an .npz file.
- On subsequent uses, loads the split directly from disk.

Mixed-variable setup:
---------------------
Discrete part (Dd = 11 bits):
    * 1 bit  -> booster ∈ {gbtree, dart}
    * 4 bits -> max_depth ∈ {2, 3, 4, 5, 6}         (ordinal via bit-count)
    * 3 bits -> n_estimators ∈ {50, 100, 200, 400}  (ordinal via bit-count)
    * 3 bits -> min_child_weight ∈ {1, 3, 5, 7}     (ordinal via bit-count)

Continuous part (Dc = 5), each in [0,1], decoded as:
    * x0 -> learning_rate     in [0.01, 0.30] with x0 first remapped u0 := 0.2 + 0.8*x0 (cuts out [0,0.2))
    * x1 -> subsample         in [0.50, 1.00]
    * x2 -> colsample_bytree  in [0.50, 1.00]
    * x3 -> log10(lambda)     in [-3, 1]  => reg_lambda in [1e-3, 10]
    * x4 -> log10(alpha)      in [-3, 1]  => reg_alpha  in [1e-3, 10]

Objective: validation log-loss (cross-entropy), to be MINIMIZED.

API:
    from xgb_breast_cancer_objective import make_xgb_breast_cancer_problem
    space, f = make_xgb_breast_cancer_problem()

Then f(xd_bits, xc_unit) returns a float loss.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch as _t

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

from xgboost import XGBClassifier

from MIVABO import MixedSpace, DiscreteSpec, ContinuousSpec
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*use_label_encoder.*are not used.*",
    category=UserWarning,
)


def _load_or_create_breast_cancer_split(
    cache_path: Path,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load (X_train, X_valid, y_train, y_valid) from cache_path if it exists,
    otherwise create the split from sklearn's built-in dataset and save it.
    """
    if cache_path.is_file():
        data = np.load(cache_path)
        X_train = data["X_train"]
        X_valid = data["X_valid"]
        y_train = data["y_train"]
        y_valid = data["y_valid"]
        return X_train, X_valid, y_train, y_valid

    # Otherwise, fetch Covertype (REAL dataset) and build a 10,000-sample binary subset
    ds = fetch_covtype()
    X_full = ds.data
    y_full = ds.target
    # Derive binary target (class 2 vs others) for a well-populated class
    y_bin = (y_full == 2).astype(np.int64)
    # Stratified sample of ~10,000 rows for speed vs representativeness
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=10000, random_state=random_state)
    # test portion of shuffle split used as our working subset; train portion discarded
    for _train_idx, subset_idx in sss.split(X_full, y_bin):
        X = X_full[subset_idx]
        y = y_bin[subset_idx]
        break

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train = np.asarray(X_train, dtype=np.float32)
    X_valid = np.asarray(X_valid, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_valid = np.asarray(y_valid, dtype=np.int64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
    )

    return X_train, X_valid, y_train, y_valid


def make_xgb_breast_cancer_problem(
    test_size: float = 0.2,
    random_state: int = 0,
    cache_file: str | os.PathLike = "breast_cancer_xgb_split.npz",
    metric: str = "misclassification",
):
    """
    Factory for a mixed discrete/continuous black-box objective based on
    training XGBoost on the Breast Cancer dataset, with local data caching.

    Parameters
    ----------
    test_size : float, optional
        Fraction of data used for validation split. Default: 0.2.
    random_state : int, optional
        Random seed used for the train/valid split and model. Default: 0.
    cache_file : str or Path, optional
        Path to .npz file used to cache the pre-split dataset. Default:
        'breast_cancer_xgb_split.npz' in the current working directory.

    Returns
    -------
    space : MixedSpace
        Mixed search space with Dd=11 discrete bits, Dc=5 continuous dims.
    f : callable
        f(xd_bits, xc_unit) -> float
        xd_bits: 1D tensor of 0/1 of length 11
        xc_unit: 1D tensor in [0,1]^5
    """
    # normalize metric selection
    metric_key = str(metric).strip().lower()
    if metric_key in {"error", "miscls", "misclassification", "acc", "accuracy"}:
        metric_key = "misclassification"
    elif metric_key in {"logloss", "log_loss", "xent", "cross_entropy"}:
        metric_key = "logloss"
    else:
        raise ValueError(
            "metric must be one of {'misclassification','logloss'} (case-insensitive, common aliases allowed)"
        )
    cache_path = Path(cache_file)

    # ------------------------------------------------------------------
    # 1. Load or create dataset split ONCE (no per-eval I/O).
    # ------------------------------------------------------------------
    X_train, X_valid, y_train, y_valid = _load_or_create_breast_cancer_split(
        cache_path=cache_path,
        test_size=test_size,
        random_state=random_state,
    )

    # ------------------------------------------------------------------
    # 2. Define MixedSpace: 11 discrete bits, 5 continuous dims.
    # ------------------------------------------------------------------
    Dd = 11  # 1 + 4 + 3 + 3
    Dc = 5

    space = MixedSpace(
        DiscreteSpec(Dd=Dd),
        ContinuousSpec(Dc=Dc),
        lin_cons=None,
    )

    # Levels for ordinal discrete hyperparameters
    max_depth_levels = [2, 3, 4, 5, 6]        # K=5 -> 4 bits
    n_estimators_levels = [50, 100, 200, 400] # K=4 -> 3 bits
    min_child_weight_levels = [1, 3, 5, 7]    # K=4 -> 3 bits

    def _decode_ordinal_group(bits_1d: _t.Tensor, levels: list[int]) -> int:
        """
        Map a 0/1 bit pattern (length B) to one of K=len(levels) ordinal levels.

        We:
            - round and clamp bits to {0,1}
            - count ones: count = sum(bits)
            - idx = min(count, K-1)
            - return levels[idx]

        This ensures:
            - Hamming distance between encodings is monotone in |count1 - count2|
            - Flipping one bit changes the index by at most 1
        """
        bits_clean = bits_1d.round().clamp(0.0, 1.0)
        count_ones = int(bits_clean.sum().item())
        K = len(levels)
        idx = max(0, min(count_ones, K - 1))
        return levels[idx]

    def _decode_continuous(xc_unit_1d: _t.Tensor):
        """
        Map xc in [0,1]^5 to actual XGBoost continuous hyperparameters.
        Note: the first continuous variable x0 is remapped to u0 := 0.2 + 0.8*x0
        before use, so the effective domain excludes [0.0, 0.2).
        """
        xc = xc_unit_1d.detach().cpu().double().view(-1)

        def _lin(lo: float, hi: float, u: float) -> float:
            return float(lo + (hi - lo) * u)

        # learning_rate in [0.01, 0.30]; remap x0 → u0 ∈ [0.2, 1.0]
        u0 = float(0.2 + 0.8 * float(xc[0].item()))
        if u0 < 0.2:
            u0 = 0.2
        if u0 > 1.0:
            u0 = 1.0
        learning_rate = _lin(0.01, 0.30, u0)

        # subsample, colsample_bytree in [0.5, 1.0]
        subsample = _lin(0.5, 1.0, xc[1])
        colsample_bytree = _lin(0.5, 1.0, xc[2])

        # log10(lambda), log10(alpha) in [-3, 1] => [1e-3, 10]
        log10_lambda = _lin(-3.0, 1.0, xc[3])
        log10_alpha = _lin(-3.0, 1.0, xc[4])
        reg_lambda = 10.0 ** log10_lambda
        reg_alpha = 10.0 ** log10_alpha

        return learning_rate, subsample, colsample_bytree, reg_lambda, reg_alpha

    # ------------------------------------------------------------------
    # 3. Objective: train XGBoost model and compute validation log-loss.
    # ------------------------------------------------------------------
    def f(xd_bits: _t.Tensor, xc_unit: _t.Tensor) -> float:
        """
        Mixed-variable black-box objective:

            - xd_bits: tensor of shape (11,), values ~{0,1}
            - xc_unit: tensor of shape (5,), values in [0,1]

        Returns
        -------
        float
            Validation log-loss of XGBClassifier, to be minimized.
        """
        # Validate shapes
        if xd_bits.numel() != Dd:
            raise ValueError(f"xd_bits must have {Dd} elements, got {xd_bits.numel()}")
        if xc_unit.numel() != Dc:
            raise ValueError(f"xc_unit must have {Dc} elements, got {xc_unit.numel()}")

        # Work on CPU doubles for decoding
        xd = xd_bits.detach().cpu().double().view(-1)

        # --- Decode discrete part ---

        # booster: 1 bit -> {gbtree, dart}
        booster_bit = int(xd[0].round().clamp(0.0, 1.0).item())
        booster = "gbtree" if booster_bit == 0 else "dart"

        # max_depth: 4 bits -> {2, 3, 4, 5, 6}
        max_depth_bits = xd[1:5]   # length 4
        max_depth = _decode_ordinal_group(max_depth_bits, max_depth_levels)

        # n_estimators: 3 bits -> {50, 100, 200, 400}
        n_estimators_bits = xd[5:8]  # length 3
        n_estimators = _decode_ordinal_group(n_estimators_bits, n_estimators_levels)

        # min_child_weight: 3 bits -> {1, 3, 5, 7}
        min_child_weight_bits = xd[8:11]  # length 3
        min_child_weight = _decode_ordinal_group(min_child_weight_bits, min_child_weight_levels)

        # --- Decode continuous part ---
        learning_rate, subsample, colsample_bytree, reg_lambda, reg_alpha = _decode_continuous(xc_unit)

        # --- Train XGBoost model (small dataset, cheap eval) ---

        # Choose eval metric based on requested return metric. For early stopping to track
        # the same quantity we return, we set eval_metric accordingly.
        eval_metric = "error" if metric_key == "misclassification" else "logloss"

        model = XGBClassifier(
            booster=booster,
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            min_child_weight=min_child_weight,
            objective="binary:logistic",
            eval_metric=eval_metric,
            tree_method="hist",   # fast on CPU
            n_jobs=4,
            random_state=random_state,
            use_label_encoder=False,
            # pass early stopping rounds in constructor to avoid deprecation warning
            early_stopping_rounds=20,
        )

        # Early stopping to avoid wasteful training
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        # Compute requested validation metric
        if metric_key == "logloss":
            prob_valid = model.predict_proba(X_valid)
            loss = log_loss(y_valid, prob_valid)
            return float(loss)
        else:  # misclassification rate
            y_pred = model.predict(X_valid)
            acc = accuracy_score(y_valid, y_pred)
            err = 1.0 - float(acc)
            return err

    return space, f
