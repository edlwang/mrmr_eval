"""Slim benchpred registry used by the mrmr_eval tutorial.

Exposes a small set of benchmark-prediction methods:

* mRMR with MID/MIQ schemes, ``_y`` relevance target, Ridge or polynomial
  Kernel Ridge regressor, for ``mi_k`` in 3..9.  Unified MI naming: binary
  scores use the Ross (2014) discrete-continuous estimator; continuous
  scores use the PCA-corrected KSG (LNC) estimator.  Both are exposed
  under the ``MI*`` family — there is no separate ``PMI*`` family.

* IRT baselines (PIRTPred / GPIRTPred and their k=1 / k=5 variants), with
  Beta variants for continuous data.

* Random sampling / search baselines plus their Kernel-Ridge counterparts.

* Anchor-points, Lasso, PCA, AIPW, Double-Optimize, MetaBench and the
  Sort-and-Search baselines from the paper.
"""

from .anchor_points import AnchorPointsWeightedPred, AnchorPointPredictorPred
from .double_optimize import DoubleOptimizePred
from .lasso import LassoPred
from .pca import PCAPred
from .random import (
    RandomSampling,
    RandomSamplingAndLearn,
    SampleFirstAndLearn,
    RandomSearchAndLearn,
    SmallSearchAndLearn,
    _make_krr_random_variant,
)
from .aipw import AIPWPred
from .tiny_bench import (
    PIRTPred,
    GPIRTPred,
    BetaPIRTPred,
    BetaGPIRTPred,
    PIRTPred1,
    PIRTPred5,
    GPIRTPred1,
    GPIRTPred5,
    BetaPIRTPred1,
    BetaPIRTPred5,
    BetaGPIRTPred1,
    BetaGPIRTPred5,
)
from .metabench import MetaBench
from .sort_search import SortAndSearchSum, SortAndSearchRecursiveSum

# MRMR family — only _y MIQ / MID with ridge and KRR.  k-NN MI variants
# (k=3 default; k=4..9 generated programmatically) are looked up via
# globals() because they are dynamically attached to the module by mrmr.py.
from . import mrmr as _mrmr_module
from .mrmr import (
    _make_krr_variant,
    MRMRPred,
    MRMRPred_MID_y,
    MRMRPred_MIQ_y,
    MRMRPred_MI_y,
)

all_methods = {
    # Random / search baselines
    "random_sampling": RandomSampling,
    "random_sampling_and_learn": RandomSamplingAndLearn,
    "sample_first_and_learn": SampleFirstAndLearn,
    "random_search_and_learn": RandomSearchAndLearn,
    "small_search_and_learn": SmallSearchAndLearn,
    # Other baselines
    "aipw": AIPWPred,
    "pca": PCAPred,
    "anchor_points_weighted": AnchorPointsWeightedPred,
    "anchor_points_predictor": AnchorPointPredictorPred,
    "double_optimize": DoubleOptimizePred,
    "lasso": LassoPred,
    "metabench": MetaBench,
    "sort_search_sum": SortAndSearchSum,
    "sort_search_recursive_sum": SortAndSearchRecursiveSum,
    # IRT methods (binary)
    "pirt": PIRTPred,
    "gpirt": GPIRTPred,
    "pirt1": PIRTPred1,
    "pirt5": PIRTPred5,
    "gpirt1": GPIRTPred1,
    "gpirt5": GPIRTPred5,
    # IRT methods (continuous)
    "B_pirt": BetaPIRTPred,
    "B_gpirt": BetaGPIRTPred,
    "B_pirt1": BetaPIRTPred1,
    "B_pirt5": BetaPIRTPred5,
    "B_gpirt1": BetaGPIRTPred1,
    "B_gpirt5": BetaGPIRTPred5,
    # mRMR — k=3 (default mi_k) variants
    "mrmr": MRMRPred,
    "mrmr_MID_y": MRMRPred_MID_y,
    "mrmr_MIQ_y": MRMRPred_MIQ_y,
    "mrmr_MI_y": MRMRPred_MI_y,
}

# Register KRR (degree-2) variants of random-sampling/search baselines.
# Keys are prefixed with "k": random_search_and_learn -> krandom_search_and_learn.
for _key in (
    "random_sampling_and_learn",
    "sample_first_and_learn",
    "random_search_and_learn",
    "small_search_and_learn",
):
    all_methods["k" + _key] = _make_krr_random_variant(all_methods[_key], degree=2)

# Register k=4..9 MI nearest-neighbour variants for every MIQ_y / MID_y / MI_y
# mRMR method.  The variant classes are generated dynamically in mrmr.py and
# attached to its module globals, so we look them up by name and register
# them under keys like mrmr5_MIQ_y, mrmr7_MID_y, etc.
for _key, _cls in list(all_methods.items()):
    if not _key.startswith("mrmr"):
        continue
    for _k in (4, 5, 6, 7, 8, 9):
        _variant_key = _key.replace("mrmr", f"mrmr{_k}", 1)
        _variant_class_name = _cls.__name__.replace("MRMR", f"MRMR{_k}", 1)
        all_methods[_variant_key] = getattr(_mrmr_module, _variant_class_name)

# Register kernel-ridge (degree-2) variants for every mRMR method.  Keys are
# prefixed with "k": mrmr_MIQ_y -> kmrmr_MIQ_y, mrmr5_MID_y -> kmrmr5_MID_y.
for _key, _cls in list(all_methods.items()):
    if not _key.startswith("mrmr"):
        continue
    all_methods["k" + _key] = _make_krr_variant(_cls, degree=2)

# ---------------------------------------------------------------------------
# Refit / backfill methods: reuse coresets from IRT / anchor-points methods and
# refit a Ridge or Kernel Ridge regressor (same pipeline as mrmr / kmrmr).
# Key naming: base_key+ / kbase_key+ / k3base_key+ / k4base_key+
# ---------------------------------------------------------------------------
from .backfill import _make_backfill_ridge, _make_backfill_krr

_backfill_base_methods = [
    "pirt", "gpirt",
    "pirt1", "gpirt1", "pirt5", "gpirt5",
    "B_pirt", "B_gpirt",
    "B_pirt1", "B_gpirt1", "B_pirt5", "B_gpirt5",
    "anchor_points_weighted", "anchor_points_predictor",
    "lasso",
]
for _base_key in _backfill_base_methods:
    all_methods[f"{_base_key}+"] = _make_backfill_ridge(_base_key)
    all_methods[f"k{_base_key}+"] = _make_backfill_krr(_base_key, degree=2)
    all_methods[f"k3{_base_key}+"] = _make_backfill_krr(_base_key, degree=3)
    all_methods[f"k4{_base_key}+"] = _make_backfill_krr(_base_key, degree=4)

# ---------------------------------------------------------------------------
# DKPS (Data Kernel Perspective Space): response-based, transductive predictor.
# Unlike every method above, it consumes raw model responses (HELM only) rather
# than the score matrix; the runner detects this via ``requires_model_outputs``.
# The import pulls the vendored DKPS code + graspologic (optional [dkps] extra),
# so it is guarded: a missing graspologic disables DKPS without breaking the
# rest of the registry.
# ---------------------------------------------------------------------------
try:
    from .dkps import (
        DKPSPred,
        DKPSGooglePred,
        DKPSMRMRPred,
        _make_dkps_backfill,
        _make_dkps_mrmr_variant,
    )

    all_methods["dkps"] = DKPSPred              # one-hot embedder (categorical tasks)
    all_methods["dkps_google"] = DKPSGooglePred  # Gemini embedder (free-text tasks)

    # Iterative, per-target DKPS-guided mRMR coreset selection (transductive,
    # response-based, HELM-only).  Needs the runner's ``requires_full_target_outputs``
    # branch.  ``k`` is the number of nearest source models restricting each
    # inner mRMR pick; default scheme is MID.
    all_methods["dkps_mrmr"] = DKPSMRMRPred                      # onehot, k=10, MID
    all_methods["dkps_mrmr_google"] = _make_dkps_mrmr_variant(embedder="google")
    for _k in (5, 15, 20):
        all_methods[f"dkps_mrmr_k{_k}"] = _make_dkps_mrmr_variant(k=_k)
    all_methods["dkps_mrmr_MIQ"] = _make_dkps_mrmr_variant(miq_scheme=True)
    all_methods["dkps_mrmr_MI"] = _make_dkps_mrmr_variant(only_relevance=True)

    # DKPS evaluated on coresets borrowed from other methods (backfill "+"
    # style, e.g. anchor_points_weighted+).  Key naming: dkps+<base_key>.  Each
    # requires the base method to have been run first so its coreset checkpoint
    # exists (same dataset / seed / coreset_size / nmodels); otherwise the
    # runner skips that trial.  Add a `dkps_google+<base>` for free-text tasks
    # via `_make_dkps_backfill(base, embedder="google")`.
    _dkps_backfill_base_methods = [
        "anchor_points_weighted",
        "mrmr5_MIQ_y",
        "kmrmr5_MIQ_y",
        "gpirt1",
        "lasso",
    ]
    for _dkps_base in _dkps_backfill_base_methods:
        if _dkps_base in all_methods:
            all_methods[f"dkps+{_dkps_base}"] = _make_dkps_backfill(_dkps_base)
except ImportError as _dkps_err:  # graspologic (or google-genai) not installed
    import warnings as _warnings

    _warnings.warn(
        f"DKPS methods not registered: {_dkps_err}. "
        "Install the optional extra with `pip install -e .[dkps]` to enable "
        "'dkps' / 'dkps_google'."
    )
