"""DKPS (Data Kernel Perspective Space) benchmark-prediction method.

Adapts the transductive DKPS predictor to the ``BenchPred`` interface used by
``method_runner``.  Unlike the score-based methods in this repo, DKPS consumes
raw model **responses**: the runner detects the ``requires_model_outputs`` flag
and hands ``fit``/``predict`` slices of ``model_outputs`` (HELM only) instead of
the binary score matrix.

Pipeline (mirrors dkps ``examples/helm/runners/dkps.py``):

  fit(source responses, scores)  -> pick a random coreset of queries; stash the
                                    source responses on those queries + the
                                    per-source mean score (regression target).
  predict(target responses)      -> embed source + target responses *together*
                                    (transductive), build the DKPS space, fit a
                                    LinearRegression on source coords, predict
                                    the target coords, clip to [0, 1].

Embedders:
  * ``"onehot"`` (default, dependency-free): one-hot over the union of observed
    responses.  Suitable for categorical / multiple-choice tasks (med_qa,
    legalbench).
  * ``"google"`` (``DKPSGooglePred``): Gemini text embeddings via the vendored
    ``embed_api``.  Needs ``GEMINI_API_KEY`` and the ``[dkps]`` extra
    (``google-genai``).  Use for free-text tasks (math, wmt_14).

Both embedders run inside ``predict`` over source ∪ target so every model's
embedding has the same dimension (DKPS asserts identical shapes).
"""

import numpy as np
import joblib as jbl
from scipy.special import psi
from sklearn.linear_model import LinearRegression

from .base import BenchPred, set_random_seed
from .mrmr import MRMRPred
from ._vendor.dkps_cmds import DataKernelPerspectiveSpace


def _as_text(x):
    """Coerce a raw response cell to a string ('' for missing/NaN)."""
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


class DKPSPred(BenchPred):
    """DKPS predictor with one-hot response embedding (dependency-free)."""

    # Tell the runner to pass raw model responses (not scores) to fit/predict.
    requires_model_outputs = True

    embedder = "onehot"          # "onehot" | "google"
    embed_model = None           # provider-specific model id (None -> default)
    n_components_cmds = 8

    def __init__(self):
        super().__init__()
        self.compressed_data_indices = None
        self._source_resp = None     # (M_source, k) raw responses on the coreset
        self._source_scores = None   # (M_source,) regression targets
        self.selection_metrics = None

    # -- BenchPred interface ------------------------------------------------

    def fit(self, source_full_scores, coreset_size, seed=42,
            source_model_outputs=None, **kwargs):
        if source_model_outputs is None:
            raise ValueError(
                "DKPSPred.fit requires source_model_outputs (raw responses). "
                "DKPS-family methods only run with data_source='helm'."
            )
        set_random_seed(seed)

        source_full_scores = np.asarray(source_full_scores)
        num_models, num_data = source_full_scores.shape
        k = max(1, min(int(coreset_size), num_data))

        self.compressed_data_indices = np.sort(
            np.random.choice(num_data, size=k, replace=False)
        )
        self._source_resp = np.asarray(source_model_outputs, dtype=object)[
            :, self.compressed_data_indices
        ]
        # Regression target: each source model's mean accuracy over the full set
        # (== true_acc for that model), matching the score-based methods.
        self._source_scores = source_full_scores.mean(axis=1)
        return self

    def get_coreset(self):
        return self.compressed_data_indices

    def predict(self, target_coreset_outputs):
        target = np.asarray(target_coreset_outputs, dtype=object)
        if target.ndim == 1:
            target = target.reshape(1, -1)

        n_src = self._source_resp.shape[0]
        # Embed source + target together so every model shares one embedding dim
        # (DKPS is transductive and asserts identical shapes across models).
        all_resp = np.concatenate([self._source_resp, target], axis=0)
        emb = self._embed(all_resp)                       # (M_all, k, dim)

        data = {i: emb[i][:, None, :] for i in range(emb.shape[0])}  # (k, 1, dim)
        n_comp = int(max(1, min(self.n_components_cmds, n_src - 1)))
        coords = DataKernelPerspectiveSpace(
            n_components_cmds=n_comp
        ).fit_transform(data, return_dict=True)

        x_src = np.vstack([coords[i] for i in range(n_src)])
        x_tgt = np.vstack([coords[i] for i in range(n_src, emb.shape[0])])

        regressor = LinearRegression().fit(x_src, self._source_scores)
        return np.clip(regressor.predict(x_tgt), 0.0, 1.0)

    def save(self, path_save):
        jbl.dump(
            {
                "embedder": self.embedder,
                "embed_model": self.embed_model,
                "n_components_cmds": self.n_components_cmds,
                "compressed_data_indices": self.compressed_data_indices,
                "source_resp": self._source_resp,
                "source_scores": self._source_scores,
            },
            path_save,
        )

    def load(self, path_load):
        state = jbl.load(path_load)
        self.embedder = state["embedder"]
        self.embed_model = state["embed_model"]
        self.n_components_cmds = state["n_components_cmds"]
        self.compressed_data_indices = state["compressed_data_indices"]
        self._source_resp = state["source_resp"]
        self._source_scores = state["source_scores"]
        return self

    # -- embedding ----------------------------------------------------------

    def _embed(self, responses):
        """Embed an (M, k) array of raw responses -> (M, k, dim) float array."""
        if self.embedder == "onehot":
            return self._embed_onehot(responses)
        if self.embedder == "google":
            return self._embed_google(responses)
        raise ValueError(f"Unknown embedder {self.embedder!r}")

    @staticmethod
    def _embed_onehot(responses):
        """One-hot over the union of observed labels (out-of-vocab -> all-zero)."""
        m, k = responses.shape
        flat = [_as_text(x) for x in responses.reshape(-1)]
        vocab = sorted({s for s in flat if s != ""})
        index = {label: i for i, label in enumerate(vocab)}
        dim = max(len(vocab), 1)

        out = np.zeros((m * k, dim), dtype=np.float64)
        for i, s in enumerate(flat):
            j = index.get(s)
            if j is not None:
                out[i, j] = 1.0
        return out.reshape(m, k, dim)

    def _embed_google(self, responses):
        """Gemini text embeddings (de-duplicated to minimise API calls)."""
        from ._vendor.dkps_embed import embed_api  # lazy: needs google-genai

        m, k = responses.shape
        flat = [_as_text(x) for x in responses.reshape(-1)]
        unique = list(dict.fromkeys(flat))            # preserve order, drop dups
        vectors = np.asarray(
            embed_api(provider="google", input_strs=unique, model=self.embed_model)
        )
        lookup = {s: vectors[i] for i, s in enumerate(unique)}
        dim = vectors.shape[1]
        out = np.array([lookup[s] for s in flat], dtype=np.float64)
        return out.reshape(m, k, dim)


class DKPSGooglePred(DKPSPred):
    """DKPS predictor using Gemini text embeddings (free-text tasks)."""

    embedder = "google"


class DKPSBackfillPred(DKPSPred):
    """DKPS evaluated on a coreset selected by *another* method.

    Mirrors the backfill ("+") pattern in ``benchpred/backfill.py``: the runner
    detects ``_base_method_key``, locates that base method's saved coreset
    (``ckpt.jbl`` from a prior run at the same dataset / seed / coreset_size /
    nmodels) and passes its path as ``base_ckpt_path``.  This subclass loads the
    base coreset instead of drawing a random one; embedding, MDS and regression
    are all inherited from :class:`DKPSPred`.

    Because it also inherits ``requires_model_outputs = True``, the runner hands
    it *both* ``source_model_outputs`` and ``base_ckpt_path`` (the two fit-kwarg
    mechanisms compose with no runner changes).

    Requires the base method to have been run first; otherwise the runner finds
    no checkpoint and skips the trial (returns ``None``), exactly like
    ``anchor_points_weighted+`` / ``lasso+``.
    """

    _base_method_key = None  # set by _make_dkps_backfill

    def fit(self, source_full_scores, coreset_size, seed=42,
            source_model_outputs=None, base_ckpt_path=None, **kwargs):
        if source_model_outputs is None:
            raise ValueError(
                "DKPSBackfillPred.fit requires source_model_outputs (raw "
                "responses). DKPS-family methods only run with data_source='helm'."
            )
        if base_ckpt_path is None:
            raise ValueError(
                f"{type(self).__name__} requires base_ckpt_path; run the base "
                f"method {self._base_method_key!r} first so its coreset exists."
            )
        # Coreset comes from the base method's checkpoint -- no random sampling.
        from . import all_methods

        base_method = all_methods[self._base_method_key]()
        base_method.load(base_ckpt_path)
        self.compressed_data_indices = np.asarray(base_method.get_coreset())

        self._source_resp = np.asarray(source_model_outputs, dtype=object)[
            :, self.compressed_data_indices
        ]
        self._source_scores = np.asarray(source_full_scores).mean(axis=1)
        return self


def _make_dkps_backfill(base_method_key, embedder="onehot"):
    """Build a :class:`DKPSBackfillPred` subclass bound to *base_method_key*.

    ``embedder`` selects the response embedder ("onehot" or "google"), exactly
    as for the standalone DKPS methods.
    """

    class _Variant(DKPSBackfillPred):
        _base_method_key = base_method_key

    _Variant.embedder = embedder
    tag = "" if embedder == "onehot" else f"_{embedder}"
    name = f"DKPSBackfill{tag}_{base_method_key}"
    _Variant.__name__ = name
    _Variant.__qualname__ = name
    _Variant.__module__ = __name__
    _Variant.__doc__ = (
        f"DKPS ({embedder} embedder) on the coreset selected by "
        f"'{base_method_key}'."
    )
    return _Variant


class DKPSMRMRPred(MRMRPred, DKPSPred):
    """Iterative, per-target DKPS-guided mRMR coreset selection.

    A transductive, response-based sibling of :class:`DKPSPred`.  Instead of a
    random (or borrowed) coreset, the coreset is grown one item at a time and is
    **tailored to each target model**:

      1. Seed with a single global mRMR pick (highest relevance to the mean
         source score; shared across targets).
      2. While the coreset is smaller than ``coreset_size``:
         a. DKPS-embed source ∪ {target} from their *raw responses* on the
            current coreset.
         b. Take the ``k`` source models nearest the target in that space.
         c. Restrict the source score matrix to those ``k`` models and pick the
            next item by one greedy mRMR step (relevance − redundancy for MID,
            relevance / redundancy for MIQ).
      3. Predict via the standard DKPS transductive regression on the final
         per-target coreset: embed source ∪ {target}, fit ``LinearRegression``
         from *all* source coordinates to source mean scores, read off the
         target's prediction.

    ``k`` therefore gates only the inner mRMR selection; the final regression
    still uses every source model as an anchor.

    Because the coreset differs per target, the runner cannot slice a single
    coreset of columns: this class sets ``requires_full_target_outputs`` so the
    runner hands ``predict`` the *full* target response matrix.  The per-target
    coresets used for the test models are exposed on ``target_coresets_`` after
    ``predict`` for result logging.

    HELM-only (needs raw responses + the ``[dkps]`` extra), like all DKPS-family
    methods.
    """

    # Capability flags read by method_runner.
    requires_model_outputs = True
    requires_full_target_outputs = True

    # Class-level defaults; variant subclasses override these.
    embedder = "onehot"          # "onehot" | "google"
    embed_model = None
    n_components_cmds = 8
    k = 10                       # nearest source models for the inner mRMR step
    mi_k = 3                     # MI neighbours (Ross / LNC estimators)
    miq_scheme = False           # False -> MID, True -> MIQ
    only_relevance = False       # True -> relevance-only inner picks

    def __init__(self):
        super().__init__()
        # Honour class-level overrides (MRMRPred.__init__ hard-codes some of
        # these on the instance, so re-apply them from the concrete class).
        cls = type(self)
        self.k = cls.k
        self.mi_k = cls.mi_k
        self.miq_scheme = cls.miq_scheme
        self.only_relevance = cls.only_relevance

        self._source_resp_full = None    # (n_src, N) raw responses, all items
        self._source_scores_full = None  # (n_src, N) scores, all items
        self._source_mean = None         # (n_src,) regression / relevance target
        self._item1 = None               # globally selected seed item
        self.coreset_size = None
        self.target_coresets_ = None     # per-target coresets from last predict

    # -- BenchPred interface ------------------------------------------------

    def fit(self, source_full_scores, coreset_size, seed=42,
            source_model_outputs=None, **kwargs):
        if source_model_outputs is None:
            raise ValueError(
                "DKPSMRMRPred.fit requires source_model_outputs (raw responses). "
                "DKPS-family methods only run with data_source='helm'."
            )
        set_random_seed(seed)

        scores = np.asarray(source_full_scores, dtype=np.float64)
        num_models, num_data = scores.shape

        self._binary = self._is_binary_scores(scores)
        # Impute so the MI estimators see no NaNs (mirrors MRMRPred.fit).
        if np.any(~np.isfinite(scores)):
            if self._binary:
                scores = np.nan_to_num(scores, nan=0.0)
            else:
                col_means = np.nanmean(scores, axis=0)
                inds = np.where(np.isnan(scores))
                scores[inds] = col_means[inds[1]]

        self._source_scores_full = scores
        self._source_resp_full = np.asarray(source_model_outputs, dtype=object)
        self._source_mean = scores.mean(axis=1)
        self.coreset_size = max(1, min(int(coreset_size), num_data - 1))

        # Seed item: global highest-relevance pick over ALL source models.
        mi_rel, _ = self._get_mi_estimators(self._binary)
        relevance = np.array([
            mi_rel(scores[:, c], self._source_mean, k=self.mi_k)
            for c in range(num_data)
        ])
        self._item1 = int(np.argmax(relevance))
        return self

    def get_coreset(self):
        # Per-target coresets are exposed via ``target_coresets_``; this returns
        # the shared seed item purely for interface compatibility.
        return np.array([self._item1]) if self._item1 is not None else None

    def predict(self, target_coreset_outputs):
        target = np.asarray(target_coreset_outputs, dtype=object)
        if target.ndim == 1:
            target = target.reshape(1, -1)

        preds = np.empty(target.shape[0], dtype=np.float64)
        coresets = []
        for i in range(target.shape[0]):
            coreset, pred = self._predict_one(target[i])
            preds[i] = pred
            coresets.append(coreset)
        self.target_coresets_ = coresets
        return preds

    # -- core per-target routine -------------------------------------------

    def _predict_one(self, target_resp_row):
        """Build a target-specific coreset, then DKPS-regress its accuracy."""
        n_src = self._source_resp_full.shape[0]
        coreset = [self._item1]

        while len(coreset) < self.coreset_size:
            coords = self._embed_models(coreset, target_resp_row)
            src_coords, tgt_coord = coords[:n_src], coords[n_src]
            dist = np.linalg.norm(src_coords - tgt_coord, axis=1)
            kk = int(max(2, min(self.k, n_src)))
            knn = np.argsort(dist)[:kk]
            coreset.append(self._mrmr_next(coreset, knn))

        # Final transductive regression on the per-target coreset.
        coords = self._embed_models(coreset, target_resp_row)
        src_coords = coords[:n_src]
        tgt_coord = coords[n_src].reshape(1, -1)
        reg = LinearRegression().fit(src_coords, self._source_mean)
        pred = float(np.clip(reg.predict(tgt_coord), 0.0, 1.0)[0])
        return coreset, pred

    def _embed_models(self, cols, target_resp_row):
        """DKPS-embed source ∪ {target} from responses on ``cols``.

        Returns coords of shape ``(n_src + 1, n_comp)``; row ``n_src`` is the
        target.
        """
        cols = np.asarray(cols)
        src = self._source_resp_full[:, cols]
        tgt = np.asarray(target_resp_row, dtype=object)[cols].reshape(1, -1)
        all_resp = np.concatenate([src, tgt], axis=0)          # (n_src+1, c)

        emb = self._embed(all_resp)                            # (M, c, dim)
        data = {i: emb[i][:, None, :] for i in range(emb.shape[0])}  # (c, 1, dim)
        n_comp = int(max(1, min(self.n_components_cmds, src.shape[0] - 1)))
        coords = DataKernelPerspectiveSpace(
            n_components_cmds=n_comp
        ).fit_transform(data, return_dict=True)
        return np.vstack([coords[i] for i in range(emb.shape[0])])

    @staticmethod
    def _ross_relevance_batch(X, y, k):
        """Vectorised Ross (2014) MI between many binary columns and one
        continuous target ``y``.

        Equivalent to calling
        ``MRMRPred._mutual_information_ross_estimator(X[:, c], y, k)`` for every
        column ``c``, but precomputes the fixed 1-D distance matrix of ``y`` and
        replaces the ~3 cKDTree builds + Python neighbour loop *per candidate*
        with array ops over an ``(n, n)`` matrix.  This is the inner-loop hotspot
        of :meth:`_mrmr_next` (≈96% of predict time before this change).

        Args:
            X: ``(n, C)`` binary (0/1) matrix — one candidate per column.
            y: ``(n,)`` continuous target (shared across candidates).
            k: number of nearest neighbours.

        Returns:
            ``(C,)`` MI estimates (non-negative); 0 for columns whose 0/1 class
            is smaller than ``k + 1`` (matching the original fallback).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, C = X.shape
        out = np.zeros(C)
        if n == 0:
            return out

        nx1 = X.sum(axis=0)
        nx0 = n - nx1
        valid = (nx0 >= k + 1) & (nx1 >= k + 1)
        if not np.any(valid):
            return out

        Xc = X[:, valid].T                              # (V, n)
        D = np.abs(y[:, None] - y[None, :])             # (n, n) fixed

        # Per candidate / point i: distance to the (k+1)-th nearest SAME-class
        # point (self included at 0) -> the KSG radius.
        same = Xc[:, :, None] == Xc[:, None, :]         # (V, n, n)
        masked = np.where(same, D[None, :, :], np.inf)
        radius = np.partition(masked, k, axis=2)[:, :, k]   # (V, n)

        # m_i = # of ALL-class points strictly inside the radius (the
        # ``radius - 1e-15`` in the original excludes the k-th neighbour).
        m = (D[None, :, :] <= (radius[:, :, None] - 1e-15)).sum(axis=2)  # (V, n)

        nx0_v, nx1_v = nx0[valid], nx1[valid]
        avg_psi_nx = (nx0_v * psi(nx0_v) + nx1_v * psi(nx1_v)) / n
        avg_psi_m = psi(m).mean(axis=1)
        out[valid] = np.maximum(0.0, psi(n) - avg_psi_nx + psi(k) - avg_psi_m)
        return out

    def _mrmr_next(self, coreset, knn):
        """One greedy mRMR step on the source scores restricted to ``knn`` rows."""
        S = self._source_scores_full[knn]                      # (kk, N)
        per_model_mean = S.mean(axis=1)                        # (kk,)
        mi_rel, mi_red_batch = self._get_mi_estimators(self._binary)

        selected = set(coreset)
        rem = np.array([c for c in range(S.shape[1]) if c not in selected])

        if self._binary:
            # Vectorised Ross estimator (binary candidates vs continuous mean).
            rel = self._ross_relevance_batch(S[:, rem], per_model_mean, self.mi_k)
        else:
            rel = np.array([
                mi_rel(S[:, c], per_model_mean, k=self.mi_k) for c in rem
            ])
        if self.only_relevance:
            return int(rem[int(np.argmax(rel))])

        red_sum = np.zeros(len(rem))
        for j in coreset:
            red_sum += mi_red_batch(S[:, rem], S[:, j])
        mean_red = red_sum / len(coreset)

        if self.miq_scheme:
            score = rel / (mean_red + 1e-10)
        else:
            score = rel - mean_red
        return int(rem[int(np.argmax(score))])

    # -- persistence --------------------------------------------------------

    def save(self, path_save):
        jbl.dump(
            {
                "embedder": self.embedder,
                "embed_model": self.embed_model,
                "n_components_cmds": self.n_components_cmds,
                "k": self.k,
                "mi_k": self.mi_k,
                "miq_scheme": self.miq_scheme,
                "only_relevance": self.only_relevance,
                "binary": self._binary,
                "item1": self._item1,
                "coreset_size": self.coreset_size,
                "source_resp_full": self._source_resp_full,
                "source_scores_full": self._source_scores_full,
                "source_mean": self._source_mean,
            },
            path_save,
        )

    def load(self, path_load):
        state = jbl.load(path_load)
        self.embedder = state["embedder"]
        self.embed_model = state["embed_model"]
        self.n_components_cmds = state["n_components_cmds"]
        self.k = state["k"]
        self.mi_k = state["mi_k"]
        self.miq_scheme = state["miq_scheme"]
        self.only_relevance = state["only_relevance"]
        self._binary = state["binary"]
        self._item1 = state["item1"]
        self.coreset_size = state["coreset_size"]
        self._source_resp_full = state["source_resp_full"]
        self._source_scores_full = state["source_scores_full"]
        self._source_mean = state["source_mean"]
        return self


def _make_dkps_mrmr_variant(embedder="onehot", k=10, mi_k=3,
                            miq_scheme=False, only_relevance=False):
    """Build a :class:`DKPSMRMRPred` subclass with the given hyper-parameters.

    The registry key / class name encodes the non-default knobs, e.g.
    ``dkps_mrmr`` (defaults), ``dkps_mrmr_k5``, ``dkps_mrmr_MIQ``,
    ``dkps_mrmr_google``.
    """

    class _Variant(DKPSMRMRPred):
        pass

    _Variant.embedder = embedder
    _Variant.k = k
    _Variant.mi_k = mi_k
    _Variant.miq_scheme = miq_scheme
    _Variant.only_relevance = only_relevance

    tag = ""
    if embedder != "onehot":
        tag += f"_{embedder}"
    if k != 10:
        tag += f"_k{k}"
    if mi_k != 3:
        tag += f"_mik{mi_k}"
    if only_relevance:
        tag += "_MI"
    elif miq_scheme:
        tag += "_MIQ"
    else:
        tag += "_MID"
    name = f"DKPSMRMR{tag}"
    _Variant.__name__ = name
    _Variant.__qualname__ = name
    _Variant.__module__ = __name__
    _Variant.__doc__ = (
        f"DKPS-guided iterative mRMR ({embedder} embedder, k={k}, mi_k={mi_k}, "
        f"scheme={'MI' if only_relevance else ('MIQ' if miq_scheme else 'MID')})."
    )
    return _Variant
