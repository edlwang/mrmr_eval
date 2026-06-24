# Plan: Add DKPS as a method in `mrmr_eval`

> **Status: implemented** (2026-06-22). Files added: `benchpred/_vendor/{__init__,dkps_cmds,dkps_embed}.py`,
> `benchpred/dkps.py`. Files edited: `benchpred/__init__.py` (guarded registration
> of `dkps` + `dkps_google`), `method_runner.py` (flag-gated `model_outputs`
> plumbing), `pyproject.toml` (optional `[dkps]` extra). All files pass
> `py_compile`. **Not yet run** end-to-end — needs the repo env + HELM data
> (and `GEMINI_API_KEY` for `dkps_google`); see Validation.

## Goal & key constraint

Add `DataKernelPerspectiveSpace` (**vendored** from `~/code/dkps` — copied into
this repo, not installed as an external package) as a registered method so
`python main.py --data_source helm --methods dkps …` runs it end-to-end,
comparable to mRMR / IRT / RandomSampling.

The hard part is a **modality mismatch**. Every method in this repo conforms to
`BenchPred` (`benchpred/base.py:64`) and the runner only ever hands a method a
**scores** matrix:

- `method.fit(source_full_scores=scores[source_models], …)` — `method_runner.py:427`
- `method.predict(scores[target_models][:, coreset])` — `method_runner.py:438`

But DKPS embeds raw model **responses** (`dkps/dkps.py:28`), not binary scores.
Per the decision (real response embeddings, HELM-only), responses are needed in
`fit`/`predict`.

**The data layer already exposes them — no changes to `data_utils.py` or
`data_loader.py` are needed.** Per-instance raw `predicted_text` is captured as
`model_output` (`data_utils.py:522`), assembled into a `(n_instances, n_models)`
`model_outputs` field by `get_datasets()` (`data_utils.py:219-233`), transposed
to `(n_models, n_instances)` — aligned column-for-column with `scores` — by
`HelmsLoader.load()` (`data_loader.py:51-52`), and **already unpacked into a
local variable by the runner** (`method_runner.py:541`).

The *only* gap is the last hop: that local `model_outputs` is never forwarded
from `run_for_dataset` into the per-trial functions, so it never reaches
`fit`/`predict`. So the mandatory runner change is small and confined to
`method_runner.py` — wiring an already-loaded variable one level deeper (this is
distinct from wiring DKPS into `DEFAULT_METHODS`/notebooks, which is left out).

DKPS is also **transductive**: the target model's responses participate in
building the MDS space (`dkps/docs/dkps-prediction.md:42`). Its semantics are
kept intact by stashing source responses in `fit` and running the full
transductive `fit_transform` inside `predict`, mirroring
`examples/helm/runners/dkps.py:run_one`.

## How DKPS maps onto the `BenchPred` interface

| `BenchPred` concept | DKPS realization |
|---|---|
| coreset (column indices) | a **random sample** of `coreset_size` query/question indices (DKPS does no informative selection — this matches the HELM harness which samples `n` queries, and makes it the apples-to-apples partner of `random_sampling_and_learn`) |
| `fit(source_full_scores, …)` | pick random coreset columns; stash source **raw responses** on those columns + per-source regression target `source_full_scores.mean(axis=1)` (= source `true_acc`) |
| `predict(target_coreset_outputs)` | embed source∪target coreset responses → `DataKernelPerspectiveSpace.fit_transform` → `LinearRegression` on source coords → predict target coords → `clip[0,1]` |

## Changes, file by file

### 1. Vendor DKPS into the repo (no external `pip install`)
Copy the one class we actually use — `DataKernelPerspectiveSpace`
(`~/code/dkps/dkps/dkps.py:10-61`) — into this repo instead of depending on the
external `dkps` package.

**Provenance/license:** source is the fork `github.com/edlwang/dkps` at commit
`5a84ea8`. No `LICENSE` file exists upstream; since it's your own fork this is
low-risk, but record the source URL + commit in a module header and confirm the
original `hhelm10/dkps` license terms before publishing this repo.

**Placement — keep vendored upstream code separate from our adapter:**
- `benchpred/_vendor/dkps_cmds.py` — verbatim `DataKernelPerspectiveSpace`
  (`dkps/dkps.py:10-61`), under a provenance header.
- `benchpred/_vendor/dkps_embed.py` — **slimmed, Google-only** copy of
  `dkps/embed.py`'s `embed_api` + the `disk_cache` helper from `dkps/cache.py`.
  All other providers (jina / openrouter / litellm / huggingface /
  sentence-transformers / jlai) and the `rich` / `httpx` deps are dropped, so the
  only extra dep for the text path is `google-genai`. The one-hot embedder is
  implemented directly in the adapter (pure numpy, adapted from
  `examples/helm/utils.py:43`). The rest of the upstream tree (`utils.py`,
  `examples/`) is **not** copied.
- `benchpred/dkps.py` — our `DKPSPred(BenchPred)` adapter (one-hot) + the
  `DKPSGooglePred` subclass, importing from `._vendor.dkps_cmds` (and lazily
  `._vendor.dkps_embed` for the Google path).

The MDS class's imports are `numpy`, `sklearn.metrics.pairwise_distances`,
`scipy.spatial.distance.{pdist,squareform}` — all already repo deps — plus
`graspologic.embed.ClassicalMDS`. The Google embedder additionally needs
`google-genai` and `GEMINI_API_KEY`.

**Decision: `graspologic` (+ `google-genai`) live in an optional `[dkps]`
extra**, not the core deps. `pyproject.toml` gains:

```toml
[project.optional-dependencies]
dkps = ["graspologic", "google-genai"]
```

Install with `pip install -e .[dkps]`. Vendoring `DataKernelPerspectiveSpace`
already removes the entire `jlai`/git-fork/other-provider closure, so only
`graspologic` (always) and `google-genai` (only for `dkps_google`) cross the
boundary, and the registration is guarded (§3) so a missing extra simply skips
DKPS instead of breaking the registry. **Caveat:** this repo pins `numpy<2`
(`pyproject.toml`); the resolved `graspologic` must be compatible with that —
verify it imports and runs cleanly against the installed numpy/scipy/sklearn
before relying on it, and pin it in the extra once confirmed.

### 2. New file `benchpred/dkps.py` — `DKPSPred(BenchPred)`
Implements the 6 interface methods plus a `requires_model_outputs = True` class
flag the runner keys off:

```python
import numpy as np, joblib as jbl
from sklearn.linear_model import LinearRegression
from .base import BenchPred, set_random_seed
from ._vendor.dkps_cmds import DataKernelPerspectiveSpace   # vendored, see §1

class DKPSPred(BenchPred):
    requires_model_outputs = True          # runner passes responses, not scores
    embedder = "onehot"                    # "onehot" (dep-free) | "google" | "sentence-transformers"
    n_components_cmds = 8

    def fit(self, source_full_scores, coreset_size, seed=42,
            source_model_outputs=None, **kwargs):
        set_random_seed(seed)
        M, N = source_full_scores.shape
        k = min(int(coreset_size), N)
        self.compressed_data_indices = np.sort(np.random.choice(N, k, replace=False))
        self._source_resp   = source_model_outputs[:, self.compressed_data_indices]  # raw, object
        self._source_scores = source_full_scores.mean(axis=1)
        return self

    def get_coreset(self):
        return self.compressed_data_indices

    def predict(self, target_coreset_outputs):              # raw responses, (M_tgt, k)
        emb = self._embed_union(self._source_resp, target_coreset_outputs)  # consistent dim
        n_src = len(self._source_resp)
        data = {f"m{i}": emb[i][:, None, :] for i in range(len(emb))}       # (k,1,dim)
        nc = min(self.n_components_cmds, n_src - 1)
        P = DataKernelPerspectiveSpace(n_components_cmds=nc).fit_transform(data)
        Xs = np.vstack([P[f"m{i}"] for i in range(n_src)])
        Xt = np.vstack([P[f"m{i}"] for i in range(n_src, len(emb))])
        lr = LinearRegression().fit(Xs, self._source_scores)
        return np.clip(lr.predict(Xt), 0.0, 1.0)

    def save(self, p): jbl.dump({...}, p)
    def load(self, p): ...; return self
```

Critical correctness point — **one-hot vocabulary must span source ∪ target**
(`dkps/docs/dkps-prediction.md:135`): if the label set is built from source
responses alone, a target-only label changes `embedding_dim` and trips DKPS's
"all arrays must have the same shape" assert (`dkps/dkps.py:37`). That's why raw
responses are stashed in `fit` and embedded together inside `predict` (`_embed`
over `concat([source_resp, target])`). Both embedders run over the union: the
`onehot` default builds its vocabulary from the union (out-of-vocab → all-zero),
and `google` produces a fixed embedding dim regardless. (The actual code uses
integer dict keys and a single `_embed`; the sketch above used `_embed_union` for
clarity.) One-hot is pure numpy; `google` (`DKPSGooglePred`) calls the vendored
`embed_api`, de-duplicating responses to minimise/cache API calls.

### 3. Register it — `benchpred/__init__.py`
Registered at the **end** of the module (after the backfill block) so the
dynamic mRMR/backfill loops above can't touch the new keys:
- `all_methods["dkps"] = DKPSPred` (one-hot, categorical tasks).
- `all_methods["dkps_google"] = DKPSGooglePred` (Gemini embedder, free-text tasks).
- The existing `_make_krr_variant` / backfill helpers operate on coreset *scores*
  and aren't meaningful for a response-based method, so they're skipped.
- **Import safety**: `benchpred/__init__.py` is imported on every run and in
  every worker (`method_runner.py:40`, `main.py:19`), so the DKPS import runs for
  *all* methods and pulls `graspologic`. It's wrapped in a
  `try/except ImportError` that registers `dkps` only if `graspologic` imports,
  so a missing/broken `graspologic` can't take down the whole registry.

### 4. Plumb `model_outputs` through the runner — `method_runner.py`
Minimal, flag-gated branch so all other methods are untouched:
- `run_for_dataset`: include `model_outputs` in each task tuple
  (`method_runner.py:566-571`). To avoid pickling a large object array into
  every worker on normal runs, pass `model_outputs` only if
  `any(getattr(all_methods[m], "requires_model_outputs", False) …)`, else pass
  `None`.
- Add the `model_outputs` positional param (right after `scores`) to
  `_run_single_trial` (`method_runner.py:302`) and `_do_single_trial`
  (`method_runner.py:378`), forwarding it through (`method_runner.py:339`).
- In `_do_single_trial`, branch on the flag:
  - if `requires_model_outputs` and `model_outputs is None` → raise a clear
    error ("DKPS needs model responses; only `data_source='helm'` provides
    them").
  - `fit(…, source_model_outputs=model_outputs[source_models])` (passed via the
    existing `**kwargs`; all other methods ignore it).
  - test predict → `model_outputs[target_models][:, compressed_indices]`; train
    predict → `model_outputs[source_models][:, compressed_indices]` (replacing
    the two `scores[…]` calls at `:438` and `:443`).

Everything downstream (residuals, MAE/RMSE, correlations, result/ckpt dump) is
unchanged because `predict` still returns a float accuracy vector.

## Edge cases & risks
- **HELM-only, two methods by task type**: use `dkps` (one-hot, **no API key**)
  for categorical tasks (`med_qa`, `legalbench`); use `dkps_google` (Gemini, needs
  `GEMINI_API_KEY` + the `[dkps]` extra) for free-text tasks (`math`, `wmt_14`)
  where one-hot is meaningless. One-hot on free text would technically run but be
  degenerate (each distinct string its own dimension), so pick the method to match
  the task.
- **Embedding cost under multiprocessing**: each trial re-embeds, and `predict`
  runs twice (test + train). `dkps` (one-hot) is pure numpy → effectively free.
  `dkps_google` de-dups responses and the vendored `embed_api` disk-caches chunks
  under `$DKPS_CACHE_DIR/embed/google` (default `./.cache/embed/google`), so
  repeated/overlapping responses across the two predict calls, seeds, and workers
  hit the cache. Still, many concurrent workers hitting the Gemini API can rate-
  limit — consider `--no-multi_process` or a pre-embed pass for large free-text
  sweeps. A future optimization is to pre-embed each dataset once.
- **Few source models**: MDS + 8-dim regression needs more sources than dims
  (`dkps/docs/dkps-prediction.md:127`). Default `num_train_models=30`
  (`main.py:84`) is fine; still clamp `n_components_cmds = min(8, n_source-1)`.
- **Pre-existing loader inconsistency** (out of scope, worth flagging):
  `OpenLLMLoader`/`GlueLoader` return 3-tuples (`data_loader.py:40,70`) while the
  runner unpacks 4 (`method_runner.py:541`) — those sources already can't run
  through the runner. DKPS is HELM-only so unaffected, but it confirms DKPS must
  be run with `--data_source helm`.

## Compatibility — impact on existing methods
The other ~100 registered methods are functionally and numerically **unchanged**:
- **Runner branch is flag-gated.** Existing methods don't set
  `requires_model_outputs`, so `getattr(..., False)` routes them to the unchanged
  `scores`-based `fit`/`predict` calls (`method_runner.py:427,438,443`). The new
  `source_model_outputs=` kwarg is passed only on the DKPS branch, so existing
  `fit` signatures never see an unexpected argument.
- **Reproducibility/cache preserved.** Forwarding `model_outputs` adds no
  `np.random` calls, so `set_random_seed(seed)` → `_compute_split`
  (`method_runner.py:560-561`) yields identical source/target splits; existing
  cached `result.jbl` stay valid and `_is_result_stale` (`:357`) is untouched.
- **No overhead on normal runs** if `model_outputs` is gated to `None` unless a
  requested method needs it (§4).
- **Registry addition is inert** for existing methods: the dynamic variant loops
  and backfill list filter by prefix / explicit membership
  (`benchpred/__init__.py:114-127,136-143`); `"dkps"` matches none, so no
  accidental `kdkps`/`dkps+` and no change to existing variant generation. (Minor:
  `--methods all`, `main.py:119`, now includes `dkps`.)
- **Dependency-coupling risk is the reason we vendor.** Installing the external
  `dkps` would pull git-fork `graspologic`/`jlai` and pin
  `scikit-learn>=1.7.1,<2` / `scipy>=1.16,<2` into this shared env, which could
  shift or break existing methods that lean on sklearn/scipy
  (`RidgeCV`/`KernelRidge` in mrmr/lasso/backfill/`base.py:118`; `scipy.stats`
  correlations at `method_runner.py:460-466`). Vendoring removes that closure:
  the only new dependency is one pinned `graspologic` you control, so it can't
  silently upgrade the sklearn/scipy that existing methods rely on.
- **One edit to do carefully**: inserting the `model_outputs` positional arg into
  the task tuple (`:566-571`) and the two trial signatures (`:302`, `:378`) must
  stay aligned with the trailing `(position_queue, progress_dict,
  mp_progress_mode)` appended at `:638-641`. Misalignment is the only way this
  change could break existing methods — smoke-test one non-DKPS method after.

## Validation
1. `python -c "from benchpred import all_methods; print('dkps' in all_methods)"`.
2. Smoke run on a categorical task with no API key:
   `python main.py --data_source helm --dataset_name med_qa --methods dkps --coreset_size 10% --num_run 1 --no-multi_process --no-use_git`
3. Confirm a sane MAE and compare against `random_sampling_and_learn` on the same
   split/seed.
4. Tiny synthetic check that `predict` is transductive and order-preserving
   (source rows reproduced as `pred_acc_train`).
5. Confirm the pinned `graspologic` imports and runs against the installed
   numpy/scipy/sklearn with no version conflict (and existing methods' results
   are unchanged on a fixed seed).
6. Sanity that vendoring is self-contained: `import benchpred` succeeds with the
   external `dkps` package uninstalled / off `PYTHONPATH` (only `graspologic`
   need be installed).

## Out of scope (per "register only")
No edits to `DEFAULT_METHODS` (`main.py:67`) or to `tutorial.ipynb`/`misc.ipynb`.
DKPS is reachable via `--methods dkps`.
