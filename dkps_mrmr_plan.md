# Plan: Iterative DKPS-guided mRMR coreset selection (`dkps_mrmr`)

Status: **IMPLEMENTED.** See `benchpred/dkps.py` (`DKPSMRMRPred`,
`_make_dkps_mrmr_variant`), `benchpred/__init__.py` (registry), and
`method_runner.py` (`requires_full_target_outputs` branch). Verified with
synthetic smoke tests (method contract + runner integration). Not yet run on a
real HELM dataset.

## 1. Summary

A new `BenchPred` method that builds the coreset **iteratively and per target
model**, alternating between:

1. **DKPS-embed** all models (source ∪ target) from their *raw responses* on the
   current coreset, and
2. a **model-local mRMR step**: find the `k` source models nearest the target in
   DKPS space, restrict the score matrix to those `k` models, and pick the next
   coreset item by one greedy mRMR step.

Final prediction is **DKPS transductive regression** (as in `DKPSPred`): embed
source ∪ target on the final per-target coreset, fit a `LinearRegression` from
source coordinates to source mean scores, and read off the target's prediction.

This is a transductive, response-based, HELM-only method — a direct sibling of
`benchpred/dkps.py:DKPSPred`.

### Locked design decisions (from clarification)

| Question | Decision |
| --- | --- |
| "k closest source models" — closest to what? | **The target model** (fully transductive, per-target coreset). |
| What does DKPS embed from? | **Raw model responses** (one-hot default / Gemini), HELM-only. |
| Final prediction model? | **DKPS transductive regression** (all source models are anchors). |
| Role of `k` | Restricts **only the inner mRMR selection**, not the final regression. |

## 2. Algorithm

### fit (cheap — just stash full source data)

```
fit(source_full_scores, source_model_outputs, coreset_size, k, seed):
    stash:  source responses (FULL, all items)
            source scores     (FULL, all items)   # needed for inner mRMR MI
            source mean score (regression target)
            coreset_size, k
    seed item:  item1 = global mRMR pick-1 over ALL source models
                (highest relevance to mean source score; shared across targets)
```

All adaptive work happens in `predict`, because it depends on the target — this
mirrors `DKPSPred`, which also defers embedding/regression to `predict`.

### predict (per target row)

```
predict(target_full_outputs):           # shape (M_target, N_all_items)
  for each target model t:
    coreset = [item1]
    while len(coreset) < coreset_size:
      # (a) DKPS-embed source ∪ {t} from responses on `coreset`
      emb   = DKPS(responses[source ∪ {t}][:, coreset])     # model -> R^d
      # (b) k source models closest to t in DKPS space
      knn   = argsort_dist(emb[source], emb[t])[:k]
      # (c) one greedy mRMR step on the score matrix restricted to knn rows
      cand  = all_items \ coreset
      rel   = MI(scores[knn, cand], mean_score_over_knn)     # relevance
      red   = mean_j∈coreset MI(scores[knn, cand], scores[knn, j])   # redundancy
      next  = argmax(rel - red)        # MID  (or rel/red for MIQ)
      coreset.append(next)
    # final transductive regression on the per-target coreset
    emb     = DKPS(responses[source ∪ {t}][:, coreset])
    reg     = LinearRegression().fit(emb[source], source_mean_scores)
    pred[t] = clip(reg.predict(emb[t]), 0, 1)
  return pred
```

Notes:
- The `while` loop's last embedding can be reused for the final regression
  (one fewer DKPS call per target).
- When `predict` is called on **source** rows (the runner does this for train
  predictions / correlation metrics), each source row is treated as the
  "target," exactly as `DKPSPred` already does. Minor self-inclusion
  inconsistency, mirrored from existing behaviour — not corrected here.

## 3. Reuse (don't rewrite)

- **mRMR MI machinery** — subclass `MRMRPred` to reuse `_get_mi_estimators`,
  `_mutual_information_ross_estimator` / `_mutual_information_discrete_discrete_batch`
  (binary) and `_mutual_information_lnc*` (continuous). The inner "pick next" is
  one iteration of the existing greedy loop, evaluated over the restricted
  `knn` rows.
- **`DataKernelPerspectiveSpace`** from `benchpred/_vendor/dkps_cmds.py` for the
  embedding, and the **response embedders** (`_embed_onehot` / `_embed_google`)
  from `DKPSPred` — factor them so both classes share them.

## 4. Defaults & parameters (chosen; easy to flip)

- **Inner mRMR scheme:** MID (`relevance − mean redundancy`). Expose MIQ and
  relevance-only variants via the same naming convention as the `mrmr*` family.
- **`mi_k`** (MI neighbours): default 3, reuse the existing `mi_k` variant
  factory.
- **`k`** (nearest source models): new tuneable. Register `dkps_mrmr_k{5,10,...}`
  variants. Guard `2 ≤ k ≤ n_source`; warn when `k` is very small (MI over `k`
  rows is noisy).
- **DKPS dimension:** clamp like `DKPSPred`:
  `n_comp = max(1, min(n_components_cmds=8, n_source − 1))`.
- **Item 1:** global (shared across targets) — matches "start by using mRMR to
  pick 1 item," and saves recomputation.
- **Embedder:** `onehot` default (`dkps_mrmr`), plus a `google` variant
  (`dkps_mrmr_google`) for free-text tasks.

## 5. Runner integration (requires a small `method_runner` change)

The standard contract — `get_coreset()` returns one global coreset and
`predict` receives only the coreset columns — **cannot express a per-target
adaptive coreset.** Proposed minimal change to `method_runner._do_single_trial`:

- Add a capability flag on the method, e.g. `requires_full_target_outputs = True`
  (implies `requires_model_outputs`).
- For such methods, skip coreset slicing and pass the **full** response matrices:
  - `pred_acc_test  = method.predict(model_outputs[target_models])`   # all items
  - `pred_acc_train = method.predict(model_outputs[source_models])`
- Result logging: there is no single coreset. Store the **per-target coresets**
  the method exposes (e.g. `method.target_coresets_`) under
  `result_dict["selection_metrics"]`; have `get_coreset()` return `item1` (or
  `None`) purely for interface compatibility.

This is additive and gated by the flag, so every existing method is unaffected
(same pattern as the existing `requires_model_outputs` branch).

## 6. Registry

In `benchpred/__init__.py`, inside the existing guarded DKPS `try` block:

```python
all_methods["dkps_mrmr"]        = DKPSMRMRPred           # onehot, MID, k=default
all_methods["dkps_mrmr_google"] = DKPSMRMRGooglePred
# + k / scheme / mi_k variants via small factories, mirroring the mrmr* and
#   dkps backfill registration loops.
```

## 7. Cost & risks

- **Cost is the main concern.** `predict` runs, per target model, a
  `coreset_size`-iteration loop where each iteration does (i) one MDS embedding
  over `n_source + 1` models and (ii) a relevance/redundancy MI sweep over all
  remaining candidate items restricted to `k` rows. Total ≈
  `O(T · coreset_size · (MDS + num_data · MI))`. For large model pools / large
  coresets this is much heavier than `DKPSPred` (which embeds once per predict).
  Mitigations to consider if too slow: cache/incrementally update embeddings,
  cap the candidate pool, or amortise item-1 / early picks across targets.
- **Small-`k` MI noise:** MI estimated from only `k` model rows is high-variance;
  keep `k` reasonably large.
- **Early embeddings are weak:** with 1–2 coreset items the DKPS space is nearly
  degenerate. Option (off by default): seed with 2–3 plain mRMR picks before the
  adaptive loop.
- **HELM-only:** like all DKPS methods, needs raw responses + the `[dkps]` extra;
  guarded import keeps the rest of the registry working without it.

## 8. Implementation checklist (once approved)

1. Factor shared response-embedding helpers out of `DKPSPred`.
2. Add `DKPSMRMRPred` (+ `google` variant) in `benchpred/dkps.py` (or a new
   `benchpred/dkps_mrmr.py`), subclassing to reuse mRMR MI + DKPS embed.
3. Add `requires_full_target_outputs` handling in `method_runner._do_single_trial`
   and per-target coreset logging.
4. Register methods + `k` / scheme / `mi_k` variants in `benchpred/__init__.py`.
5. Smoke test on a small HELM dataset (e.g. `med_qa`, `legalbench`) at a small
   coreset size; compare MAE / correlation against `dkps` and `mrmr*`.
```
