"""Regenerate plots/combined_grid_med_qa_filtered.pdf with dkps_mrmr included.

Reuses the tutorial notebook's own aggregation / style / plotting cells so the
figure matches the tutorial exactly.  Set RUN_DKPS=1 to (re)compute the
dkps_mrmr trials the plot needs (other methods are read from the cached
results/ tree); RUN_DKPS=0 just re-plots from whatever is on disk.
"""
import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, kendalltau
import joblib as jbl

warnings.filterwarnings("ignore")

# --- tutorial configuration (mirrors notebook cells 2-3) -------------------
DATASET = "med_qa"
RESULTS_DIR = "./results"
MODEL_SPLIT_METHOD = "binned_interpolation"
CORESET_SIZES = ["1%", "2%", "3%", "4%", "5%", "10%", "15%"]
NUM_TRAIN_MODELS = [20, 30, 40, 50]
NUM_SEEDS = 3
FIXED_CORESET_FOR_BOTTOM_ROW = "1%"
EXCLUDE_FROM_PLOT = set()
METHODS = [
    "random_sampling_and_learn",
    "random_sampling",
    "dkps",
    "dkps_mrmr",
    "kmrmr5_MIQ_y",
    "dkps+mrmr5_MIQ_y",
    "kanchor_points_weighted+",
    "dkps+anchor_points_weighted",
    "krandom_search_and_learn",
]


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


# --- optionally compute the dkps_mrmr cells the plot displays --------------
if os.environ.get("RUN_DKPS", "0") == "1":
    from zarth_utils.config import Config
    from method_runner import MethodRunner

    fixed_nmodels = NUM_TRAIN_MODELS[len(NUM_TRAIN_MODELS) // 2]  # top row
    needed = sorted(
        {(fixed_nmodels, c) for c in CORESET_SIZES}
        | {(n, FIXED_CORESET_FOR_BOTTOM_ROW) for n in NUM_TRAIN_MODELS}
    )
    print(f"dkps_mrmr cells to compute: {needed}", flush=True)
    for n, c in needed:
        print(f"\n##### dkps_mrmr  nmodels={n}  coreset={c} #####", flush=True)
        cfg = Config(
            default_config_dict={
                "data_source": "helm",
                "datasets": [DATASET],
                "dir_results": RESULTS_DIR,
                "exp_suffix": "",
                "coreset_size": c,
                "methods": ["dkps_mrmr"],
                "model_split_method": MODEL_SPLIT_METHOD,
                "num_train_models": int(n),
                "seed_start": 0,
                "num_run": NUM_SEEDS,
                "multi_process": False,
                "mp_progress_mode": "terminal",
                "use_git": False,
            },
            use_argparse=False,
        )
        MethodRunner(cfg).run_all_datasets()

# --- aggregate + style + plot by exec'ing the notebook's own cells ---------
nb = json.load(open("tutorial.ipynb"))
cells = {c["id"]: "".join(c["source"]) for c in nb["cells"] if "id" in c}
g = globals()
exec(cells["94a69310"], g)   # aggregation -> builds `raw`, `summary`
exec(cells["3764cf52"], g)   # method_style / pretty_label helpers
exec(cells["71141d58"], g)   # plotting -> writes plots/combined_grid_med_qa_filtered.pdf

present = sorted(set(summary["method"]) & set(METHODS))  # noqa: F821
print(f"\nmethods plotted: {present}")
print("dkps_mrmr in plot:", "dkps_mrmr" in present)
print("DONE -> plots/combined_grid_med_qa_filtered.pdf")
