"""Data Kernel Perspective Space (DKPS) — vendored.

Verbatim copy of ``DataKernelPerspectiveSpace`` from the DKPS project, kept here
so benchpred does not depend on the external ``dkps`` package being installed.

Provenance:
    source : https://github.com/edlwang/dkps  (dkps/dkps.py, lines 10-61)
    commit : 5a84ea8945ba329d4157e57433d74fbc5980f15d

The only third-party import is ``graspologic.embed.ClassicalMDS`` (classical
multidimensional scaling); numpy/scipy/scikit-learn are already benchpred deps.
Install graspologic via the optional extra: ``pip install -e .[dkps]``.

Upstream carried a commented-out ``DataKernelFunctionalSpace`` prototype; it is
unused here and intentionally not vendored.
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from graspologic.embed import ClassicalMDS

from scipy.spatial.distance import pdist, squareform


class DataKernelPerspectiveSpace:
    def __init__(
            self,
            response_distribution_fn=None,
            response_distribution_axis=1,
            metric_cmds='euclidean',
            n_components_cmds=None,
            n_elbows_cmds=2,
            dissimilarity="precomputed",
        ):

        self.response_distribution_fn   = response_distribution_fn
        self.response_distribution_axis = response_distribution_axis
        self.metric_cmds                = metric_cmds
        self.n_components_cmds          = n_components_cmds
        self.n_elbows_cmds              = n_elbows_cmds
        self.dissimilarity              = dissimilarity

    def fit_transform(self, data, return_dict=True):
        """
        data: dict {model_name: np.array(n_queries, n_replicates, embedding_dim)}
        """

        # qc checks
        assert isinstance(data, dict),                                  'data must be a dict'
        assert all([isinstance(x, np.ndarray) for x in data.values()]), 'all values must be numpy arrays'
        assert all([x.ndim == 3 for x in data.values()]),               'all arrays must be 3D - np.array(n_queries, n_replicates, embedding_dim)'
        assert len(set([x.shape for x in data.values()])) == 1,         'all arrays must have the same shape'

        # aggregate over replicates -> (n_models, n_queries, embedding_dim)
        if self.response_distribution_fn is None:
            X = np.stack([v[:, 0] for v in data.values()])
        else:
            X = np.stack([self.response_distribution_fn(v, axis=self.response_distribution_axis) for k, v in data.items()])

        n_models, n_queries, embedding_dim = X.shape

        # flatten -> (n_models, n_queries * embedding_dim)
        X_flat = X.reshape(len(X), -1)

        if self.metric_cmds == 'euclidean':
            dist_matrix = pairwise_distances(X_flat, metric='euclidean') / np.sqrt(n_queries)
            dist_matrix = (dist_matrix + dist_matrix.T) / 2
        else:
            dist_matrix = squareform(pdist(X_flat, metric=self.metric_cmds)) / np.sqrt(n_queries)

        cmds_embds = ClassicalMDS(n_components=self.n_components_cmds, n_elbows=self.n_elbows_cmds, dissimilarity=self.dissimilarity).fit_transform(dist_matrix)

        if return_dict:
            return {key: cmds_embds[i] for i, key in enumerate(data.keys())}
        else:
            return cmds_embds
