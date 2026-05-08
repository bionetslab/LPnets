import pandas as pd
import numpy as np
import networkx as nx


class GraphMetricComputer:

    def __init__(self, all_nodes):
        self.all_nodes = sorted(all_nodes)

    def calculate_graph_metrics(self, G):
        measures = {}

        # --- shared precompute ---
        nodes = sorted(G.nodes())
        n = len(nodes)

        # signed adjacency matrix for spectrum + norms + triangles
        if G.number_of_edges() > 0:
            A = nx.adjacency_matrix(
                G, nodelist=nodes, weight='weight'
            ).toarray().astype(float)
        else:
            A = np.zeros((n, n))
        A_abs = np.abs(A)

        # |w| version of G for methods that want a nx.Graph
        G_abs = G.copy()
        for u, v, d in G_abs.edges(data=True):
            d['weight'] = abs(d.get('weight', 1.0))

        # positive / negative subgraphs
        G_pos = nx.Graph(
            (u, v, d) for u, v, d in G.edges(data=True)
            if d.get('weight', 1.0) > 0
        )
        G_pos.add_nodes_from(G.nodes())

        G_neg = nx.Graph(
            (u, v, {'weight': abs(d.get('weight', 1.0))})
            for u, v, d in G.edges(data=True) if d.get('weight', 1.0) < 0
        )
        G_neg.add_nodes_from(G.nodes())

        # eigenvalues once, reused by spectral + matrix norms
        eigenvalues = None
        if n > 1 and G.number_of_edges() > 0:
            try:
                eigenvalues = np.sort(np.linalg.eigvalsh(A))[::-1]
            except Exception:
                eigenvalues = None

        weights = np.array(
            [d.get('weight', 1.0) for _, _, d in G.edges(data=True)]
        )

        # --- basic topology ---
        measures['num_edges'] = G.number_of_edges()

        # --- connected components ---
        components = list(nx.connected_components(G))
        measures['connected_components'] = len(components)

        # --- edge weight distribution ---
        measures.update(self._weight_distribution(weights))

        # --- node strength ---
        measures.update(self._strength_distributions(G, G_abs))

        # --- matrix norms ---
        measures.update(self._matrix_norms(A, eigenvalues))

        # --- per-node Frobenius ---
        measures.update(self._row_frobenius_stats(A_abs))

        # --- top-k biggest edges ---
        measures.update(self._top_k_weights(weights))

        # weighted clustering on |w|
        measures['avg_weighted_clustering'] = nx.average_clustering(
            G_abs, weight='weight'
        )

        # --- spectral ---
        measures.update(self._spectral_features(eigenvalues))

        # --- Fiedler value on LCC ---
        measures.update(self._algebraic_connectivity(G_abs, components))

        # --- triangles ---
        measures['num_triangles'] = self._count_triangles(A_abs)
        measures['frac_balanced_triangles'] = self._balanced_triangle_fraction(
            A, measures['num_triangles']
        )
        measures['avg_triangle_intensity'] = self._triangle_intensity(A_abs)

        # --- communities ---
        measures.update(self._weighted_modularity(G_abs))

        # --- disparity ---
        measures.update(self._disparity(G_abs))

        # --- positive / negative subgraph metrics ---
        measures.update(self._signed_subgraph_metrics(G_pos, G_neg))

        # --- eigenvector centralities on LCC ---
        measures.update(self._weighted_centralities(G_abs, components))

        return measures


    def _weight_distribution(self, weights):
        m = {}
        keys = ['weight_std', 'weight_min', 'weight_max',
                'weight_skewness', 'weight_kurtosis', 'frac_positive_edges',
                'weight_entropy']
        if len(weights) == 0:
            for k in keys:
                m[k] = 0
            return m

        m['weight_std'] = float(np.std(weights))
        m['weight_min'] = float(np.min(weights))
        m['weight_max'] = float(np.max(weights))
        m['weight_skewness'] = float(pd.Series(weights).skew()) if len(weights) >= 3 else 0
        m['weight_kurtosis'] = float(pd.Series(weights).kurtosis()) if len(weights) >= 4 else 0
        m['frac_positive_edges'] = float(np.mean(weights > 0))

        # Shannon entropy of normalized |w|
        abs_w = np.abs(weights)
        total = float(np.sum(abs_w))
        if total > 0:
            p = abs_w / total
            p = p[p > 0]
            m['weight_entropy'] = float(-np.sum(p * np.log(p)))
        else:
            m['weight_entropy'] = 0
        return m

    def _strength_distributions(self, G, G_abs):
        # node strength = sum of edge weights on that node
        m = {}
        signed_strength = dict(G.degree(weight='weight'))
        abs_strength = dict(G_abs.degree(weight='weight'))

        vals = (np.array(list(signed_strength.values()))
                if signed_strength else np.array([0.0]))
        m['signed_strength_mean'] = float(np.mean(vals))
        m['signed_strength_std'] = float(np.std(vals))
        m['signed_strength_max'] = float(np.max(vals))
        m['signed_strength_min'] = float(np.min(vals))

        vals = (np.array(list(abs_strength.values()))
                if abs_strength else np.array([0.0]))
        m['abs_strength_max'] = float(np.max(vals))
        return m

    def _matrix_norms(self, A, eigenvalues):
        # frobenius_norm = sqrt(sum w^2), total anomaly energy
        # nuclear_norm = sum |lambda|, rank-aware version
        # stable_rank = ||A||_F^2 / ||A||_2^2
        # participation_ratio = (sum l^2)^2 / sum l^4, effective # of modes
        m = {}
        keys = ['frobenius_norm', 'nuclear_norm', 'stable_rank',
                'participation_ratio']
        if A.size == 0 or not np.any(A):
            for k in keys:
                m[k] = 0
            return m

        m['frobenius_norm'] = float(np.linalg.norm(A, 'fro'))

        if eigenvalues is not None and len(eigenvalues) > 0:
            abs_eig = np.abs(eigenvalues)
            spec_radius = float(np.max(abs_eig))
            m['nuclear_norm'] = float(np.sum(abs_eig))
            m['stable_rank'] = (
                float(np.sum(eigenvalues ** 2) / (spec_radius ** 2))
                if spec_radius > 0 else 0
            )
            sum_l2 = float(np.sum(eigenvalues ** 2))
            sum_l4 = float(np.sum(eigenvalues ** 4))
            m['participation_ratio'] = (
                (sum_l2 ** 2) / sum_l4 if sum_l4 > 0 else 0
            )
        else:
            m['nuclear_norm'] = 0
            m['stable_rank'] = 0
            m['participation_ratio'] = 0
        return m

    def _row_frobenius_stats(self, A_abs):
        # per-node anomaly magnitude, doesn't let +/- cancel
        keys = ['row_frob_max']
        if A_abs.size == 0 or A_abs.shape[0] == 0:
            return {k: 0 for k in keys}

        row_frob = np.sqrt(np.sum(A_abs ** 2, axis=1))
        return {
            'row_frob_max': float(np.max(row_frob)),
        }

    def _top_k_weights(self, weights):
        # sum of top-10 biggest |edges|
        if len(weights) == 0:
            return {'top10_abs_weight_sum': 0}
        abs_w = np.sort(np.abs(weights))[::-1]
        return {'top10_abs_weight_sum': float(np.sum(abs_w[:10]))}

    def _spectral_features(self, eigenvalues):
        # extremal eigenvalues of the signed adjacency
        m = {}
        keys = ['largest_eigenvalue', 'smallest_eigenvalue', 'spectral_gap']
        if eigenvalues is None or len(eigenvalues) == 0:
            for k in keys:
                m[k] = 0
            return m

        m['largest_eigenvalue'] = float(eigenvalues[0])
        m['smallest_eigenvalue'] = float(eigenvalues[-1])
        m['spectral_gap'] = (
            float(eigenvalues[0] - eigenvalues[1])
            if len(eigenvalues) > 1 else 0
        )
        return m

    def _algebraic_connectivity(self, G_abs, components):
        # Fiedler value on LCC
        m = {}
        if G_abs.number_of_nodes() > 1 and components:
            largest_cc = max(components, key=len)
            if len(largest_cc) > 1:
                G_lcc = G_abs.subgraph(largest_cc).copy()
                try:
                    val = nx.algebraic_connectivity(G_lcc, weight='weight')
                    # networkx sometimes returns NaN instead of raising
                    m['algebraic_connectivity'] = (
                        float(val) if np.isfinite(val) else 0
                    )
                except Exception:
                    m['algebraic_connectivity'] = 0
            else:
                m['algebraic_connectivity'] = 0
        else:
            m['algebraic_connectivity'] = 0
        return m

    @staticmethod
    def _count_triangles(W):
        # trace(A^3) / 6 counts triangles; round() because floats.
        n = W.shape[0]
        if n < 3:
            return 0
        A_bin = (W > 0).astype(float)
        try:
            return int(round(np.trace(A_bin @ A_bin @ A_bin) / 6.0))
        except Exception:
            return 0

    @staticmethod
    def _balanced_triangle_fraction(A, num_triangles):
        # trace(sign(A)^3)/6 = #balanced - #unbalanced
        if num_triangles == 0 or A.size == 0:
            return 0
        try:
            A_sign = np.sign(A)
            bal_minus_unbal = round(np.trace(A_sign @ A_sign @ A_sign) / 6.0)
            n_balanced = int(round((num_triangles + bal_minus_unbal) / 2))
            return float(n_balanced / num_triangles)
        except Exception:
            return 0

    @staticmethod
    def _triangle_intensity(W):
        # Onnela's triangle intensity: geometric mean of the 3 edge weights per triangle, averaged
        n = W.shape[0]
        if n < 3:
            return 0

        A_bin = (W > 0)
        intensities = []
        for i in range(n):
            nbrs_i = np.where(A_bin[i])[0]
            nbrs_i = nbrs_i[nbrs_i > i]
            for jdx, j in enumerate(nbrs_i):
                for k in nbrs_i[jdx + 1:]:
                    if A_bin[j, k]:
                        w1, w2, w3 = W[i, j], W[i, k], W[j, k]
                        intensities.append((w1 * w2 * w3) ** (1.0 / 3.0))
        return float(np.mean(intensities)) if intensities else 0

    def _weighted_modularity(self, G_abs):
        # greedy modularity on |w|
        m = {}
        keys = ['modularity', 'community_size_std']
        if G_abs.number_of_edges() == 0:
            for k in keys:
                m[k] = 0
            return m

        try:
            communities = list(
                nx.community.greedy_modularity_communities(
                    G_abs, weight='weight'
                )
            )
            m['modularity'] = float(
                nx.community.modularity(G_abs, communities, weight='weight')
            )
            sizes = [len(c) for c in communities]
            m['community_size_std'] = float(np.std(sizes))
        except Exception:
            for k in keys:
                m[k] = 0
        return m

    def _disparity(self, G_abs):
        # Barrat disparity: Y_i = sum_j (w_ij / s_i)^2; is a node's weight spread evenly or concentrated on one edge?
        m = {}
        keys = ['disparity_mean', 'disparity_std']
        disparities = []
        for node in G_abs.nodes():
            neighbors = list(G_abs.neighbors(node))
            if not neighbors:
                continue
            strength = sum(
                G_abs[node][nbr].get('weight', 1.0) for nbr in neighbors
            )
            if strength == 0:
                continue
            y_i = sum(
                (G_abs[node][nbr].get('weight', 1.0) / strength) ** 2
                for nbr in neighbors
            )
            disparities.append(y_i)

        if disparities:
            m['disparity_mean'] = float(np.mean(disparities))
            m['disparity_std'] = float(np.std(disparities))
        else:
            for k in keys:
                m[k] = 0
        return m

    def _signed_subgraph_metrics(self, G_pos, G_neg):
        # separate metrics on + and - subgraphs
        m = {}
        for prefix, subG in [('pos', G_pos), ('neg', G_neg)]:
            m[f'{prefix}_num_edges'] = subG.number_of_edges()

            if subG.number_of_edges() > 0:
                ws = np.array([
                    d.get('weight', 1.0)
                    for _, _, d in subG.edges(data=True)
                ])
                m[f'{prefix}_weight_mean'] = float(np.mean(ws))
                m[f'{prefix}_weight_std'] = float(np.std(ws))

                strengths = np.array(list(
                    dict(subG.degree(weight='weight')).values()
                ))
                m[f'{prefix}_strength_mean'] = float(np.mean(strengths))
                m[f'{prefix}_strength_std'] = float(np.std(strengths))

                # only pos gets clustering, neg is basically always empty of triangles
                if prefix == 'pos':
                    m['pos_avg_clustering'] = float(
                        nx.average_clustering(subG, weight='weight')
                    )
            else:
                for suf in ['weight_mean', 'weight_std',
                            'strength_mean', 'strength_std']:
                    m[f'{prefix}_{suf}'] = 0
                if prefix == 'pos':
                    m['pos_avg_clustering'] = 0
        return m

    def _weighted_centralities(self, G_abs, components):
        # eigenvector centrality on LCC
        m = {}
        keys = ['lcc_avg_eigenvector_centrality',
                'lcc_max_eigenvector_centrality',
                'lcc_std_eigenvector_centrality']
        if G_abs.number_of_nodes() > 1 and components:
            largest_cc = max(components, key=len)
            if len(largest_cc) > 1:
                G_lcc = G_abs.subgraph(largest_cc).copy()
                try:
                    ev = nx.eigenvector_centrality(
                        G_lcc, weight='weight', max_iter=1000
                    )
                    ev_vals = np.array(list(ev.values()))
                    m['lcc_avg_eigenvector_centrality'] = float(np.mean(ev_vals))
                    m['lcc_max_eigenvector_centrality'] = float(np.max(ev_vals))
                    m['lcc_std_eigenvector_centrality'] = float(np.std(ev_vals))
                except Exception:
                    for k in keys:
                        m[k] = 0
            else:
                for k in keys:
                    m[k] = 0
        else:
            for k in keys:
                m[k] = 0
        return m
