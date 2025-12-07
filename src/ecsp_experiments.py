import igraph as ig
import numpy as np
import random
from math import inf
import math
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================
# 0. Utility: normalization
# ============================================================

def normalize_edge_centrality(edge_centrality, mode="max"):
    """
    Normalize edge centrality scores for easier comparison / plotting.

    Parameters
    ----------
    edge_centrality : dict
        Mapping (u, v) -> score (u < v assumed).
    mode : str
        "max" : divide by maximum absolute value (default).
        "sum" : divide by sum of absolute values.

    Returns
    -------
    norm_centrality : dict
        Mapping (u, v) -> normalized score (float).
    """
    if not edge_centrality:
        return {}

    values = list(edge_centrality.values())

    if mode == "max":
        max_val = max(abs(v) for v in values)
        if max_val == 0:
            return {e: 0.0 for e in edge_centrality}
        return {e: (v / max_val) for e, v in edge_centrality.items()}

    elif mode == "sum":
        sum_val = sum(abs(v) for v in values)
        if sum_val == 0:
            return {e: 0.0 for e in edge_centrality}
        return {e: (v / sum_val) for e, v in edge_centrality.items()}

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


# ============================================================
# 1. Distances & ECSP on igraph
# ============================================================

def bfs_distances_ig(G, s):
    """
    Shortest-path distances from s using igraph.

    Returns list dist[v].
    """
    dist_list = G.shortest_paths(source=s)[0]
    return dist_list


def edge_central_shortest_path_ig(G, s, t, edge_centrality):
    """
    ECSP on an igraph.Graph:
    among all shortest s-t paths, find one maximizing sum of edge centrality.

    edge_centrality: dict with keys (u,v) where u < v (vertex indices)
    """
    n = G.vcount()
    dist = bfs_distances_ig(G, s)
    if dist[t] == inf:
        return None, None

    # DP arrays
    best = [-inf] * n
    pred = [-1] * n
    best[s] = 0.0

    nodes_by_dist = sorted(range(n), key=lambda v: dist[v])

    for u in nodes_by_dist:
        if best[u] == -inf:
            continue
        du = dist[u]
        # neighbors of u
        for v in G.neighbors(u):
            if dist[v] == du + 1:
                key = (u, v) if u < v else (v, u)
                edge_score = edge_centrality.get(key, 0.0)
                cand = best[u] + edge_score
                if cand > best[v]:
                    best[v] = cand
                    pred[v] = u

    if best[t] == -inf:
        return None, None

    # reconstruct path
    path = []
    cur = t
    while cur != -1:
        path.append(cur)
        cur = pred[cur]
    path.reverse()

    return path, best[t]


# ============================================================
# 2. Edge Betweenness, Closeness, Gravity, ECHO in igraph
# ============================================================

def compute_edge_betweenness_ig(G, norm_mode="max"):
    """
    Edge betweenness using igraph, normalized to [0,1].
    """
    bet = G.edge_betweenness()  # list length m
    edges = G.get_edgelist()
    ec = {}
    for idx, (u, v) in enumerate(edges):
        key = (u, v) if u < v else (v, u)
        ec[key] = bet[idx]
    return normalize_edge_centrality(ec, mode=norm_mode)


def compute_edge_closeness_ig(G, norm_mode="max"):
    """
    Edge closeness via line graph (igraph): each edge -> vertex in line graph.
    """
    LG = G.linegraph()
    clos = LG.closeness()  # one value per original edge
    edges = G.get_edgelist()
    ec = {}
    for idx, (u, v) in enumerate(edges):
        key = (u, v) if u < v else (v, u)
        ec[key] = clos[idx]
    return normalize_edge_centrality(ec, mode=norm_mode)


def compute_edge_gravity_ig(
    G, num_sources=100, num_targets=100, norm_mode="max", random_seed=0
):
    """
    Approximate gravity: count how often edges appear in all-shortest paths
    between sampled node pairs. This is a sampled version of gravity.
    """
    rng = random.Random(random_seed)

    edges = G.get_edgelist()
    ec = {}
    for (u, v) in edges:
        key = (u, v) if u < v else (v, u)
        ec[key] = 0.0

    n = G.vcount()
    vertices = list(range(n))

    num_sources = min(num_sources, n)
    sources = rng.sample(vertices, num_sources)

    for s in sources:
        targets = rng.sample(vertices, min(num_targets, n))
        for t in targets:
            if t == s:
                continue
            try:
                paths = G.get_all_shortest_paths(s, to=t)
            except:
                continue
            for path in paths:
                for u, v in zip(path, path[1:]):
                    key = (u, v) if u < v else (v, u)
                    ec[key] += 1.0

    return normalize_edge_centrality(ec, mode=norm_mode)


def compute_echo_centrality_ig(G, alpha=0.5, norm_mode="max"):
    """
    ECHO edge centrality for igraph, as linear system:
        (I - (alpha/2) E^T D^{-1} E) z = (1-alpha) x

    NOTE: This is dense and may not scale to very large graphs.
    """
    nodes = list(range(G.vcount()))
    edges = G.get_edgelist()
    n = len(nodes)
    m = len(edges)

    deg = G.degree()

    # incidence matrix E (n x m)
    E_mat = np.zeros((n, m), dtype=float)
    for j, (u, v) in enumerate(edges):
        E_mat[u, j] = 1.0
        E_mat[v, j] = 1.0

    # D^{-1}
    Dinv = np.zeros((n, n), dtype=float)
    for i, d in enumerate(deg):
        Dinv[i, i] = 1.0 / d if d > 0 else 0.0

    # x_j = 1 / sqrt(deg(u)+deg(v))
    x = np.zeros(m, dtype=float)
    for j, (u, v) in enumerate(edges):
        x[j] = 1.0 / np.sqrt(deg[u] + deg[v])

    Gmat = E_mat.T @ Dinv @ E_mat  # m x m
    M = np.eye(m) - (alpha / 2.0) * Gmat
    b = (1.0 - alpha) * x

    z = np.linalg.solve(M, b)

    ec = {}
    for j, (u, v) in enumerate(edges):
        key = (u, v) if u < v else (v, u)
        ec[key] = float(z[j])

    return normalize_edge_centrality(ec, mode=norm_mode)


# ============================================================
# 3. Graph generation (ER & BA) in igraph
# ============================================================
import random
import numpy as np
import igraph as ig

def generate_er_graph_ig(n=1000, p=0.01, seed=None):
    """
    Generate reproducible ER graph by controlling both Python
    and NumPy RNG before igraph creates edges.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G = ig.Graph.Erdos_Renyi(n=n, p=p, directed=False, loops=False)

    if not G.is_connected():
        G = G.clusters().giant().copy()

    return G


def generate_ba_graph_ig(n=1000, m=2, seed=None):
    """
    Generate reproducible BA graph with consistent RNG state.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G = ig.Graph.Barabasi(n=n, m=m, directed=False)

    return G



# ============================================================
# 4. Sampling node pairs
# ============================================================

def sample_node_pairs_ig(G, num_pairs=100, min_dist=2):
    """
    Sample up to num_pairs unordered node pairs (s,t) with dist(s,t)>=min_dist.
    """
    vertices = list(range(G.vcount()))
    pairs = []
    tried = set()
    max_attempts = len(vertices) * len(vertices)

    attempts = 0
    while len(pairs) < num_pairs and attempts < max_attempts:
        s = random.choice(vertices)
        t = random.choice(vertices)
        attempts += 1
        if s == t:
            continue
        key = (s, t) if s < t else (t, s)
        if key in tried:
            continue
        tried.add(key)
        dist_list = G.shortest_paths(source=s)[0]
        d = dist_list[t]
        if d == inf or d < min_dist:
            continue
        pairs.append((s, t))

    return pairs


# ============================================================
# 5. Robustness metrics
# ============================================================

def global_efficiency_igraph(G):
    """
    Global efficiency:
      E = (1 / (n*(n-1))) * sum_{i!=j} 1 / d(i,j)
    where d(i,j) is the shortest-path distance.
    """
    n = G.vcount()
    if n <= 1:
        return 0.0

    dist = G.distances()  # all-pairs distances
    s = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dist[i][j]
            if math.isfinite(d) and d > 0:
                s += 1.0 / d

    return s / (n * (n - 1))


def giant_component_fraction(G):
    """
    Size of the largest connected component / n.
    """
    comps = G.connected_components()
    if len(comps) == 0:
        return 0.0
    giant = comps.giant()
    return giant.vcount() / G.vcount()


# ============================================================
# 6. ECSP + path-usage counting on a single graph
# ============================================================

CENTRALITIES = ["EBC", "ECL", "GRAV", "ECHO"]
ATTACK_TYPES = ["PLAIN", "EBC", "ECL", "GRAV", "ECHO"]
REMOVAL_FRACTIONS = np.linspace(0.01, 0.10, 10)  # 1% .. 10%

def add_path_edges_to_counts(edge_counts, attack_type, path):
    """
    Update edge_counts[attack_type][(u,v)] by +1 for each edge in the path.
    """
    if path is None or len(path) < 2:
        return
    d = edge_counts[attack_type]
    for u, v in zip(path, path[1:]):
        e = tuple(sorted((int(u), int(v))))
        d[e] += 1


def run_ecsp_experiment_on_graph_ig(G, graph_id="G1", graph_type="ER", num_pairs=50):
    """
    Run ECSP experiments on a single igraph.Graph, and simultaneously
    build path-usage counts for robustness attacks.

    Returns
    -------
    results : list of dict
        ECSP per (s,t) pair.
    edge_counts_by_attack : dict
        attack_type -> dict[(u,v)] -> frequency
    """
    # 1) Centralities
    ebc  = compute_edge_betweenness_ig(G)
    ecl  = compute_edge_closeness_ig(G)
    grav = compute_edge_gravity_ig(G, num_sources=100, num_targets=100,
                                   random_seed=0)
    echo = compute_echo_centrality_ig(G, alpha=0.5)

    centralities = {
        "EBC": ebc,
        "ECL": ecl,
        "GRAV": grav,
        "ECHO": echo,
    }

    # 2) Sample pairs
    pairs = sample_node_pairs_ig(G, num_pairs=num_pairs, min_dist=2)
    results = []

    # 3) Initialize path-usage counts
    edge_counts_by_attack = {atk: defaultdict(int) for atk in ATTACK_TYPES}

    # 4) For each pair: shortest path + ECSP for each centrality
    for (s, t) in pairs:
        dist_st = G.shortest_paths(source=s)[0][t]
        # shortest path for reference
        sp_v = G.get_shortest_paths(s, to=t)[0]  # vertex indices
        plain_path = sp_v

        # update PLAIN usage
        add_path_edges_to_counts(edge_counts_by_attack, "PLAIN", plain_path)

        row = {
            "graph_id": graph_id,
            "graph_type": graph_type,
            "s": s,
            "t": t,
            "dist": dist_st,
            "plain_path": plain_path,
        }

        for name, C in centralities.items():
            path, score = edge_central_shortest_path_ig(G, s, t, C)
            row[f"path_{name}"] = path
            row[f"score_{name}"] = score
            if path is not None and len(path) > 1:
                row[f"mean_{name}"] = score / (len(path) - 1)
            else:
                row[f"mean_{name}"] = None

            # update usage counts for this ECSP
            add_path_edges_to_counts(edge_counts_by_attack, name, path)

        results.append(row)

    return results, edge_counts_by_attack


# ============================================================
# 7. Attack experiment on the SAME graph
# ============================================================

def run_attack_experiment_on_graph_ig(G, graph_id, graph_type, edge_counts_by_attack):
    """
    Given a graph G and the edge usage counts induced by different
    routing rules (PLAIN/EBC/ECL/GRAV/ECHO), perform robustness experiments
    by removing top-frequent edges for each attack type.

    Returns
    -------
    robust_rows : list of dict
    """
    robust_rows = []

    n0 = G.vcount()
    m0 = G.ecount()

    # Precompute edge key -> edge id mapping for this graph
    ekeys = [tuple(sorted(e)) for e in G.get_edgelist()]
    edge_to_id = {e: i for i, e in enumerate(ekeys)}

    # Baseline (no removal) metrics, for reference
    base_eff = global_efficiency_igraph(G)
    base_gcc = giant_component_fraction(G)
    print(f"{graph_id}: base_eff={base_eff:.4f}, base_gcc={base_gcc:.4f}")

    for atk in ATTACK_TYPES:
        counts = edge_counts_by_attack.get(atk, {})
        if not counts:
            # No paths for this attack type (should not happen), skip
            continue

        # Sort edges by frequency descending
        sorted_edges = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        sorted_edge_list = [e for (e, freq) in sorted_edges]

        for frac in REMOVAL_FRACTIONS:
            # Number of edges to remove
            k = int(round(frac * m0))
            if k <= 0:
                continue
            # Limit by number of available attack edges
            k = min(k, len(sorted_edge_list))
            edges_to_remove = sorted_edge_list[:k]
            if not edges_to_remove:
                continue

            # Copy the graph and delete edges
            G_attack = G.copy()
            # Map (u,v) -> edge ids (ignore edges that do not exist, just in case)
            eids = [edge_to_id[e] for e in edges_to_remove if e in edge_to_id]
            if not eids:
                continue

            G_attack.delete_edges(eids)

            eff = global_efficiency_igraph(G_attack)
            gcc = giant_component_fraction(G_attack)

            robust_rows.append({
                "graph_type": graph_type,
                "graph_id": graph_id,
                "attack_type": atk,
                "frac_removed": frac,
                "num_removed_edges": len(eids),
                "total_edges": m0,
                "efficiency": eff,
                "giant_comp_frac": gcc,
                "base_eff": base_eff,
                "base_gcc": base_gcc,
            })

    return robust_rows


# ============================================================
# 8. Family-level experiment runners (ER + BA) with robustness
# ============================================================

def run_er_experiments_ig(
    num_graphs=3, n=1000, p=0.01, num_pairs=100, base_seed=0
):
    all_ecsp_results = []
    all_robust_results = []
    for i in range(num_graphs):
        seed = base_seed + i
        random.seed(seed)
        G = generate_er_graph_ig(n=n, p=p, seed=seed)
        graph_id = f"ER_n{n}_p{p}_seed{seed}_large_#{i}"
        print(f"[ER] Graph {i+1}/{num_graphs}: "
              f"{G.vcount()} nodes, {G.ecount()} edges.")

        ecsp_results, edge_counts = run_ecsp_experiment_on_graph_ig(
            G, graph_id=graph_id, graph_type="ER", num_pairs=num_pairs
        )
        robust_results = run_attack_experiment_on_graph_ig(
            G, graph_id=graph_id, graph_type="ER", edge_counts_by_attack=edge_counts
        )

        all_ecsp_results.extend(ecsp_results)
        all_robust_results.extend(robust_results)

    return all_ecsp_results, all_robust_results


def run_ba_experiments_ig(
    num_graphs=3, n=1000, m=2, num_pairs=100, base_seed=1000
):
    all_ecsp_results = []
    all_robust_results = []
    for i in range(num_graphs):
        seed = base_seed + i
        random.seed(seed)
        G = generate_ba_graph_ig(n=n, m=m, seed=seed)
        graph_id = f"BA_n{n}_m{m}_seed{seed}_large_#{i}"
        print(f"[BA] Graph {i+1}/{num_graphs}: "
              f"{G.vcount()} nodes, {G.ecount()} edges.")

        ecsp_results, edge_counts = run_ecsp_experiment_on_graph_ig(
            G, graph_id=graph_id, graph_type="BA", num_pairs=num_pairs
        )
        robust_results = run_attack_experiment_on_graph_ig(
            G, graph_id=graph_id, graph_type="BA", edge_counts_by_attack=edge_counts
        )

        all_ecsp_results.extend(ecsp_results)
        all_robust_results.extend(robust_results)

    return all_ecsp_results, all_robust_results


def run_all_experiments_ig(
    er_num_graphs=3,
    er_n=1000,
    er_p=0.01,
    ba_num_graphs=3,
    ba_n=1000,
    ba_m=2,
    num_pairs=100,
    er_base_seed=0,
    ba_base_seed=1000,
    save_ecsp_csv_path="ecsp_er_ba_results_larger.csv",
    save_robust_csv_path="ecsp_robustness_results_larger.csv",
):
    print("=== Running ER experiments (large) ===")
    er_ecsp, er_robust = run_er_experiments_ig(
        num_graphs=er_num_graphs,
        n=er_n,
        p=er_p,
        num_pairs=num_pairs,
        base_seed=er_base_seed
    )

    print("=== Running BA experiments (large) ===")
    ba_ecsp, ba_robust = run_ba_experiments_ig(
        num_graphs=ba_num_graphs,
        n=ba_n,
        m=ba_m,
        num_pairs=num_pairs,
        base_seed=ba_base_seed
    )

    all_ecsp = er_ecsp + ba_ecsp
    all_robust = er_robust + ba_robust

    print(f"Total ECSP rows collected: {len(all_ecsp)}")
    print(f"Total robustness rows collected: {len(all_robust)}")

    df_ecsp = pd.DataFrame(all_ecsp)
    df_robust = pd.DataFrame(all_robust)

    if save_ecsp_csv_path is not None:
        df_ecsp.to_csv(save_ecsp_csv_path, index=False)
        print(f"Saved ECSP results to {save_ecsp_csv_path}")

    if save_robust_csv_path is not None:
        df_robust.to_csv(save_robust_csv_path, index=False)
        print(f"Saved robustness results to {save_robust_csv_path}")

    return df_ecsp, df_robust


# ============================================================
# 9. ECSP agreement analysis & plots
# ============================================================

def add_agreement_columns(df):
    df['agree_EBC_ECL']   = (df['path_EBC'] == df['path_ECL']).astype(int)
    df['agree_EBC_GRAV']  = (df['path_EBC'] == df['path_GRAV']).astype(int)
    df['agree_EBC_ECHO']  = (df['path_EBC'] == df['path_ECHO']).astype(int)
    df['agree_ECL_GRAV']  = (df['path_ECL'] == df['path_GRAV']).astype(int)
    df['agree_ECL_ECHO']  = (df['path_ECL'] == df['path_ECHO']).astype(int)
    df['agree_GRAV_ECHO'] = (df['path_GRAV'] == df['path_ECHO']).astype(int)

    df['all_four_agree'] = (
        (df['path_EBC'] == df['path_ECL']) &
        (df['path_EBC'] == df['path_GRAV']) &
        (df['path_EBC'] == df['path_ECHO'])
    ).astype(int)

    return df


def summarize_agreement(df):
    group = df.groupby('graph_type')
    agg = group[[
        'agree_EBC_ECL',
        'agree_EBC_GRAV',
        'agree_EBC_ECHO',
        'agree_ECL_GRAV',
        'agree_ECL_ECHO',
        'agree_GRAV_ECHO',
        'all_four_agree'
    ]].mean()
    print("Agreement rates by graph type (large graphs):")
    print(agg)
    return agg


def plot_agreement_bars(agg, save_path=None):
    plt.figure(figsize=(12, 6))
    columns = agg.columns.tolist()
    x = range(len(columns))

    er_vals = agg.loc["ER"].tolist()
    ba_vals = agg.loc["BA"].tolist()

    width = 0.35
    plt.bar([i - width/2 for i in x], er_vals, width, label="ER")
    plt.bar([i + width/2 for i in x], ba_vals, width, label="BA")

    plt.xticks(x, columns, rotation=45, ha="right")
    plt.ylabel("Agreement Rate")
    plt.ylim(0, 1)
    plt.title("Path Agreement Rates Between Centralities (ER vs BA, large)")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_score_difference(df, centrality1, centrality2, bins=30, save_path=None):
    col1 = f"score_{centrality1}"
    col2 = f"score_{centrality2}"
    diff = df[col1] - df[col2]

    plt.figure(figsize=(10, 5))
    plt.hist(diff, bins=bins, alpha=0.7)
    plt.axvline(0, color="red", linestyle="--")

    plt.title(f"Score Difference Histogram (large): {centrality1} - {centrality2}")
    plt.xlabel(f"{centrality1} score - {centrality2} score")
    plt.ylabel("Count")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    plt.show()


PAIR_TO_COL = {
    ("EBC", "ECL"):  "agree_EBC_ECL",
    ("ECL", "EBC"):  "agree_EBC_ECL",

    ("EBC", "GRAV"): "agree_EBC_GRAV",
    ("GRAV", "EBC"): "agree_EBC_GRAV",

    ("EBC", "ECHO"): "agree_EBC_ECHO",
    ("ECHO", "EBC"): "agree_EBC_ECHO",

    ("ECL", "GRAV"): "agree_ECL_GRAV",
    ("GRAV", "ECL"): "agree_ECL_GRAV",

    ("ECL", "ECHO"): "agree_ECL_ECHO",
    ("ECHO", "ECL"): "agree_ECL_ECHO",

    ("GRAV", "ECHO"): "agree_GRAV_ECHO",
    ("ECHO", "GRAV"): "agree_GRAV_ECHO",
}


def build_agreement_matrix(agg, graph_type):
    mat = np.ones((4, 4), dtype=float)
    def get(c1, c2):
        if c1 == c2:
            return 1.0
        key = (c1, c2)
        col_name = PAIR_TO_COL[key]
        return agg.loc[graph_type, col_name]
    for i, c1 in enumerate(CENTRALITIES):
        for j, c2 in enumerate(CENTRALITIES):
            mat[i, j] = get(c1, c2)
    return mat


def plot_agreement_heatmap(agg, graph_type, save_path=None):
    mat = build_agreement_matrix(agg, graph_type)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(mat, vmin=0, vmax=1)
    plt.colorbar(im, label="Agreement rate")
    plt.xticks(range(4), CENTRALITIES)
    plt.yticks(range(4), CENTRALITIES)
    plt.title(f"ECSP Path Agreement Matrix ({graph_type}, large)")

    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{mat[i, j]:.2f}",
                     ha="center", va="center", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved heatmap to {save_path}")
    plt.show()


def build_score_diff_matrix(df, graph_type=None):
    if graph_type is not None:
        df = df[df["graph_type"] == graph_type]
    mat = np.zeros((4, 4), dtype=float)
    for i, c1 in enumerate(CENTRALITIES):
        for j, c2 in enumerate(CENTRALITIES):
            col1 = f"score_{c1}"
            col2 = f"score_{c2}"
            diff = df[col1] - df[col2]
            mat[i, j] = diff.mean()
    return mat


def plot_score_diff_heatmap(df, graph_type=None, save_path=None):
    mat = build_score_diff_matrix(df, graph_type)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(mat)
    plt.colorbar(im, label="Mean(score_i - score_j)")
    plt.xticks(range(4), CENTRALITIES)
    plt.yticks(range(4), CENTRALITIES)
    if graph_type is None:
        title = "Mean ECSP Score Differences (all, large)"
    else:
        title = f"Mean ECSP Score Differences ({graph_type}, large)"
    plt.title(title)
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{mat[i, j]:.2f}",
                     ha="center", va="center", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved heatmap to {save_path}")
    plt.show()


# ============================================================
# 10. Robustness aggregation + plots
# ============================================================

ATTACK_COLORS = {
    "PLAIN": "black",
    "EBC":   "tab:blue",
    "ECL":   "tab:green",
    "GRAV":  "tab:orange",
    "ECHO":  "tab:red",
}

def plot_robustness_curves(robust_df,
                           efficiency_prefix="robust_efficiency",
                           gcc_prefix="robust_gcc"):
    # Average over graphs of same type
    agg = (robust_df
           .groupby(["graph_type", "attack_type", "frac_removed"])
           [["efficiency", "giant_comp_frac"]]
           .mean()
           .reset_index())
    print(agg.head())

    # Efficiency curves
    for gtype in ["ER", "BA"]:
        sub = agg[agg["graph_type"] == gtype]

        plt.figure(figsize=(8, 5))
        for atk in ATTACK_TYPES:
            sub_atk = sub[sub["attack_type"] == atk]
            if sub_atk.empty:
                continue
            xs = sub_atk["frac_removed"]
            ys = sub_atk["efficiency"]
            plt.plot(xs, ys, marker="o", label=atk,
                     color=ATTACK_COLORS.get(atk, None))

        plt.xlabel("Fraction of edges removed")
        plt.ylabel("Global efficiency")
        plt.title(f"Robustness: global efficiency vs removal fraction ({gtype}, large graphs)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = f"{efficiency_prefix}_{gtype}_larger.png"
        plt.savefig(fname, dpi=300)
        print("Saved", fname)
        plt.show()

    # Giant component curves
    for gtype in ["ER", "BA"]:
        sub = agg[agg["graph_type"] == gtype]

        plt.figure(figsize=(8, 5))
        for atk in ATTACK_TYPES:
            sub_atk = sub[sub["attack_type"] == atk]
            if sub_atk.empty:
                continue
            xs = sub_atk["frac_removed"]
            ys = sub_atk["giant_comp_frac"]
            plt.plot(xs, ys, marker="o", label=atk,
                     color=ATTACK_COLORS.get(atk, None))

        plt.xlabel("Fraction of edges removed")
        plt.ylabel("Giant component size / n")
        plt.title(f"Robustness: giant component vs removal fraction ({gtype}, large graphs)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = f"{gcc_prefix}_{gtype}_larger.png"
        plt.savefig(fname, dpi=300)
        print("Saved", fname)
        plt.show()


# ============================================================
# 11. Main
# ============================================================

if __name__ == "__main__":
    # 1) Run experiments on large graphs (you can change n=500 if you want)
    df_ecsp, df_robust = run_all_experiments_ig(
        er_num_graphs=3,
        er_n=1000,
        er_p=0.01,
        ba_num_graphs=3,
        ba_n=1000,
        ba_m=2,
        num_pairs=100,
        er_base_seed=0,
        ba_base_seed=1000,
        save_ecsp_csv_path="ecsp_er_ba_results_larger.csv",
        save_robust_csv_path="ecsp_robustness_results_larger.csv"
    )

    # 2) ECSP agreement analysis & plots
    df_ecsp = add_agreement_columns(df_ecsp)
    agg_agree = summarize_agreement(df_ecsp)

    plot_agreement_bars(agg_agree, save_path="agreement_rates_er_vs_ba_larger.png")

    plot_score_difference(df_ecsp, "EBC",  "ECHO", save_path="diff_EBC_ECHO_larger.png")
    plot_score_difference(df_ecsp, "ECL",  "EBC",  save_path="diff_ECL_EBC_larger.png")
    plot_score_difference(df_ecsp, "ECL",  "GRAV", save_path="diff_ECL_GRAV_larger.png")
    plot_score_difference(df_ecsp, "ECL",  "ECHO", save_path="diff_ECL_ECHO_larger.png")
    plot_score_difference(df_ecsp, "GRAV", "EBC",  save_path="diff_GRAV_EBC_larger.png")
    plot_score_difference(df_ecsp, "GRAV", "ECHO", save_path="diff_GRAV_ECHO_larger.png")

    plot_agreement_heatmap(agg_agree, "ER", save_path="agreement_matrix_ER_larger.png")
    plot_agreement_heatmap(agg_agree, "BA", save_path="agreement_matrix_BA_larger.png")

    plot_score_diff_heatmap(df_ecsp, graph_type=None,
                            save_path="score_diff_all_larger.png")
    plot_score_diff_heatmap(df_ecsp, graph_type="ER",
                            save_path="score_diff_ER_larger.png")
    plot_score_diff_heatmap(df_ecsp, graph_type="BA",
                            save_path="score_diff_BA_larger.png")

    # 3) Robustness plots
    plot_robustness_curves(df_robust,
                           efficiency_prefix="robust_efficiency",
                           gcc_prefix="robust_gcc")





import os

# ============================================================
# 12. Real dataset loaders (SNAP-style edge lists) in igraph
# ============================================================
def load_real_graph_ig(path, directed=False, make_undirected=True,
                       largest_cc=True, simplify_graph=True):
    edges = []
    path = os.path.expanduser(path)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            v = int(parts[1])
            edges.append((u, v))

    if not edges:
        raise ValueError(f"No edges read from file: {path}")

    # Build igraph
    G = ig.Graph(edges=edges, directed=directed)

    # Optionally symmetrize (in-place on your igraph)
    if directed and make_undirected:
        G.to_undirected()   # <- NO assignment

    # Simplify: remove multi-edges and self-loops (also in-place)
    if simplify_graph:
        G.simplify(multiple=True, loops=True)

    # Largest connected component
    if largest_cc:
        if not G.is_connected():
            comps = G.connected_components()
            G = comps.giant().copy()
        else:
            G = G.copy()

    return G



def load_ca_grqc_ig(path="CA-GrQc.txt"):
    """
    CA-GrQc collaboration network (SNAP).
    Treated as undirected.
    """
    return load_real_graph_ig(
        path,
        directed=False,
        make_undirected=False,
        largest_cc=True,
        simplify_graph=True,
    )


def load_powergrid_ig(path="powergrid.txt"):
    """
    Power Grid network.
    Already undirected.
    """
    return load_real_graph_ig(
        path,
        directed=False,
        make_undirected=False,
        largest_cc=True,
        simplify_graph=True,
    )


def load_email_eu_core_ig(path="email-Eu-core.txt"):
    """
    Email-Eu-core network.
    Original is directed; we symmetrize to get an undirected graph.
    """
    return load_real_graph_ig(
        path,
        directed=True,        # read as directed
        make_undirected=True, # then symmetrize
        largest_cc=True,
        simplify_graph=True,
    )




# ============================================================
# 13. Real dataset experiment runner (ECSP + robustness)
# ============================================================

def run_real_dataset_experiment_ig(
    dataset_name,
    path,
    loader_fn,
    num_pairs=100,
    save_ecsp_csv_path=None,
    save_robust_csv_path=None,
    seed=123,
):
    """
    Run ECSP + path-based robustness on a single real dataset.

    Parameters
    ----------
    dataset_name : str
        Short code for the dataset, e.g. "CA", "EM", "PG".
        This will go into df['graph_type'].
    path : str
        Path to the edge list file.
    loader_fn : callable
        Function that loads the graph, e.g. load_ca_grqc_ig.
    num_pairs : int
        Number of (s,t) pairs for ECSP.
    save_ecsp_csv_path : str or None
        Where to save ECSP results CSV (or None to skip saving).
    save_robust_csv_path : str or None
        Where to save robustness results CSV (or None to skip saving).

    Returns
    -------
    df_ecsp : pd.DataFrame
    df_robust : pd.DataFrame

    """
 # ---- FIXED SEED FOR REPRODUCIBILITY ----
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # ----------------------------------------

    # 1) Load graph
    G = loader_fn(path)
    n = G.vcount()
    m = G.ecount()
    graph_id = f"{dataset_name}_n{n}_m{m}"

    print(f"[{dataset_name}] Loaded graph {graph_id}: {n} nodes, {m} edges.")

    # 2) ECSP + path-usage
    ecsp_results, edge_counts = run_ecsp_experiment_on_graph_ig(
        G, graph_id=graph_id, graph_type=dataset_name, num_pairs=num_pairs
    )

    # 3) Robustness (same G, same edge_counts)
    robust_rows = run_attack_experiment_on_graph_ig(
        G, graph_id=graph_id, graph_type=dataset_name,
        edge_counts_by_attack=edge_counts
    )

    df_ecsp = pd.DataFrame(ecsp_results)
    df_robust = pd.DataFrame(robust_rows)

    # 4) Save if requested
    if save_ecsp_csv_path is not None:
        df_ecsp.to_csv(save_ecsp_csv_path, index=False)
        print(f"[{dataset_name}] Saved ECSP results to {save_ecsp_csv_path}")

    if save_robust_csv_path is not None:
        df_robust.to_csv(save_robust_csv_path, index=False)
        print(f"[{dataset_name}] Saved robustness results to {save_robust_csv_path}")

    return df_ecsp, df_robust



# ============================================================
# 14. Robustness plots for a single real graph_type
# ============================================================

def plot_robustness_curves_single(
    robust_df,
    dataset_name,
    efficiency_prefix="robust_efficiency_real",
    gcc_prefix="robust_gcc_real",
):
    """
    Plot robustness curves (efficiency and GCC fraction) for a single
    real dataset identified by graph_type == dataset_name.
    """
    # Average over graphs with same graph_type (here we likely have only one)
    agg = (robust_df[robust_df["graph_type"] == dataset_name]
           .groupby(["graph_type", "attack_type", "frac_removed"])
           [["efficiency", "giant_comp_frac"]]
           .mean()
           .reset_index())

    print(f"[{dataset_name}] Robustness summary:")
    print(agg.head())

    # Efficiency curves
    plt.figure(figsize=(8, 5))
    for atk in ATTACK_TYPES:
        sub_atk = agg[agg["attack_type"] == atk]
        if sub_atk.empty:
            continue
        xs = sub_atk["frac_removed"]
        ys = sub_atk["efficiency"]
        plt.plot(xs, ys, marker="o", label=atk,
                 color=ATTACK_COLORS.get(atk, None))

    plt.xlabel("Fraction of edges removed")
    plt.ylabel("Global efficiency")
    plt.title(f"{dataset_name}: global efficiency vs removal fraction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fname_eff = f"{efficiency_prefix}_{dataset_name}.png"
    plt.savefig(fname_eff, dpi=300)
    print(f"[{dataset_name}] Saved", fname_eff)
    plt.show()

    # Giant component curves
    plt.figure(figsize=(8, 5))
    for atk in ATTACK_TYPES:
        sub_atk = agg[agg["attack_type"] == atk]
        if sub_atk.empty:
            continue
        xs = sub_atk["frac_removed"]
        ys = sub_atk["giant_comp_frac"]
        plt.plot(xs, ys, marker="o", label=atk,
                 color=ATTACK_COLORS.get(atk, None))

    plt.xlabel("Fraction of edges removed")
    plt.ylabel("Giant component size / n")
    plt.title(f"{dataset_name}: giant component vs removal fraction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fname_gcc = f"{gcc_prefix}_{dataset_name}.png"
    plt.savefig(fname_gcc, dpi=300)
    print(f"[{dataset_name}] Saved", fname_gcc)
    plt.show()




# ============================================================
# 15. Agreement analysis for real datasets (CA, EM, PG)
# ============================================================

REAL_DATASETS = [
    ("CA", "ecsp_CA-GrQc_results.csv"),
    ("EM", "ecsp_email-Eu-core_results.csv"),
    ("PG", "ecsp_powergrid_results.csv"),
]

def load_real_ecsp_results(dataset_name, path):
    """
    Load ECSP results for a single real dataset.
    Ensures graph_type is set to dataset_name.
    """
    df = pd.read_csv(path)

    # If graph_type not present or inconsistent, overwrite
    df["graph_type"] = dataset_name
    return df


def summarize_agreement_single(df, dataset_name):
    """
    Compute mean agreement rates for one dataset
    using existing add_agreement_columns logic.
    Returns a 1-row DataFrame ready for heatmap / printing.
    """
    df = add_agreement_columns(df)

    cols = [
        'agree_EBC_ECL',
        'agree_EBC_GRAV',
        'agree_EBC_ECHO',
        'agree_ECL_GRAV',
        'agree_ECL_ECHO',
        'agree_GRAV_ECHO',
        'all_four_agree',
    ]
    means = df[cols].mean().to_frame().T
    means.index = [dataset_name]
    return means, df


def run_agreement_analysis_real():
    """
    For each real dataset (CA, EM, PG):
      - load ECSP results
      - compute agreement statistics
      - plot a 4x4 heatmap for that dataset

    Also returns a combined summary table.
    """
    all_aggs = []

    for dataset_name, path in REAL_DATASETS:
        print(f"=== Agreement analysis for {dataset_name} ===")
        df = load_real_ecsp_results(dataset_name, path)
        agg_row, df_with_agree = summarize_agreement_single(df, dataset_name)
        all_aggs.append(agg_row)

        # Use the same heatmap function as ER/BA
        # It expects an 'agg' DataFrame with a row for this graph_type
        agg_for_heatmap = agg_row.copy()
        agg_for_heatmap.index = [dataset_name]

        plot_agreement_heatmap(
            agg_for_heatmap,
            graph_type=dataset_name,
            save_path=f"agreement_matrix_{dataset_name}.png"
        )

    agg_all = pd.concat(all_aggs, axis=0)
    print("Agreement rates for real datasets (CA, EM, PG):")
    print(agg_all)
    agg_all.to_csv("agreement_real_datasets_summary.csv")
    print("Saved agreement summary to agreement_real_datasets_summary.csv")
    return agg_all



# ============================================================
# 16. CENTRALITY SCORE CORRELATION MATRICES
# ============================================================

def compute_edge_centralities_df(
    G,
    num_sources=100,
    num_targets=100,
    alpha_echo=0.5,
    random_seed_grav=0,
):
    """
    For a given igraph Graph G, compute edge-level centralities and
    return a DataFrame with one row per edge and columns:
      ['u', 'v', 'EBC', 'ECL', 'GRAV', 'ECHO'].
    """
    # Compute centralities using your existing functions
    ebc = compute_edge_betweenness_ig(G, norm_mode="max")
    ecl = compute_edge_closeness_ig(G, norm_mode="max")
    grav = compute_edge_gravity_ig(
        G,
        num_sources=num_sources,
        num_targets=num_targets,
        norm_mode="max",
        random_seed=random_seed_grav,
    )
    echo = compute_echo_centrality_ig(G, alpha=alpha_echo, norm_mode="max")

    rows = []
    for (u, v) in G.get_edgelist():
        key = (u, v) if u < v else (v, u)
        rows.append({
            "u": u,
            "v": v,
            "EBC": ebc.get(key, 0.0),
            "ECL": ecl.get(key, 0.0),
            "GRAV": grav.get(key, 0.0),
            "ECHO": echo.get(key, 0.0),
        })
    return pd.DataFrame(rows)


def centrality_correlation_for_graph(G, graph_name, save_prefix=None):
    """
    Compute Pearson correlation matrix of edge centralities on G,
    show a heatmap, and optionally save CSV/PNG.
    """
    df_cent = compute_edge_centralities_df(G)
    corr = df_cent[["EBC", "ECL", "GRAV", "ECHO"]].corr(method="pearson")

    print(f"Centrality score correlations for {graph_name}:")
    print(corr)

    if save_prefix is not None:
        corr.to_csv(f"{save_prefix}_centrality_corr.csv")
        print(f"Saved correlation matrix to {save_prefix}_centrality_corr.csv")

    # Heatmap
    plt.figure(figsize=(5, 4))
    im = plt.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, label="Pearson correlation")
    labels = ["EBC", "ECL", "GRAV", "ECHO"]
    plt.xticks(range(4), labels)
    plt.yticks(range(4), labels)
    plt.title(f"Edge centrality score correlations ({graph_name})")

    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{corr.values[i, j]:.2f}",
                     ha="center", va="center", fontsize=8, color="black")
    plt.tight_layout()
    if save_prefix is not None:
        plt.savefig(f"{save_prefix}_centrality_corr.png", dpi=300)
        print(f"Saved correlation heatmap to {save_prefix}_centrality_corr.png")
    plt.show()

    return corr


# Convenience wrappers: build graphs and compute correlations
def run_correlation_all_networks():
    """
    Compute centrality score correlations on:
      - ER (first synthetic graph)
      - BA (first synthetic graph)
      - CA, EM, PG (real datasets)
    Adjust seeds/paths to match your setup.
    """
    # ---- Synthetic ER / BA (use same generators & seeds as before) ----
    er_G = generate_er_graph_ig(n=1000, p=0.01, seed=0)
    ba_G = generate_ba_graph_ig(n=1000, m=2, seed=1000)

    centrality_correlation_for_graph(er_G, "ER", save_prefix="ER")
    centrality_correlation_for_graph(ba_G, "BA", save_prefix="BA")

    # ---- Real datasets (explicit paths here) ----
    ca_G = load_ca_grqc_ig(
        r"C:\Users\Kusal Thapa\Desktop\PhD\doc\matsypura\article\dataset\CA-GrQc.txt"
    )

    em_G = load_email_eu_core_ig(
        r"C:\Users\Kusal Thapa\Desktop\PhD\doc\matsypura\article\dataset\email-Eu-core.txt"
    )

    pg_G = load_powergrid_ig(
        r"C:\Users\Kusal Thapa\Desktop\PhD\doc\matsypura\article\dataset\powergrid.txt"
    )

    centrality_correlation_for_graph(ca_G, "CA", save_prefix="CA")
    centrality_correlation_for_graph(em_G, "EM", save_prefix="EM")
    centrality_correlation_for_graph(pg_G, "PG", save_prefix="PG")

# ============================================================
# 17. PATH OVERLAP METRICS (Jaccard edge overlap)
# ============================================================

import ast
import math

CENTRALITIES = ["EBC", "ECL", "GRAV", "ECHO"]

def parse_path_general(s):
    """
    Safely parse path column which may be a string representation of a list.
    """
    if isinstance(s, float) and math.isnan(s):
        return []
    if isinstance(s, list):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def path_to_edge_set(path):
    """
    Convert a vertex-path [v0, v1, ..., vk] into a set of undirected edges.
    """
    if path is None or len(path) < 2:
        return set()
    edges = set()
    for u, v in zip(path, path[1:]):
        u = int(u)
        v = int(v)
        e = (u, v) if u < v else (v, u)
        edges.add(e)
    return edges


# ------------------------------------------------------------
# 17a. Overlap: plain shortest path vs each centrality ECSP
# ------------------------------------------------------------

def compute_path_overlap_for_dataset(df, dataset_name):
    """
    Given an ECSP results DataFrame df for one dataset,
    compute mean Jaccard edge-overlap between:
      - plain shortest path vs each centrality ECSP path.
    Returns a one-row DataFrame with columns:
      ['J_plain_EBC', 'J_plain_ECL', 'J_plain_GRAV', 'J_plain_ECHO'].
    """
    # Ensure paths are lists
    PATH_COLS = ["plain_path"] + [f"path_{c}" for c in CENTRALITIES]
    for col in PATH_COLS:
        if col in df.columns:
            df[col] = df[col].apply(parse_path_general)

    overlaps = {f"J_plain_{c}": [] for c in CENTRALITIES}

    for _, row in df.iterrows():
        base_edges = path_to_edge_set(row["plain_path"])
        if not base_edges:
            continue
        for c in CENTRALITIES:
            col = f"path_{c}"
            if col not in df.columns:
                continue
            c_edges = path_to_edge_set(row[col])
            if not c_edges:
                overlaps[f"J_plain_{c}"].append(0.0)
            else:
                inter = len(base_edges & c_edges)
                union = len(base_edges | c_edges)
                jacc = inter / union if union > 0 else 0.0
                overlaps[f"J_plain_{c}"].append(jacc)

    mean_vals = {k: (sum(v) / len(v) if v else 0.0)
                 for k, v in overlaps.items()}
    res_df = pd.DataFrame([mean_vals], index=[dataset_name])

    print(f"Mean Jaccard overlap with plain shortest path for {dataset_name}:")
    print(res_df)

    res_df.to_csv(f"path_overlap_{dataset_name}.csv")
    print(f"Saved path overlap summary to path_overlap_{dataset_name}.csv")

    return res_df


def run_path_overlap_all():
    """
    Compute path overlap metrics for:
      - ER and BA (from ecsp_er_ba_results_larger.csv)
      - CA, EM, PG (their own ECSP CSVs)
    Adjust file names if yours differ.
    """
    # Synthetic combined file
    df_syn = pd.read_csv("ecsp_er_ba_results_larger.csv")
    if "graph_type" not in df_syn.columns:
        df_syn["graph_type"] = df_syn["graph_id"].str.slice(0, 2)

    res_list = []

    for gtype in ["ER", "BA"]:
        df_g = df_syn[df_syn["graph_type"] == gtype].copy()
        res = compute_path_overlap_for_dataset(df_g, gtype)
        res_list.append(res)

    # Real datasets
    df_ca = pd.read_csv("ecsp_CA-GrQc_results.csv")
    df_em = pd.read_csv("ecsp_email-Eu-core_results.csv")
    df_pg = pd.read_csv("ecsp_powergrid_results.csv")

    res_list.append(compute_path_overlap_for_dataset(df_ca, "CA"))
    res_list.append(compute_path_overlap_for_dataset(df_em, "EM"))
    res_list.append(compute_path_overlap_for_dataset(df_pg, "PG"))

    all_res = pd.concat(res_list, axis=0)
    all_res.to_csv("path_overlap_all_datasets.csv")
    print("Saved combined path overlap summary to path_overlap_all_datasets.csv")
    return all_res


# ------------------------------------------------------------
# 17b. Pairwise overlap: centrality vs centrality ECSP paths
# ------------------------------------------------------------

def compute_pairwise_overlap_for_dataset(df, dataset_name):
    """
    Given an ECSP results DataFrame df for one dataset,
    compute the mean Jaccard edge-overlap between ECSP paths
    for all pairs of centralities (EBC, ECL, GRAV, ECHO).

    Returns
    -------
    overlap_df : pd.DataFrame
        4x4 DataFrame with rows/cols = CENTRALITIES and
        entries = mean Jaccard overlap. Diagonal entries are 1.0.
    """
    # Ensure path columns are parsed as lists
    for c in CENTRALITIES:
        col = f"path_{c}"
        if col in df.columns:
            df[col] = df[col].apply(parse_path_general)

    # Store overlaps for each (ci, cj)
    pair_values = {
        (ci, cj): []
        for ci in CENTRALITIES
        for cj in CENTRALITIES
        if ci != cj
    }

    for _, row in df.iterrows():
        # Build edge sets for each centrality on this (s,t)
        edge_sets = {}
        for c in CENTRALITIES:
            col = f"path_{c}"
            if col not in df.columns:
                edge_sets[c] = set()
            else:
                edge_sets[c] = path_to_edge_set(row[col])

        # Accumulate Jaccard overlaps
        for ci in CENTRALITIES:
            for cj in CENTRALITIES:
                if ci == cj:
                    continue
                Ei = edge_sets[ci]
                Ej = edge_sets[cj]
                if not Ei and not Ej:
                    # no path for either centrality; skip
                    continue
                inter = len(Ei & Ej)
                union = len(Ei | Ej)
                jacc = inter / union if union > 0 else 0.0
                pair_values[(ci, cj)].append(jacc)

    # Build 4x4 matrix
    mat = np.zeros((len(CENTRALITIES), len(CENTRALITIES)))
    for i, ci in enumerate(CENTRALITIES):
        for j, cj in enumerate(CENTRALITIES):
            if ci == cj:
                mat[i, j] = 1.0
            else:
                vals = pair_values[(ci, cj)]
                mat[i, j] = sum(vals) / len(vals) if vals else 0.0

    overlap_df = pd.DataFrame(
        mat,
        index=CENTRALITIES,
        columns=CENTRALITIES
    )

    print(f"Pairwise ECSP Jaccard overlap for {dataset_name}:")
    print(overlap_df)

    out_path = f"path_overlap_pairwise_{dataset_name}.csv"
    overlap_df.to_csv(out_path)
    print(f"Saved pairwise path overlap matrix to {out_path}")

    return overlap_df


def plot_pairwise_overlap_heatmap(overlap_df, dataset_name, save_prefix="pairwise_overlap"):
    """
    Plot a 4x4 heatmap of pairwise Jaccard overlaps between centrality-based ECSP paths.
    """
    plt.figure(figsize=(5, 4))
    im = plt.imshow(overlap_df.values, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, label="Mean Jaccard overlap")

    labels = overlap_df.columns.tolist()
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.title(f"ECSP pairwise path overlap ({dataset_name})")

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j, i,
                f"{overlap_df.values[i, j]:.2f}",
                ha="center", va="center", fontsize=8
            )

    plt.tight_layout()
    fname = f"{save_prefix}_{dataset_name}.png"
    plt.savefig(fname, dpi=300)
    print(f"Saved pairwise overlap heatmap to {fname}")
    plt.show()


def run_pairwise_overlap_all():
    """
    Compute pairwise ECSP path-overlap matrices for:
      - ER, BA (from ecsp_er_ba_results_larger.csv)
      - CA, EM, PG (their own ECSP CSVs).

    Saves one CSV per dataset:
      path_overlap_pairwise_ER.csv, ..., path_overlap_pairwise_PG.csv
    and plots heatmaps.
    """
    results = {}

    # --- Synthetic ER & BA ---
    df_syn = pd.read_csv("ecsp_er_ba_results_larger.csv")
    if "graph_type" not in df_syn.columns:
        df_syn["graph_type"] = df_syn["graph_id"].str.slice(0, 2)

    for gtype in ["ER", "BA"]:
        df_g = df_syn[df_syn["graph_type"] == gtype].copy()
        overlap_df = compute_pairwise_overlap_for_dataset(df_g, gtype)
        plot_pairwise_overlap_heatmap(overlap_df, gtype,
                                      save_prefix="pairwise_overlap")
        results[gtype] = overlap_df

    # --- Real datasets ---
    df_ca = pd.read_csv("ecsp_CA-GrQc_results.csv")
    df_em = pd.read_csv("ecsp_email-Eu-core_results.csv")
    df_pg = pd.read_csv("ecsp_powergrid_results.csv")

    for dataset_name, df_real in [("CA", df_ca), ("EM", df_em), ("PG", df_pg)]:
        overlap_df = compute_pairwise_overlap_for_dataset(df_real, dataset_name)
        plot_pairwise_overlap_heatmap(overlap_df, dataset_name,
                                      save_prefix="pairwise_overlap")
        results[dataset_name] = overlap_df

    return results




# ============================================================
# 18. SINGLE COMPARATIVE FIGURE (ALL NETWORKS, EBC ATTACK)
# ============================================================

def plot_efficiency_comparison_all():
    """
    Build a single comparative figure:
    global efficiency vs fraction of edges removed
    for all networks under EBC-based ECSP attack.

    Uses robustness result files:
      - ecsp_robustness_results_larger.csv        (ER, BA)
      - ecsp_CA-GrQc_robustness.csv              (CA)
      - ecsp_email-Eu-core_robustness.csv        (EM)
      - ecsp_powergrid_robustness.csv            (PG)
    Adjust file names if needed.
    """
    # Load synthetic robustness
    df_syn = pd.read_csv("ecsp_robustness_results_larger.csv")

    # Make sure graph_type is correct ("ER", "BA")
    # (it should already be; if not, you can reconstruct from graph_id)

    # Load real robustness
    df_ca = pd.read_csv("ecsp_CA-GrQc_robustness.csv")
    df_em = pd.read_csv("ecsp_email-Eu-core_robustness.csv")
    df_pg = pd.read_csv("ecsp_powergrid_robustness.csv")

    df_ca["graph_type"] = "CA"
    df_em["graph_type"] = "EM"
    df_pg["graph_type"] = "PG"

    df_all = pd.concat([df_syn, df_ca, df_em, df_pg], axis=0)

    # Filter to one attack type (EBC)
    df_ebc = df_all[df_all["attack_type"] == "EBC"].copy()

    # For safety, average over graphs if multiple per type
    agg = (df_ebc
           .groupby(["graph_type", "frac_removed"])["efficiency"]
           .mean()
           .reset_index())

    plt.figure(figsize=(8, 5))

    # Define plot order and labels
    order = ["ER", "BA", "CA", "EM", "PG"]
    labels = {
        "ER": "ER (random)",
        "BA": "BA (scale-free)",
        "CA": "CA-GrQc (collab.)",
        "EM": "Email-Eu-core",
        "PG": "Power Grid",
    }

    for gtype in order:
        sub = agg[agg["graph_type"] == gtype]
        if sub.empty:
            continue
        xs = sub["frac_removed"]
        ys = sub["efficiency"]
        plt.plot(xs, ys, marker="o", label=labels.get(gtype, gtype))

    plt.xlabel("Fraction of edges removed (EBC-ECSP attack)")
    plt.ylabel("Global efficiency")
    plt.title("Global efficiency under EBC-based edge removal\nSynthetic vs real networks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("efficiency_comparison_all_networks_EBC.png", dpi=300)
    print("Saved comparative figure to efficiency_comparison_all_networks_EBC.png")
    plt.show()




if __name__ == "__main__":
    # --- (your existing synthetic run_all_experiments_ig here, if you still want it) ---

    # === Real datasets ===
    # CA-GrQc
   
    agg_real = run_agreement_analysis_real()
        # 1) Centrality score correlations
    run_correlation_all_networks()

    # 2) Path overlap metrics
    run_path_overlap_all()

    # 3) Single comparative figure
    plot_efficiency_comparison_all()

    df_ca_ecsp, df_ca_robust = run_real_dataset_experiment_ig(
        dataset_name="CA",
        path=r"C:\Users\Kusal Thapa\Desktop\PhD\doc\matsypura\article\dataset\CA-GrQc.txt",
        loader_fn=load_ca_grqc_ig,
        num_pairs=100,
        save_ecsp_csv_path="ecsp_CA-GrQc_results.csv",
        save_robust_csv_path="ecsp_CA-GrQc_robustness.csv",
         seed=123,
    )
    plot_robustness_curves_single(df_ca_robust, dataset_name="CA",
                                  efficiency_prefix="robust_efficiency_CA",
                                  gcc_prefix="robust_gcc_CA")

    # Email-Eu-core (symmetrized)
    df_em_ecsp, df_em_robust = run_real_dataset_experiment_ig(
        dataset_name="EM",
        path=r"C:\Users\Kusal Thapa\Desktop\PhD\doc\matsypura\article\dataset\email-Eu-core.txt",
        loader_fn=load_email_eu_core_ig,
        num_pairs=100,
        save_ecsp_csv_path="ecsp_email-Eu-core_results.csv",
        save_robust_csv_path="ecsp_email-Eu-core_robustness.csv",
    )
    plot_robustness_curves_single(df_em_robust, dataset_name="EM",
                                  efficiency_prefix="robust_efficiency_EM",
                                  gcc_prefix="robust_gcc_EM")

    # Power Grid
    df_pg_ecsp, df_pg_robust = run_real_dataset_experiment_ig(
        dataset_name="PG",
        path=r"C:\Users\Kusal Thapa\Desktop\PhD\doc\matsypura\article\dataset\powergrid.txt",
        loader_fn=load_powergrid_ig,
        num_pairs=100,
        save_ecsp_csv_path="ecsp_powergrid_results.csv",
        save_robust_csv_path="ecsp_powergrid_robustness.csv",
    )
    plot_robustness_curves_single(df_pg_robust, dataset_name="PG",
                                  efficiency_prefix="robust_efficiency_PG",
                                  gcc_prefix="robust_gcc_PG")
    
    run_path_overlap_all()      # plain vs ECSP
    run_pairwise_overlap_all()  # ECSP vs ECSP

    



