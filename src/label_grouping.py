import os, sys


def suppress_openblas_stderr():
    fd = sys.stderr.fileno()
    old = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd)
    os.close(devnull)
    return old

def restore_stderr(old_fd):
    fd = sys.stderr.fileno()
    os.dup2(old_fd, fd)
    os.close(old_fd)


# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# # optional, usually default anyway
# os.environ["OPENBLAS_VERBOSE"] = "0"

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def coarse_spherical_kmeans(embs_norm, group_size=32, bucket_factor=16, random_state=123):
    """
    embs_norm   : (N, D) normalized embeddings
    group_size  : desired group size for final groups
    bucket_factor: avg cluster size ≈ bucket_factor * group_size
    """
    N, D = embs_norm.shape
    n_clusters = max(1, N // (bucket_factor * group_size))
    print(f"[coarse k-means] N={N}, group_size={group_size}, "
          f"bucket_factor={bucket_factor} -> n_clusters={n_clusters}")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=8192,
        n_init=5,
        random_state=random_state,
        verbose=1,
    )
    coarse_ids = kmeans.fit_predict(embs_norm)
    return coarse_ids


def greedy_groups_local(embs_norm, idxs, group_size, rng):
    """
    embs_norm : (N, D) normalized embeddings (global)
    idxs      : 1D array of global indices in this coarse cluster
    group_size: desired final group size
    rng       : np.random.Generator
    Returns   : list[list[int]] of global indices (each inner list is a group)
    """
    idxs = np.asarray(idxs)
    if idxs.size == 0:
        return []

    local = idxs.copy()
    n_local = len(local)
    unassigned = np.ones(n_local, dtype=bool)

    cluster_embs = embs_norm[local]   # (n_local, D)
    groups_global = []

    while unassigned.any():
        remaining = np.flatnonzero(unassigned)
        seed_local = rng.choice(remaining)

        seed_vec = cluster_embs[seed_local:seed_local+1]   # (1, D)
        sims = (cluster_embs @ seed_vec.T).ravel()         # cosine sims

        sims[~unassigned] = -1e9                           # ignore assigned

        k = min(group_size, len(remaining))
        topk = np.argpartition(-sims, k - 1)[:k]

        # make sure seed is in the group
        if seed_local not in topk:
            worst = topk[np.argmin(sims[topk])]
            topk[topk == worst] = seed_local

        unassigned[topk] = False
        groups_global.append(local[topk].tolist())

    return groups_global

def build_groups(embs_norm, labels, coarse_ids, group_size=32, seed=123):
    """
    embs_norm : (N, D) normalized embeddings
    labels    : (N,) array of label IDs
    coarse_ids: (N,) int cluster index per label
    group_size: desired final group size
    Returns   : list[list[label_id]]
    """
    rng = np.random.default_rng(seed)
    coarse_ids = np.asarray(coarse_ids)

    all_groups_label_ids = []
    unique_clusters = np.unique(coarse_ids)
    print("Num coarse clusters:", len(unique_clusters))

    for cid in unique_clusters:
        cluster_idxs = np.where(coarse_ids == cid)[0]
        if cluster_idxs.size == 0:
            continue

        local_groups = greedy_groups_local(
            embs_norm=embs_norm,
            idxs=cluster_idxs,
            group_size=group_size,
            rng=rng,
        )

        for g in local_groups:
            all_groups_label_ids.append([labels[i] for i in g])

    return all_groups_label_ids




def group_labels(
    cfg,
    label_list=None,                      # None → use ALL labels (non-ht-split)
    emb_path="amazon670k_label_embs.npz",
    group_size=16,
    bucket_factor=16):

    """
    cfg        : Hydra cfg (not used here, but kept for API symmetry)
    label_list : list of label IDs to cluster.
                 - None  → all labels in emb_path (non-ht-split)
                 - not None → only these labels (ht-split tail labels)
    emb_path   : path to npz with keys "labels", "embs"
    group_size : final group size (16 or 32)
    bucket_factor : avg coarse cluster size ≈ bucket_factor * group_size
    save_path  : if not None, np.save(save_path, groups) is called

    Returns    : groups = list[list[label_id]]
    """
    # ---- load NPZ ----
    z = np.load(os.path.join('embeddings',emb_path), allow_pickle=True)
    emb_labels = z["labels"].astype(str)        # (N_all,)
    embs = z["embs"].astype(np.float32)

    # ---- decide which labels to cluster ----
    if label_list is None:
        # non-ht-split: use all labels from .npz
        used_labels = emb_labels
        used_embs = embs
        print(f"Clustering ALL labels: N={used_labels.shape[0]}")
    else:
        # ht-split: cluster only subset (e.g., tail labels)
        label_list = [str(l) for l in label_list]
        mask = np.isin(emb_labels, label_list)
        used_labels = emb_labels[mask]
        used_embs = embs[mask]
        print(f"Clustering subset: N={used_labels.shape[0]} out of {emb_labels.shape[0]} "
              f"(label_list size={len(label_list)})")

        missing = set(label_list) - set(used_labels)
        if missing:
            print(f"WARNING: {len(missing)} labels in label_list have no embedding in {emb_path}.")

    N_used, D = used_embs.shape
    print(f"Used embeddings: N={N_used}, D={D}")

    old_fd = suppress_openblas_stderr()
    # ---- coarse k-means ----
    coarse_ids = coarse_spherical_kmeans(
        used_embs,
        group_size=group_size,
        bucket_factor=bucket_factor,
        random_state=123,
    )

    restore_stderr(old_fd)

    # ---- build groups ----
    used_labels_list = list(used_labels)
    groups = build_groups(
        embs_norm=used_embs,
        labels=used_labels_list,
        coarse_ids=coarse_ids,
        group_size=group_size,
        seed=123,
    )

    print("Num groups:", len(groups))
    print("First 3 group sizes:", [len(g) for g in groups[:3]])
    print("Example group:", groups[0][:10])
    print(" Sanity check...")
    sanity_check(used_embs,groups,used_labels_list)

    return groups




## --------------------------- EVALUATION (SANITY CHECKS) --------------------------------------- ##

def evaluate_grouping(embs_norm, groups, label_to_idx, sample_groups=1000, seed=0):
    """
    embs_norm    : (N, D) normalized embeddings
    groups       : list[list[label_id]]
    label_to_idx : dict[label_id -> row index in embs_norm]
    sample_groups: how many groups to subsample for eval
    """
    rng = np.random.default_rng(seed)
    G = len(groups)
    if G == 0:
        return None

    group_indices = np.arange(G)
    if G > sample_groups:
        group_indices = rng.choice(group_indices, size=sample_groups, replace=False)

    all_scores = []

    for gi in group_indices:
        g = groups[gi]
        if len(g) <= 1:
            continue

        # map label_ids -> embedding indices
        idxs = [label_to_idx[lbl] for lbl in g]
        emb_group = embs_norm[idxs]           # (k, D)

        # centroid
        centroid = emb_group.mean(axis=0, keepdims=True)  # (1, D)
        centroid /= (np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-8)

        # cosine sims to centroid
        scores = (emb_group @ centroid.T).ravel()
        all_scores.append(scores)

    if not all_scores:
        return None

    all_scores = np.concatenate(all_scores)
    return {
        "mean_cos": float(all_scores.mean()),
        "median_cos": float(np.median(all_scores)),
        "p10_cos": float(np.percentile(all_scores, 10)),
        "min_cos": float(all_scores.min()),
    }

def make_random_groups(labels, group_size=32, seed=0):
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(labels))
    rng.shuffle(idxs)

    groups = []
    for start in range(0, len(idxs), group_size):
        chunk = idxs[start:start+group_size]
        groups.append([labels[i] for i in chunk])
    return groups

def sanity_check(embs,groups,labels):
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)} 
    stats = evaluate_grouping(embs, groups, label_to_idx, sample_groups=1000)
    print("Grouping cohesion stats:", stats)
    rand_groups = make_random_groups(labels, group_size=32, seed=42)
    rand_stats = evaluate_grouping(embs, rand_groups, label_to_idx, sample_groups=1000)
    print("Random grouping stats:", rand_stats)


