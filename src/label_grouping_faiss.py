import os
import numpy as np
from label_grouping import evaluate_grouping, make_random_groups

def group_labels_faiss(
    cfg,
    label_list=None,
    emb_path="amazon670k_label_embs.npz",
    group_size=16,
    probe_k=512,          # how many neighbors to retrieve per seed (>= group_size)
    seed=123,
):
    """
    Global greedy NN packing using FAISS.

    Returns: groups = list[list[label_id]]
    """
    import faiss  # pip install faiss-cpu or faiss-gpu

    # ---- load NPZ ----
    z = np.load(os.path.join("embeddings", emb_path), allow_pickle=True)
    emb_labels = z["labels"].astype(str)
    embs = z["embs"].astype(np.float32)  # assume already L2-normalized

    # ---- select subset if requested ----
    if label_list is None:
        used_labels = emb_labels
        used_embs = embs
        print(f"FAISS grouping ALL labels: N={used_labels.shape[0]}")
    else:
        label_list = [str(l) for l in label_list]
        mask = np.isin(emb_labels, label_list)
        used_labels = emb_labels[mask]
        used_embs = embs[mask]
        print(f"FAISS grouping subset: N={used_labels.shape[0]} out of {emb_labels.shape[0]}")

        missing = set(label_list) - set(used_labels.tolist())
        if missing:
            print(f"WARNING: {len(missing)} labels in label_list have no embedding in {emb_path}.")

    N, D = used_embs.shape
    assert probe_k >= group_size, "probe_k must be >= group_size"
    print(f"Used embeddings: N={N}, D={D}, group_size={group_size}, probe_k={probe_k}")

    # ---- build FAISS index (cosine = inner product because embeddings are normalized) ----
    index = faiss.IndexHNSWFlat(D, 32)   # M=32 is a decent default
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = max(64, probe_k)
    index.add(used_embs)

    rng = np.random.default_rng(seed)
    unassigned = np.ones(N, dtype=bool)
    groups = []

    # For fast filtering: map from row idx -> label_id string
    used_labels_list = used_labels.tolist()

    while unassigned.any():
        remaining = np.flatnonzero(unassigned)
        seed_idx = rng.choice(remaining)

        # Query for a bunch of neighbors; include self
        k = min(probe_k, N)
        sims, nbrs = index.search(used_embs[seed_idx:seed_idx+1], k)
        nbrs = nbrs.ravel()

        # Keep only unassigned
        nbrs = nbrs[unassigned[nbrs]]

        # If not enough found (rare), fallback to random fill
        if nbrs.size < group_size:
            need = group_size - nbrs.size
            extra = remaining[remaining != seed_idx]
            if extra.size > 0:
                take = min(need, extra.size)
                nbrs = np.concatenate([nbrs, rng.choice(extra, size=take, replace=False)])

        # Finalize group (may be smaller only at the very end)
        grp_idxs = nbrs[:min(group_size, nbrs.size)]
        unassigned[grp_idxs] = False

        groups.append([used_labels_list[i] for i in grp_idxs])

    print("Num groups:", len(groups))
    print("First 3 group sizes:", [len(g) for g in groups[:3]])
    print("Example group:", groups[0][:10])

    print("Sanity check (FAISS vs Random)...")
    sanity_check_faiss(
        embs_norm=used_embs,
        groups=groups,
        labels=used_labels_list,
        group_size=group_size,
        sample_groups=1000,
    )



    return groups


def sanity_check_faiss(embs_norm, groups, labels, group_size, sample_groups=1000):
    # labels must be in the SAME order as embs_norm rows
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    print("FAISS grouping cohesion stats:")
    stats = evaluate_grouping(
        embs_norm,
        groups,
        label_to_idx,
        sample_groups=sample_groups,
        seed=0,
    )
    print(stats)

    print("Random grouping cohesion stats:")
    rand_groups = make_random_groups(labels, group_size=group_size, seed=42)
    rand_stats = evaluate_grouping(
        embs_norm,
        rand_groups,
        label_to_idx,
        sample_groups=sample_groups,
        seed=0,
    )
    print(rand_stats)
