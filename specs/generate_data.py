#!/usr/bin/env python3

from __future__ import annotations
import argparse, json, os
from typing import List, Tuple, Dict

import numpy as np
from torch.utils.data import ConcatDataset
from torchvision.datasets import CIFAR10

from data_utils import iid_split, pathological_non_iid_split, by_labels_non_iid_split

# dataset sizes ----------------------------------------------------------
TRAIN_SIZE, TEST_SIZE = 50_000, 10_000
RAW = "raw_data/"
N_CLASSES = 10


# ────────────────────────────────────────────────────────────────────────
#  helpers
# ────────────────────────────────────────────────────────────────────────
def factorise_three_layer(n: int) -> Tuple[int, List[int]]:
    """
    'Roundest' 3-layer decomposition  n = 1 + d + Σ g_i   (root + middles + leaves)
    Returns d and [g₁ … g_d] so that |g-d| is minimum when a perfect tree exists,
    otherwise distributes the remainder as evenly as possible.

    Logic copied (verbatim) from make_3layer_tree._roundest_factorisation.
    """
    candidates: list[tuple[int, int, int]] = []        # (|g-d|, d, g)
    for d in range(1, n):
        leaves = n - 1 - d
        if leaves < d:
            break
        g, r = divmod(leaves, d)
        if r == 0:
            candidates.append((abs(g - d), d, g))

    if candidates:                                      # perfect tree exists
        _, d, g = min(candidates)
        return d, [g] * d

    # otherwise minimise the remainder
    best: Tuple[int, int, int] | None = None           # (rem, d, g)
    for d in range(1, n):
        leaves = n - 1 - d
        if leaves < d:
            break
        g, r = divmod(leaves, d)
        if best is None or r < best[0]:
            best = (r, d, g)
    if best is None:
        raise ValueError("cannot form a 3-layer tree with so few nodes")
    r, d, g = best
    return d, [g + 1] * r + [g] * (d - r)


def split_train_test(indices: List[int]) -> Tuple[List[int], List[int]]:
    idx = np.asarray(indices)
    return idx[idx < TRAIN_SIZE].tolist(), idx[idx >= TRAIN_SIZE].tolist()


def allocate_counts(pool_sizes: List[int], target: int) -> List[int]:
    """
    Proportional allocation WITHOUT exceeding individual pool sizes.
    Guarantees Σ out ≤ target and out_i ≤ pool_i.
    """
    avail = sum(pool_sizes)
    if avail == 0 or target == 0:
        return [0] * len(pool_sizes)
    scale = min(1.0, target / avail)
    raw = [int(np.floor(scale * s)) for s in pool_sizes]
    short = target - sum(raw)
    if short > 0:                                       # distribute remainder
        order = np.argsort([-s for s in pool_sizes])
        for i in order:
            room = pool_sizes[i] - raw[i]
            take = min(room, short)
            raw[i] += take
            short -= take
            if short == 0:
                break
    return raw


def pick_unique(rng: np.random.Generator,
                universe: np.ndarray,
                k: int,
                already: set[int] | None = None) -> List[int]:
    """
    Pick k distinct indices from `universe` avoiding the ones in `already`.
    """
    if k == 0:
        return []
    if already:
        universe = np.setdiff1d(universe, list(already), assume_unique=True)
    if k > len(universe):
        raise RuntimeError("BUG: ask more elements than available")
    return rng.choice(universe, size=k, replace=False).tolist()


def parse_partition(txt: str) -> Tuple[float, float, float]:
    parts = [float(x) for x in txt.split(",")]
    if len(parts) != 3 or any(p < 0 for p in parts):
        raise ValueError("--partition needs three comma-separated numbers")
    s = sum(parts)
    if not 99.9 < s < 100.1:
        raise ValueError("percentages must sum to 100")
    return tuple(p / 100 for p in parts)


def scenario(name: str) -> Tuple[float, float, float]:
    name = name.lower()
    if name == "equal":
        return 0.333, 0.333, 0.334
    if name == "biased":
        return 0.571, 0.286, 0.143
    if name == "highly_biased":
        return 0.767, 0.199, 0.034
    raise ValueError("--scenario must be equal|biased|highly_biased")


def pick(rng: np.random.Generator, pool: List[int], k: int) -> List[int]:
    if k == 0:
        return []
    if len(pool) < k:
        raise RuntimeError("BUG: asked more than available")
    return rng.choice(pool, size=k, replace=False).tolist()


# ──────────────────────────────────────────────────────────────────────
# core assignment builder
# ──────────────────────────────────────────────────────────────────────
def build_assignment(pools: List[List[int]],
                     n_clients: int,
                     p_root: float, p_mid: float, p_leaf: float,
                     s_frac: float, s_frac_test: float,
                     seed: int) -> Dict[str, Dict[str, List[int]]]:

    rng = np.random.default_rng(seed)

    # -------- tree structure -------------------------------------------
    d, g_list = factorise_three_layer(n_clients)        # middles & leaves
    root_id  = 0
    mid_ids  = list(range(1, 1 + d))
    leaf_ids = list(range(1 + d, n_clients))

    # -------- layer targets --------------------------------------------
    tot_tr = int(round(TRAIN_SIZE * s_frac))
    tot_te = int(round(TEST_SIZE  * s_frac_test))

    tgt_root_tr = int(round(tot_tr * p_root))
    tgt_mid_tr  = int(round(tot_tr * p_mid))
    tgt_leaf_tr = tot_tr - tgt_root_tr - tgt_mid_tr      # remove rounding drift

    tgt_root_te = int(round(tot_te * p_root))
    tgt_mid_te  = int(round(tot_te * p_mid))
    tgt_leaf_te = tot_te - tgt_root_te - tgt_mid_te

    # ------------------------------------------------------------------
    assign: Dict[str, Dict[str, List[int]]] = {}

    all_train = np.arange(TRAIN_SIZE)
    all_test  = np.arange(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE)

    # ===== 1) leaves – **purely from their own (non-IID) pools** =====
    leaf_tr_sz, leaf_te_sz = [], []
    for idx in leaf_ids:
        tr, te = split_train_test(pools[idx])
        leaf_tr_sz.append(len(tr)); leaf_te_sz.append(len(te))

    leaf_tr_quota = allocate_counts(leaf_tr_sz, tgt_leaf_tr)
    leaf_te_quota = allocate_counts(leaf_te_sz, tgt_leaf_te)

    for out_id, pool_id, ktr, kte in zip(leaf_ids, leaf_ids,
                                         leaf_tr_quota, leaf_te_quota):
        tr_pool, te_pool = split_train_test(pools[pool_id])
        assign[str(out_id)] = {
            "train": rng.choice(tr_pool, size=ktr, replace=False).tolist(),
            "test":  rng.choice(te_pool, size=kte, replace=False).tolist()
        }

    # keep track of what *leaf* data we already grabbed (to avoid duplicates
    # **within** a client, duplicates ACROSS clients are allowed/expected)
    used_train = {i for v in assign.values() for i in v["train"]}
    used_test  = {i for v in assign.values() for i in v["test"]}

    # ===== 2) middles – take what they can from their own pool, top-up =====
    mid_tr_sz, mid_te_sz = [], []
    for idx in mid_ids:
        tr, te = split_train_test(pools[idx])
        mid_tr_sz.append(len(tr)); mid_te_sz.append(len(te))

    # first grab *everything* from their own pool (do NOT oversample – that
    # keeps the non-IID flavour), then fill the gap from the global universe
    mid_tr_from_pool = allocate_counts(mid_tr_sz, tgt_mid_tr)
    mid_te_from_pool = allocate_counts(mid_te_sz, tgt_mid_te)

    remaining_tr_mid = tgt_mid_tr - sum(mid_tr_from_pool)
    remaining_te_mid = tgt_mid_te - sum(mid_te_from_pool)

    for out_id, pool_id, ktr, kte in zip(mid_ids, mid_ids,
                                         mid_tr_from_pool, mid_te_from_pool):
        tr_pool, te_pool = split_train_test(pools[pool_id])
        tr_sel = rng.choice(tr_pool, size=ktr, replace=False).tolist()
        te_sel = rng.choice(te_pool, size=kte, replace=False).tolist()
        assign[str(out_id)] = {"train": tr_sel, "test": te_sel}
        used_train.update(tr_sel)
        used_test.update(te_sel)

    # now top-up middles (round-robin) from the global dataset --------------
    for cid in mid_ids:
        need_tr = tgt_mid_tr // d - len(assign[str(cid)]["train"])
        need_te = tgt_mid_te // d - len(assign[str(cid)]["test"])
        if need_tr > 0:
            extra = pick_unique(rng, all_train, need_tr, used_train)
            assign[str(cid)]["train"].extend(extra); used_train.update(extra)
        if need_te > 0:
            extra = pick_unique(rng, all_test, need_te, used_test)
            assign[str(cid)]["test"].extend(extra);  used_test.update(extra)

    # ===== 3) root – completely global sample (duplicates with others OK) ==
    assign["0"] = {
        "train": pick_unique(rng, all_train, tgt_root_tr),
        "test":  pick_unique(rng, all_test,  tgt_root_te)
    }

    return assign


# ────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_clients", required=True, type=int)
    # split style
    sty = p.add_mutually_exclusive_group()
    sty.add_argument("--pathological_split", action="store_true")
    sty.add_argument("--by_labels_split",    action="store_true")
    # knobs for non-IID splits
    p.add_argument("--n_shards",     type=int, default=2)
    p.add_argument("--n_components", type=int, default=-1)
    p.add_argument("--alpha",        type=float, default=0.5)
    # fractions
    p.add_argument("--s_frac",       type=float, default=0.9)
    p.add_argument("--s_frac_test",  type=float, default=1.0)
    # budget
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--scenario", choices=["equal", "biased", "highly_biased"])
    grp.add_argument("--partition",
                     help="pct_root,pct_middle,pct_leaf – must sum to 100")
    # misc
    p.add_argument("--seed",    type=int, default=12345)
    p.add_argument("--outfile", required=True)
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────
#  main
# ────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = cli()

    if args.partition:
        p_root, p_mid, p_leaf = parse_partition(args.partition)
    else:
        p_root, p_mid, p_leaf = scenario(args.scenario or "equal")

    # download dataset once ---------------------------------------------
    dataset = ConcatDataset([
        CIFAR10(root=RAW, download=True,  train=True),
        CIFAR10(root=RAW, download=False, train=False)
    ])

    # candidate pools ----------------------------------------------------
    if args.pathological_split:
        pools = pathological_non_iid_split(
            dataset, N_CLASSES, args.n_clients,
            n_classes_per_client=args.n_shards,
            frac=1.0, seed=args.seed)
        split_name = "pathological"
    elif args.by_labels_split:
        pools = by_labels_non_iid_split(
            dataset, N_CLASSES, args.n_clients,
            n_components=args.n_components, alpha=args.alpha,
            frac=1.0, seed=args.seed)
        split_name = "by_labels"
    else:
        pools = iid_split(dataset, args.n_clients, frac=1.0, seed=args.seed)
        split_name = "iid"

    assignment = build_assignment(
        pools, args.n_clients,
        p_root, p_mid, p_leaf,
        args.s_frac, args.s_frac_test,
        args.seed
    )

    # -------- duplicate test → validation ------------------------------
    for client_dict in assignment.values():
        client_dict["validation"] = list(client_dict["test"])

    cfg = {
        "name": "cifar10",
        "node_dataset_assignment": assignment
    }

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    with open(args.outfile, "w") as f:
        json.dump(cfg, f, indent=2)

    # -------- debug ----------------------------------------------------
    tot_tr = sum(len(v["train"]) for v in assignment.values())
    tot_va = sum(len(v["validation"]) for v in assignment.values())
    tot_te = sum(len(v["test"]) for v in assignment.values())

    print(f"Wrote {args.outfile}")
    print(f"    totals: {tot_tr} train | {tot_va} validation | {tot_te} test")
    for cid in sorted(assignment, key=int):
        v = assignment[cid]
        print(f"    client {cid:>2}: "
              f"{len(v['train']):5d} train | "
              f"{len(v['validation']):4d} val | "
              f"{len(v['test']):4d} test")


if __name__ == "__main__":
    main()
