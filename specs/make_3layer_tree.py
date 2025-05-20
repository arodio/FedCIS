#!/usr/bin/env python3
"""
make_tree.py  –  produce 3-layer tree JSONs.

Usage examples
--------------
# 1) the original 7-node 80-15-5
./make_tree.py 7 80 15 5

# 2) the 17-client 45-35-20
./make_tree.py 17 45 35 20

# 3) any new size (e.g. 100 nodes) with 60-30-10 profile
./make_tree.py 100 60 30 10
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys
from typing import Dict, List, Tuple


# ────────────────────────────────────────────────────────────────────────
# 0.  helpers
# ────────────────────────────────────────────────────────────────────────
def _roundest_factorisation(N: int) -> Tuple[int, List[int]]:
    """
    Return the size of the middle layer (d) and a list with the g_i
    (leaves hanging from each middle) for the 'roundest' 3-layer tree.
    """
    candidates: list[tuple[int,int,int]] = []        # ( |g-d| , d , g )
    for d in range(1, N):
        leaves = N - 1 - d
        if leaves < d:
            break
        g, r = divmod(leaves, d)
        if r == 0:
            candidates.append((abs(g-d), d, g))

    if candidates:                                   # perfect tree exists
        _, d, g = min(candidates)
        return d, [g]*d

    # otherwise distribute the remainders
    best: Tuple[int,int,int] | None = None           # (rem, d, g)
    for d in range(1, N):
        leaves = N - 1 - d
        if leaves < d:
            break
        g, r = divmod(leaves, d)
        if best is None or r < best[0]:
            best = (r, d, g)
    if best is None:
        raise ValueError("cannot build 3-layer tree with so few nodes")
    r, d, g = best
    g_list = [g+1]*r + [g]*(d-r)                     # one extra leaf on first r middles
    return d, g_list


def _auto_lambda(n_leaves: int, p2: int, p3: int) -> float:
    """Heuristic explained in §2."""
    if n_leaves == 4 and (p2, p3) == (33, 33):       # balanced special
        return 12 / (n_leaves * (p2+p3) / 100)
    if n_leaves in (4, 12):
        return {4: 10/4, 12: 12/12}[n_leaves]
    # fallback: keep total arrival at 10
    return 10 / n_leaves


# ────────────────────────────────────────────────────────────────────────
# 1.  main builder
# ────────────────────────────────────────────────────────────────────────
def build_tree(N: int, p1: int, p2: int, p3: int,
               lambda_leaf: float | None = None) -> Dict:
    assert p1 + p2 + p3 == 100, "percentages must sum to 100"

    d, g_list = _roundest_factorisation(N)
    n_leaves = N - 1 - d
    if lambda_leaf is None:
        lambda_leaf = _auto_lambda(n_leaves, p2, p3)

    # ----- edges (root edges first, then children of each middle) -------
    edges: list[tuple[int,int]] = []
    next_leaf = 1 + d
    for m_idx, g_i in enumerate(g_list):
        m_id = m_idx + 1
        edges.append((0, m_id))                      # root → middle-i
        for _ in range(g_i):
            edges.append((m_id, next_leaf))          # middle-i → leaf
            next_leaf += 1

    # ----- layers -------------------------------------------------------
    root = {0}
    middle = set(range(1, 1+d))
    leaves = set(range(1+d, N))

    # ----- node model sizes --------------------------------------------
    node_model_size = {str(n): 3 if n in root else 2 if n in middle else 1
                       for n in range(N)}

    # ----- arrival ------------------------------------------------------
    arrival = {str(n): 0.0 for n in root | middle}
    arrival.update({str(n): lambda_leaf for n in leaves})

    # ----- departure ----------------------------------------------------
    dep: Dict[str, float] = {"0": 0.0}
    for m_id, g_i in zip(range(1, 1+d), g_list):
        dep[str(m_id)] = lambda_leaf * p3 * g_i / 100
    dep_leaf = lambda_leaf * (p2 + p3) / 100
    for n in leaves:
        dep[str(n)] = dep_leaf

    return {
        "toplogy_as_edges": edges,
        "node_model_size": node_model_size,
        "rates": {"arrival": arrival, "departure": dep},
        "serving_weights_rescale": True,
        "loss_rescale": False,
        "rescale_by_flow_map": False,
        "difficulty_quantization": 1
    }


# ────────────────────────────────────────────────────────────────────────
# 2.  CLI wrapper
# ────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("N",   type=int, help="total number of nodes")
    ap.add_argument("P1",  type=int, help="root %")
    ap.add_argument("P2",  type=int, help="middle-layer %")
    ap.add_argument("P3",  type=int, help="leaf %")
    ap.add_argument("-l", "--lambda-leaf", type=float,
                    help="override automatic lambda (leaf arrival)")
    ap.add_argument("-o", "--outfile", type=pathlib.Path,
                    help="where to save (default: name derived from inputs)")
    args = ap.parse_args()

    data = build_tree(args.N, args.P1, args.P2, args.P3, args.lambda_leaf)

    if args.outfile is None:
        suffix = f"-{args.N}_clients" if args.N != 100 else ""
        args.outfile = pathlib.Path(
            f"3-layer-tree-{args.P1}-{args.P2}-{args.P3}{suffix}.json"
        )

    args.outfile.write_text(json.dumps(data, indent=2))
    print(f"Wrote {args.outfile}")


if __name__ == "__main__":
    main()
