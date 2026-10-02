#!/usr/bin/env python3
"""Summarize the dual-side scaling matrix from runs/sc_*.jsonl.

Prints, for each eval length (4096 train / 16384 / 65536 zero-shot):
stack bpc matrix, matched-baseline bpc matrix, baseline/stack ratio matrix,
and the marginal scaling ratios (write-side L1->L8, read-side G1->G4).
"""
import json
import os

LENGTHS = [4096, 16384, 65536]
LS = [1, 2, 4, 8]
GS = [1, 2, 4]


def cell_bpc(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        r = json.loads(line)
        if r.get("event") == "eval" and "bpc" in r and "length" in r:
            out[r["length"]] = r["bpc"]
    return out


def show(title, get):
    print(f"\n== {title} (rows: read-side G, cols: write-side L) ==")
    print("      " + "".join(f"L={l:<7}" for l in LS))
    for g in GS:
        row = f"G={g}   "
        for l in LS:
            v = get(l, g)
            row += f"{v:<8.4f}" if v is not None else "-       "
        print(row)


def main():
    stack = {(l, g): cell_bpc(f"runs/sc_l{l}g{g}.jsonl") for l in LS for g in GS}
    base = {(l, g): cell_bpc(f"runs/sc_l{l}g{g}_base.jsonl") for l in LS for g in GS}
    for n in LENGTHS:
        show(f"stack bpc @ {n}", lambda l, g, n=n: stack[(l, g)].get(n))
        show(f"baseline bpc @ {n}", lambda l, g, n=n: base[(l, g)].get(n))
        show(f"baseline / stack @ {n}",
             lambda l, g, n=n: (base[(l, g)][n] / stack[(l, g)][n])
             if n in base[(l, g)] and n in stack[(l, g)] else None)
    print("\n== marginal ratios (train length 4096) ==")
    for g in GS:
        a, b = stack[(1, g)].get(4096), stack[(8, g)].get(4096)
        if a and b:
            print(f"G={g}: write-side L1->L8  {a:.4f} -> {b:.4f}  (x{a / b:.3f})")
    for l in LS:
        a, b = stack[(l, 1)].get(4096), stack[(l, 4)].get(4096)
        if a and b:
            print(f"L={l}: read-side  G1->G4  {a:.4f} -> {b:.4f}  (x{a / b:.3f})")


if __name__ == "__main__":
    main()
