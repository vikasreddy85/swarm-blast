"""Redraw the E1 and E7 figures from saved outputs, without new model calls.

  python replot.py [queue_out]
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

root = sys.argv[1] if len(sys.argv) > 1 else "queue_out"

# E1: infection curves from the saved logs
for d in sorted(glob.glob(os.path.join(root, "e1_v*"))):
    curves = {}
    for p in sorted(glob.glob(os.path.join(d, "log_*.json"))):
        top = os.path.basename(p).split("_")[1]
        curves.setdefault(top, []).append(json.load(open(p))["curve"])
    if not curves:
        continue
    cfg = yaml.safe_load(open(os.path.join(d, ".hydra", "config.yaml")))
    model, n, vig = cfg["model"], cfg["llm"]["n"], cfg["llm"]["vigilance"]
    plt.figure(figsize=(6, 4))
    for top, cs in curves.items():
        m = pd.DataFrame(cs).mean(axis=0)
        plt.plot(range(1, len(m) + 1), m.values / n, marker="o", label=top)
    plt.xlabel("round")
    plt.ylabel("fraction infected")
    plt.title("LLM agents (%s, vigilance %d)" % (model, vig))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(d, "e1_curves.png"), dpi=130)
    plt.close()
    print("redrew", d)

# E7: fraction of outbreaks detected by round, from ttd_runs.csv
for d in sorted(glob.glob(os.path.join(root, "e7_*"))):
    p = os.path.join(d, "ttd_runs.csv")
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p)
    ticks = yaml.safe_load(open(os.path.join(d, ".hydra", "config.yaml")))["sim"]["ticks"]
    tops = list(dict.fromkeys(df.topology))
    lo, hi = df.mu_core.min(), df.mu_core.max()
    fig, ax = plt.subplots(1, len(tops), figsize=(5 * len(tops), 4), sharey=True, squeeze=False)
    for i, top in enumerate(tops):
        for det, col in zip(["lex", "sem", "beh"], ["C0", "C1", "C2"]):
            for mc, ls in [(lo, "-"), (hi, "--")]:
                g = df[(df.topology == top) & (df.det == det) & (df.mu_core == mc) & (df.attack_rate >= 0.4)]
                if len(g):
                    ax[0][i].plot(range(1, ticks + 1), [(g.t_detect <= t).mean() for t in range(ticks)], color=col, linestyle=ls, label="%s mu_core=%.1f" % (det, mc))
        ax[0][i].set_title(top)
        ax[0][i].set_xlabel("round")
    ax[0][0].set_ylabel("fraction of outbreaks detected by round")
    ax[0][0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(d, "e7_ttd.png"), dpi=130)
    plt.close()
    print("redrew", d)
