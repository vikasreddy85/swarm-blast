"""Combine the queue's per-job outputs into cross-job tables and the headline figures."""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "queue_out"
out = os.path.join(root, "summary")
os.makedirs(out, exist_ok=True)
pd.set_option("display.width", 220)


def load(pattern):
    frames = []
    for p in sorted(glob.glob(os.path.join(root, pattern))):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if len(df):
            frames.append(df.assign(job=os.path.basename(os.path.dirname(p))))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---- E3 on LLMs: replication versus detection --------------------------------------------------------
ad = load("e3l_*/adaptive.csv")
if len(ad):
    ad.to_csv(os.path.join(out, "adaptive_all.csv"), index=False)
    keys = ["model", "vigilance", "variant"]
    cols = ["attack_rate", "exact_copy_frac", "auc_lex", "auc_sem", "auc_beh", "median_t_lex", "median_t_sem", "median_t_beh", "saved_sem"]
    tab = ad.groupby(keys)[cols].mean().reset_index()
    tab.to_csv(os.path.join(out, "adaptive_by_variant.csv"), index=False)
    print("== E3 on LLMs: mean over topologies ==")
    print(tab.round(2).to_string(index=False))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for a, det in zip(ax, ["lex", "sem", "beh"]):
        for variant, col in zip(["exact", "wrapper", "full"], ["C0", "C1", "C3"]):
            g = ad[ad.variant == variant]
            a.scatter(g.attack_rate, g["auc_" + det], color=col, alpha=0.7, label=variant)
        a.set_title({"lex": "lexical", "sem": "similarity", "beh": "behavioral"}[det])
        a.set_xlabel("attack rate (replication)")
        a.set_ylim(0, 1.05)
    ax[0].set_ylabel("detector AUC")
    ax[0].legend(title="payload variant")
    fig.suptitle("Replication vs detection on LLM agents (one point per model x vigilance x topology)")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "e3l_tradeoff.png"), dpi=130)

# ---- time to detect ----------------------------------------------------------------------------------
tl = load("e3l_*/ttd_runs.csv")
if len(tl):
    ob = tl[tl.attack_rate >= 0.4]
    t = ob.groupby(["variant", "det"]).agg(runs=("seed", "size"), detect_rate=("detected", "mean"), median_t=("t_detect", "median"), frac_infected_at_detect=("frac_at_detect", "median"), saved=("saved_frac", "mean")).reset_index()
    t.to_csv(os.path.join(out, "ttd_llm.csv"), index=False)
    print("\n== time to detect, LLM outbreaks (attack rate >= 0.4) ==")
    print(t.round(2).to_string(index=False))
tm = load("e7_*/ttd.csv")
if len(tm):
    tm.to_csv(os.path.join(out, "ttd_mock.csv"), index=False)
    print("\n== time to detect, mock ==")
    print(tm.round(2).to_string(index=False))

# ---- horizontal transfer ----------------------------------------------------------------------------
h = load("e8_*/hgt_auc.csv")
if len(h):
    h.to_csv(os.path.join(out, "hgt_auc.csv"), index=False)
    print("\n== horizontal-transfer test on the mock (AUC, planted recombination vs none) ==")
    print(h.round(2).to_string(index=False))
hl = ad[["model", "vigilance", "topology", "variant", "hgt_r_core_tree", "hgt_r_wrap_tree", "hgt_disc_index", "hgt_np_disc"]] if len(ad) else pd.DataFrame()
if len(hl):
    hl = hl.dropna(subset=["hgt_np_disc", "hgt_disc_index"], how="all")
    hl.to_csv(os.path.join(out, "hgt_llm.csv"), index=False)
    print("\n== horizontal-transfer statistics on real LLM logs (no ground truth; compare with the mock baseline) ==")
    print(hl.groupby(["variant", "topology"])[["hgt_r_core_tree", "hgt_r_wrap_tree", "hgt_disc_index", "hgt_np_disc"]].mean().round(2).to_string())
print("\nwritten to", out)