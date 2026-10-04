"""Build the field guide page from site/src.html, the example runs and every finished E3-LLM job.

  python site/build_site.py            writes site/index.html (standalone page) and site/fragment.html
"""
import glob
import json
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ORDER = ["gpt-4o-mini", "deepseek-v4-pro", "grok-4.3", "kimi-k2.5", "glm-4.6", "llama-4-maverick", "mistral-large-2512"]


def results():
    frames = []
    for p in glob.glob(os.path.join(ROOT, "queue_out", "e3l_v*", "adaptive.csv")):
        d = os.path.dirname(p)
        if not os.path.exists(os.path.join(d, ".done")) and "openai_gpt-4o-mini" not in d:
            continue
        frames.append(pd.read_csv(p).assign(job=os.path.basename(d)))
    df = pd.concat(frames, ignore_index=True)
    df["name"] = df.model.str.split("/").str[-1]
    out = []
    names = sorted(set(df.name), key=lambda n: ORDER.index(n) if n in ORDER else 99)
    for n in names:
        g = df[df.name == n]
        # one job per (vigilance, topology) for split runs; gpt-4o-mini jobs hold all 3 topologies
        done = len(g.groupby(["vigilance", "topology"]))
        attack = {v: {int(k): round(float(x), 3) for k, x in g[g.variant == v].groupby("vigilance").attack_rate.mean().items()} for v in ["exact", "wrapper", "full"]}
        f = g[g.variant == "full"]
        num = lambda c: None if f[c].isna().all() else round(float(f[c].mean()), 3)
        out.append(dict(name=n, done=done, attack=attack, full=dict(attack=num("attack_rate"), reworded=None if f.exact_copy_frac.isna().all() else round(1 - float(f.exact_copy_frac.mean()), 3), auc_lex=num("auc_lex_outbreak"), auc_sem=num("auc_sem_outbreak"), auc_beh=num("auc_beh_outbreak"))))
    return dict(models=out)


def main():
    t = open(os.path.join(HERE, "src.html")).read()
    t = t.replace("__DATA__", open(os.path.join(HERE, "runs.json")).read().replace("</", "<\\/"))
    t = t.replace("__RESULTS__", json.dumps(results()).replace("</", "<\\/"))
    open(os.path.join(HERE, "fragment.html"), "w").write(t)
    head = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
    open(os.path.join(HERE, "index.html"), "w").write(head + t.replace("</style>", "</style>\n</head><body>", 1) + "\n</body></html>")
    print("built", os.path.join(HERE, "index.html"))
    for m in results()["models"]:
        print(m["name"], m["done"], "/ 12 jobs", m["full"])


if __name__ == "__main__":
    main()
