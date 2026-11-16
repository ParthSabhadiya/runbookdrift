#!/usr/bin/env python3
"""Temporal evaluation of stale-step detection on RunbookDrift.

Train on window 2023-11 (labels observed by 2024-11), test on window 2024-11
(labels observed by 2025-11). Single-signal baselines need no training.
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

SCOPE = sys.argv[1] if len(sys.argv) > 1 else "maintained"   # maintained | all
tr = pd.read_csv("w2023.csv"); te = pd.read_csv("w2024.csv")


def prep(d):
    d = d[d.label != "reworded"].copy()
    if SCOPE == "maintained":
        d = d[d.doc_rewritten == 0]
    d["y"] = (d.label == "drifted").astype(int)
    d["is_code"] = (d.kind == "code").astype(int)
    for c in ["line_age", "doc_age", "ent_conf_age_max", "ent_conf_age_mean", "df_other_min",
              "df_other_mean", "doc_commits_1y", "n_ents", "retired_other_docs_max"]:
        d["log_" + c] = np.log1p(d[c].clip(lower=0))
    d["recent_retire"] = np.exp(-d.days_since_retired_elsewhere / 90.0)
    return d


tr, te = prep(tr), prep(te)

AGE = ["log_line_age", "log_doc_age"]
HIST = ["log_doc_commits_1y"]
GRAPH = ["log_ent_conf_age_max", "log_ent_conf_age_mean", "frac_unconfirmed_1y",
         "log_df_other_min", "log_df_other_mean", "frac_orphan", "decay_max", "decay_mean",
         "log_n_ents", "n_urls", "n_hosts", "is_code",
         "log_retired_other_docs_max", "retire_rate_max", "retired_same_doc",
         "frac_ents_retired_elsewhere", "recent_retire"]


def metrics(y, s, d):
    n = len(y); k = max(1, int(round(0.05 * n)))
    order = np.argsort(-s)
    topk = y[order[:k]]
    k10 = max(1, int(round(0.10 * n)))
    return dict(auroc=roc_auc_score(y, s), ap=average_precision_score(y, s),
                p_at_5pct=topk.mean(), recall_at_10pct=y[order[:k10]].sum() / y.sum())


def boot_ci(y, s, docs, fn=roc_auc_score, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    ud = np.unique(docs); idx = {d: np.where(docs == d)[0] for d in ud}
    vals = []
    for _ in range(B):
        pick = rng.choice(ud, len(ud), replace=True)
        ii = np.concatenate([idx[d] for d in pick])
        if y[ii].min() == y[ii].max():
            continue
        vals.append(fn(y[ii], s[ii]))
    return np.percentile(vals, [2.5, 97.5])


def boot_diff(y, s1, s2, docs, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    ud = np.unique(docs); idx = {d: np.where(docs == d)[0] for d in ud}
    diffs = []
    for _ in range(B):
        pick = rng.choice(ud, len(ud), replace=True)
        ii = np.concatenate([idx[d] for d in pick])
        if y[ii].min() == y[ii].max():
            continue
        diffs.append(roc_auc_score(y[ii], s1[ii]) - roc_auc_score(y[ii], s2[ii]))
    diffs = np.array(diffs)
    return np.percentile(diffs, [2.5, 97.5]), (diffs <= 0).mean()


y = te.y.values; docs = te.doc.values
res, scores = {}, {}
rng = np.random.default_rng(42)
scores["Random"] = rng.random(len(te))
scores["Line age (git blame)"] = te.line_age.values
scores["Document age"] = te.doc_age.values
scores["Reference confirmation age (single signal)"] = te.ent_conf_age_max.values
scores["Reference decay (single signal)"] = te.decay_max.values
scores["Link count (single signal)"] = te.n_urls.values + 1e-6 * rng.random(len(te))
scores["Change propagation (single signal)"] = (te.retire_rate_max + te.retired_same_doc).values

feats = {"Age + history (LR)": AGE + HIST,
         "RunbookGraph signals only (LR)": GRAPH,
         "RunbookGraph + age + history (LR)": GRAPH + AGE + HIST,
         "RunbookGraph + age + history (GBDT)": GRAPH + AGE + HIST}
for name, cols in feats.items():
    if "GBDT" in name:
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                           class_weight="balanced", random_state=0)
    else:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    m.fit(tr[cols].values, tr.y.values)
    scores[name] = m.predict_proba(te[cols].values)[:, 1]
    if name == "RunbookGraph + age + history (LR)":
        coefs = dict(zip(cols, m[-1].coef_[0].round(3)))

for k, s in scores.items():
    r = metrics(y, s, te)
    r["auroc_ci"] = boot_ci(y, s, docs).round(3).tolist()
    res[k] = {a: (round(float(b), 3) if not isinstance(b, list) else b) for a, b in r.items()}

best = "RunbookGraph + age + history (LR)"
ci, p = boot_diff(y, scores[best], scores["Line age (git blame)"], docs)
summary = dict(scope=SCOPE,
               train=dict(n=int(len(tr)), pos=int(tr.y.sum()), docs=int(tr.doc.nunique())),
               test=dict(n=int(len(te)), pos=int(te.y.sum()), docs=int(te.doc.nunique()),
                         base_rate=round(float(y.mean()), 4)),
               results=res,
               diff_best_vs_lineage=dict(ci=ci.round(3).tolist(), p_one_sided=round(float(p), 4)),
               lr_coefficients=coefs)
print(json.dumps(summary, indent=1))
json.dump(summary, open(f"results_{SCOPE}.json", "w"), indent=1)
te.assign(score=scores[best]).to_csv(f"scored_{SCOPE}.csv", index=False)
