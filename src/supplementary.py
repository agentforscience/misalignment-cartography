"""Post-hoc follow-ups to multiverse.py (not in the pre-specified plan; reported as exploratory).

Reads results/evaluations/{multiverse_cells,reference_effects}.csv, writes
results/evaluations/supp_*.csv. Usage: python src/supplementary.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "evaluations"
ALPHA = 0.05
METRICS = ["sample", "greedy", "prob", "logit"]


def main():
    cells = pd.read_csv(OUT / "multiverse_cells.csv")
    cells = cells[cells.alpha == ALPHA]
    ref = pd.read_csv(OUT / "reference_effects.csv")
    short = lambda m: m.split("-")[1]
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)

    # (a) metric x model null rate, non-placebo interventions
    mm = cells[~cells.placebo].pivot_table(index="model", columns="metric", values="null_rate")[METRICS]
    mm["greedy_minus_logit"] = mm.greedy - mm.logit
    mm.index = mm.index.map(short)
    mm.to_csv(OUT / "supp_metric_by_model.csv")
    print("=== null rate, metric x model (non-placebo) ===\n", mm)

    # (b) H4 sensitivity: reference classes from unadjusted p-values instead of Holm
    ref["ref_class_unadj"] = np.where(ref.p_greedy < ALPHA, "behavioural",
                                      np.where(ref.p_logit < ALPHA, "latent_only", "none"))
    c2 = cells.merge(ref[["model", "condition", "ref_class_unadj"]], on=["model", "condition"])
    rows = []
    for col in ("ref_class", "ref_class_unadj"):
        d = c2[~c2.placebo]
        for cls in ("behavioural", "latent_only", "none"):
            k = d[col] == cls
            rows.append({"classification": col, "ref_class": cls,
                         "n_units": d[k][["model", "condition"]].drop_duplicates().shape[0],
                         "share_of_nulls": d[k].null_rate.sum() / d.null_rate.sum(),
                         "share_of_nulls_greedy_metric": d[k & (d.metric == "greedy")].null_rate.sum()
                         / d[d.metric == "greedy"].null_rate.sum()})
    sens = pd.DataFrame(rows)
    sens.to_csv(OUT / "supp_h4_sensitivity.csv", index=False)
    print("\n=== H4 sensitivity to reference-class threshold (non-placebo) ===\n", sens)

    # (c) baseline margin vs how often a significant logit shift flips answers
    r = ref[~ref.placebo].copy()
    r["flip_rate"] = (r.flips_down + r.flips_up) / 240
    lat = r.groupby("model").agg(mean_abs_logit_delta=("logit_delta", lambda x: x.abs().mean()),
                                 mean_flip_rate=("flip_rate", "mean"), greedy_base=("greedy_base", "first"),
                                 n_latent_only=("ref_class", lambda x: (x == "latent_only").sum()),
                                 n_behavioural=("ref_class", lambda x: (x == "behavioural").sum()))
    lat.index = lat.index.map(short)
    lat.to_csv(OUT / "supp_latent_vs_flips.csv")
    print("\n=== logit shift vs answer flips, by model (non-placebo) ===\n", lat)

    # (d) sign disagreement between metrics on the full pool
    r["sign_disagree"] = (np.sign(r.logit_delta) != np.sign(r.greedy_delta)) & (r.greedy_delta != 0)
    print("\n=== units where mean logit shift and greedy-rate change have opposite signs ===\n",
          r[r.sign_disagree][["model", "condition", "logit_delta", "greedy_delta", "flips_down", "flips_up"]])
    r[r.sign_disagree].to_csv(OUT / "supp_sign_disagreement.csv", index=False)

    # (e) per-domain consistency of the reference effect (paired d of the logit shift)
    dom = ref[["model", "condition", "dz", "dz_nlp", "dz_phil", "dz_pol"]].copy()
    dom["same_sign_all_domains"] = (np.sign(dom[["dz_nlp", "dz_phil", "dz_pol"]]).nunique(axis=1) == 1)
    dom.to_csv(OUT / "supp_per_domain_dz.csv", index=False)
    print("\n=== per-domain: share of units with same-sign effect in all 3 domains ===\n",
          dom.groupby(ref.ref_class).same_sign_all_domains.agg(["mean", "size"]))

    # (f) direction of significant results in placebo cells
    p = pd.read_csv(OUT / "multiverse_cells.csv")
    p = p[(p.alpha == ALPHA) & p.placebo]
    print("\n=== placebo cells: share of studies significant, by direction ===\n",
          p.groupby("metric")[["sig_decrease", "sig_increase"]].mean().reindex(METRICS))
    p.groupby("metric")[["sig_decrease", "sig_increase"]].mean().reindex(METRICS).to_csv(OUT / "supp_placebo_direction.csv")


if __name__ == "__main__":
    main()
