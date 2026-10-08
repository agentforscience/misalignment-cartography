"""Experiments 2-4: multiverse of simulated studies, clustering analysis, null classification.

Reads results/model_outputs/*.csv (real model outputs from run_interventions.py) and writes
results/evaluations/*.csv|json and figures/*.png.

Usage: python src/multiverse.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "evaluations"
FIG = ROOT / "figures"
SEED = 42
R = 1000                      # simulated studies per cell
NS = (20, 50, 100, 200)       # study sample sizes
ALPHA = 0.05
METRICS = ("sample", "greedy", "prob", "logit")
BINARY = ("sample", "greedy")
# models are included if their run_interventions.py output exists
MODELS = tuple(m for m in ("Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct",
                           "Qwen2.5-7B-Instruct")
               if (ROOT / "results" / "model_outputs" / f"{m}.csv").exists())
MODEL_LABEL = {m: m.split("-")[1] for m in MODELS}
GRANS = ("prompt", "all_layer", "single_layer", "head")
N_PERM = 5000
N_BOOT = 2000

# reference palette (dataviz skill): categorical slots, sequential blue ramp, inks
PAL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "surface": "#fcfcfb"}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fd", "#cde2fb", "#86b6ef", "#2a78d6", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": PAL["surface"], "axes.facecolor": PAL["surface"], "savefig.facecolor": PAL["surface"],
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": PAL["ink2"], "xtick.color": PAL["muted"],
    "ytick.color": PAL["muted"], "text.color": PAL["ink"], "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": PAL["grid"], "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
})


# ------------------------------------------------------------------ data
def load():
    """Per-item metric values for every model x condition, as arrays aligned on item order."""
    data, info = {}, None
    for m in MODELS:
        d = pd.read_csv(ROOT / "results" / "model_outputs" / f"{m}.csv", keep_default_na=False)
        isA = (d.syc == "A").values
        d["logit"] = np.where(isA, d.logit_A - d.logit_B, d.logit_B - d.logit_A)
        d["p_syc_full"] = np.exp(np.where(isA, d.logp_A, d.logp_B))   # full-vocab P(syc letter)
        d["mass"] = np.exp(d.logp_A) + np.exp(d.logp_B)
        d["prob"] = d.p_syc_full / d.mass
        d["greedy"] = (d.greedy.str.strip() == d.syc).astype(float)
        d["sample_recorded"] = (d.sampled.str.strip() == d.syc).astype(float)
        items = d[d.condition == "baseline"].item_id.tolist()
        for c, g in d.groupby("condition", sort=False):
            g = g.set_index("item_id").loc[items]
            data[(m, c)] = {k: g[k].values for k in ("logit", "prob", "greedy", "p_syc_full", "mass")}
            data[(m, c)]["domain"] = g.domain.values
        if info is None:
            info = d.drop_duplicates("condition")[["condition", "granularity", "placebo"]]
            info = info[info.condition != "baseline"].reset_index(drop=True)
    return data, info


# ------------------------------------------------------------------ tests
def mcnemar_table(nmax):
    """Exact two-sided McNemar p-value for every (b, discordant total m), m <= nmax."""
    tab = np.ones((nmax + 1, nmax + 1))
    for m in range(1, nmax + 1):
        b = np.arange(m + 1)
        tab[: m + 1, m] = np.minimum(1.0, 2 * stats.binom.cdf(np.minimum(b, m - b), m, 0.5))
    return tab


def paired_t_p(d):
    """Two-sided paired t-test p-values for an array of studies (rows = studies)."""
    n = d.shape[1]
    sd = d.std(1, ddof=1)
    t = np.divide(d.mean(1), sd / np.sqrt(n), out=np.zeros(len(d)), where=sd > 0)
    p = 2 * stats.t.sf(np.abs(t), n - 1)
    return np.where(sd > 0, p, 1.0)   # no variation at all -> cannot reject


def simulate(data, info):
    """Null rate for every (model, intervention, metric, n) cell.

    A study resamples n items with replacement. For the `sample` metric each study additionally
    draws a fresh T=1 answer per item from the model's own full-vocabulary next-token
    distribution (Bernoulli with the recorded P(syc letter)), for baseline and intervention.
    Item index draws are shared across cells (common random numbers).
    """
    rng = np.random.default_rng(SEED)
    tab = mcnemar_table(max(NS))
    n_items = len(data[(MODELS[0], "baseline")]["logit"])
    idx = {n: rng.integers(0, n_items, (R, n)) for n in NS}
    rows = []
    for m in MODELS:
        base = data[(m, "baseline")]
        for _, r in info.iterrows():
            cur = data[(m, r.condition)]
            for n in NS:
                ii = idx[n]
                for met in METRICS:
                    if met == "sample":
                        a = rng.random(ii.shape) < base["p_syc_full"][ii]
                        b = rng.random(ii.shape) < cur["p_syc_full"][ii]
                    else:
                        a, b = base[met][ii], cur[met][ii]
                    if met in BINARY:
                        dec = ((a == 1) & (b == 0)).sum(1)   # intervention removed a sycophantic answer
                        inc = ((a == 0) & (b == 1)).sum(1)
                        p = tab[dec, dec + inc]
                        sign = np.sign(inc - dec)
                    else:
                        d = b - a
                        p = paired_t_p(d)
                        sign = np.sign(d.mean(1))
                    for alpha in (0.05, 0.01):
                        rows.append({
                            "model": m, "condition": r.condition, "granularity": r.granularity,
                            "placebo": bool(r.placebo), "metric": met, "n": n, "alpha": alpha,
                            "null_rate": float((p >= alpha).mean()),
                            "sig_decrease": float(((p < alpha) & (sign < 0)).mean()),
                            "sig_increase": float(((p < alpha) & (sign > 0)).mean()),
                        })
    return pd.DataFrame(rows)


def reference_effects(data, info):
    """Full-pool (n=240) effect of each intervention; Holm-corrected; three reference classes."""
    rng = np.random.default_rng(SEED)
    rows = []
    for m in MODELS:
        base = data[(m, "baseline")]
        n = len(base["logit"])
        for _, r in info.iterrows():
            cur = data[(m, r.condition)]
            d = cur["logit"] - base["logit"]
            se = d.std(ddof=1) / np.sqrt(n)
            g = cur["greedy"] - base["greedy"]
            dec, inc = int((g < 0).sum()), int((g > 0).sum())
            p_g = stats.binomtest(dec, dec + inc, 0.5).pvalue if dec + inc else 1.0
            gb = g[rng.integers(0, n, (N_BOOT, n))].mean(1)
            row = {
                "model": m, "condition": r.condition, "granularity": r.granularity,
                "placebo": bool(r.placebo),
                "logit_delta": d.mean(), "logit_ci_lo": d.mean() - 1.97 * se,
                "logit_ci_hi": d.mean() + 1.97 * se, "dz": d.mean() / d.std(ddof=1),
                "p_logit": stats.ttest_1samp(d, 0).pvalue,
                "greedy_base": base["greedy"].mean(), "greedy_delta": g.mean(),
                "greedy_ci_lo": np.quantile(gb, 0.025), "greedy_ci_hi": np.quantile(gb, 0.975),
                "flips_down": dec, "flips_up": inc, "p_greedy": p_g,
                "prob_delta": (cur["prob"] - base["prob"]).mean(),
                "answer_mass": cur["mass"].mean(),
            }
            for dom in np.unique(base["domain"]):
                k = base["domain"] == dom
                row[f"dz_{dom}"] = d[k].mean() / d[k].std(ddof=1)
            rows.append(row)
    ref = pd.DataFrame(rows)
    ref["p_logit_holm"] = multipletests(ref.p_logit, method="holm")[1]
    ref["p_greedy_holm"] = multipletests(ref.p_greedy, method="holm")[1]
    ref["ref_class"] = np.where(ref.p_greedy_holm < ALPHA, "behavioural",
                                np.where(ref.p_logit_holm < ALPHA, "latent_only", "none"))
    ref["abs_dz"] = ref.dz.abs()
    return ref


def baseline_table(data):
    """Baseline sycophancy per model and domain, with a test against the 50% no-sycophancy rate."""
    rows = []
    for m in MODELS:
        b = data[(m, "baseline")]
        for dom in ["all"] + sorted(set(b["domain"])):
            k = np.ones(len(b["greedy"]), bool) if dom == "all" else b["domain"] == dom
            x, n = int(b["greedy"][k].sum()), int(k.sum())
            ci = stats.binomtest(x, n).proportion_ci()
            rows.append({"model": m, "domain": dom, "n": n, "greedy_syc_rate": x / n,
                         "ci_lo": ci.low, "ci_hi": ci.high, "p_vs_50": stats.binomtest(x, n, 0.5).pvalue,
                         "mean_prob_syc": b["prob"][k].mean(), "mean_logit_margin": b["logit"][k].mean(),
                         "answer_mass": b["mass"][k].mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ clustering analysis
def marginals(cells, feature, rng):
    """Marginal null rate per level with a cluster bootstrap over model x intervention units."""
    units = cells[["model", "condition"]].drop_duplicates().values
    piv = {(m, c): g for (m, c), g in cells.groupby(["model", "condition"])}
    levels = list(pd.unique(cells[feature]))
    boots = []
    for _ in range(N_BOOT):
        pick = units[rng.integers(0, len(units), len(units))]
        bs = pd.concat([piv[tuple(u)] for u in pick])
        boots.append(bs.groupby(feature).null_rate.mean().reindex(levels).values)
    boots = np.array(boots, float)
    est = cells.groupby(feature).null_rate.mean().reindex(levels)
    return pd.DataFrame({"feature": feature, "level": [str(l) for l in levels], "null_rate": est.values,
                         "ci_lo": np.nanquantile(boots, 0.025, 0), "ci_hi": np.nanquantile(boots, 0.975, 0),
                         "n_cells": cells.groupby(feature).size().reindex(levels).values})


def perm_within(cells, feature, other, rng):
    """Permutation test for a within-unit factor: shuffle its labels inside each
    (model, intervention, `other`) block. Statistic: range of the marginal null rates."""
    wide = cells.pivot_table(index=["model", "condition", other], columns=feature, values="null_rate").values
    obs = np.ptp(wide.mean(0))
    hits = 0
    for _ in range(N_PERM):
        perm = np.take_along_axis(wide, rng.random(wide.shape).argsort(1), 1)
        hits += np.ptp(perm.mean(0)) >= obs - 1e-12
    return {"feature": feature, "statistic_range": float(obs), "p_perm": (hits + 1) / (N_PERM + 1)}


def perm_between(cells, key, feature, rng):
    """Permutation test for a between-unit factor: shuffle `feature` labels across the levels of `key`."""
    mean_by_key = cells.groupby(key).null_rate.mean()
    lab = cells.drop_duplicates(key).set_index(key)[feature].reindex(mean_by_key.index).values
    stat = lambda l: np.ptp(pd.Series(mean_by_key.values).groupby(l).mean().values)
    obs = stat(lab)
    hits = sum(stat(rng.permutation(lab)) >= obs - 1e-12 for _ in range(N_PERM))
    return {"feature": feature, "statistic_range": float(obs), "p_perm": (hits + 1) / (N_PERM + 1),
            "n_units": int(len(lab))}


def variance_shares(cells):
    """Share of cell-level null-rate variance attributed to each feature (type-II ANOVA),
    without and with the reference effect size as a covariate."""
    d = cells.copy()
    d["logn"] = np.log(d.n)
    out = []
    base = "null_rate ~ C(metric) + C(n) + C(granularity)" + (" + C(model)" if d.model.nunique() > 1 else "")
    specs = {"design_only": base, "plus_effect_size": base + " + abs_dz + I(abs_dz**2)"}
    for name, f in specs.items():
        fit = smf.ols(f, d).fit()
        a = sm.stats.anova_lm(fit, typ=2)
        tot = ((d.null_rate - d.null_rate.mean()) ** 2).sum()
        for term, row in a.iterrows():
            out.append({"spec": name, "term": term, "df": row.df, "share_of_total_ss": row.sum_sq / tot,
                        "F": row.F, "p": row["PR(>F)"], "r2_model": fit.rsquared})
    return pd.DataFrame(out)


def null_composition(cells, rng, mask=None, label="non_placebo"):
    """Which reference class do nulls fall in? Shares weighted by cell null rate, cluster bootstrap CI."""
    d = cells if mask is None else cells[mask]
    classes = ("behavioural", "latent_only", "none")
    share = lambda x: np.array([x[x.ref_class == c].null_rate.sum() for c in classes]) / x.null_rate.sum()
    units = d[["model", "condition"]].drop_duplicates().values
    piv = {(m, c): g for (m, c), g in d.groupby(["model", "condition"])}
    boots = np.array([share(pd.concat([piv[tuple(u)] for u in units[rng.integers(0, len(units), len(units))]]))
                      for _ in range(N_BOOT)])
    est = share(d)
    return pd.DataFrame({"subset": label, "ref_class": classes, "share_of_nulls": est,
                         "ci_lo": np.quantile(boots, 0.025, 0), "ci_hi": np.quantile(boots, 0.975, 0),
                         "n_units_in_class": [d[d.ref_class == c][["model", "condition"]].drop_duplicates().shape[0]
                                              for c in classes],
                         "mean_null_rate_in_class": [d[d.ref_class == c].null_rate.mean() for c in classes]})


# ------------------------------------------------------------------ figures
def fig_map(cells, info):
    """The map: null rate for every design cell."""
    conds = info.condition.tolist()
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 4.6), sharey=True)
    for ax, m in zip(axes, MODELS):
        piv = cells[cells.model == m].pivot_table(index="condition", columns=["metric", "n"], values="null_rate")
        piv = piv.reindex(conds)[[(k, n) for k in METRICS for n in NS]]
        im = ax.imshow(piv.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:.2f}"[1:] if v < 0.995 else "1.", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.6 else PAL["ink"])
        ax.set_xticks(range(16), [str(n) for _ in METRICS for n in NS], fontsize=7)
        for k, met in enumerate(METRICS):
            ax.text(1.5 + 4 * k, -0.9, met, ha="center", fontsize=9, color=PAL["ink2"])
            if k:
                ax.axvline(4 * k - 0.5, color=PAL["surface"], lw=3)
        bounds = np.cumsum(info.groupby("granularity", sort=False).size().values)[:-1]
        for b in bounds:
            ax.axhline(b - 0.5, color=PAL["surface"], lw=3)
        ax.set_yticks(range(len(conds)), [c + (" (placebo)" if p else "") for c, p in zip(conds, info.placebo)])
        ax.set_title(f"Qwen2.5-{MODEL_LABEL[m]}", pad=22)
        ax.set_xlabel("study sample size n, within metric")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, label="null rate (share of studies with p >= .05)")
    fig.suptitle("Null rate by design cell: intervention (rows) x metric and n (columns) x model (panels)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.04)
    fig.savefig(FIG / "fig1_null_map.png")
    plt.close(fig)


def fig_marginals(marg):
    """Marginal null rate along each design feature."""
    feats = [("metric", "Metric"), ("n", "Sample size n"), ("granularity", "Intervention granularity"),
             ("model", "Model scale")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, (f, title) in zip(axes, feats):
        d = marg[marg.feature == f]
        lab = [MODEL_LABEL.get(l, l) for l in d.level]
        ax.bar(lab, d.null_rate, color=PAL["blue"], width=0.55)
        ax.errorbar(lab, d.null_rate, yerr=[d.null_rate - d.ci_lo, d.ci_hi - d.null_rate], fmt="none",
                    ecolor=PAL["ink2"], lw=1, capsize=2)
        for x, v in enumerate(d.null_rate):
            ax.text(x, 0.02, f"{v:.2f}", ha="center", color="white", fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("mean null rate")
    fig.suptitle("Marginal null rate by design feature (bars: 95% cluster-bootstrap CI over model x intervention)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", y=1.05)
    fig.savefig(FIG / "fig2_marginal_null_rates.png")
    plt.close(fig)


def fig_reference(ref, info):
    """Reference effect of each intervention on the full pool, per model."""
    conds = info.condition.tolist()[::-1]
    cols = [PAL["blue"], PAL["orange"], PAL["aqua"], PAL["yellow"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for k, m in enumerate(MODELS):
        d = ref[ref.model == m].set_index("condition").reindex(conds)
        y = np.arange(len(conds)) + (k - (len(MODELS) - 1) / 2) * 0.19
        for ax, v, lo, hi in ((axes[0], "logit_delta", "logit_ci_lo", "logit_ci_hi"),
                              (axes[1], "greedy_delta", "greedy_ci_lo", "greedy_ci_hi")):
            ax.errorbar(d[v], y, xerr=[d[v] - d[lo], d[hi] - d[v]], fmt="o", ms=5, color=cols[k], lw=1.4,
                        label=f"Qwen2.5-{MODEL_LABEL[m]}")
    for ax, t in zip(axes, ("Change in sycophancy logit margin", "Change in greedy sycophantic-answer rate")):
        ax.axvline(0, color=PAL["ink2"], lw=0.8)
        ax.set_title(t)
        ax.set_yticks(range(len(conds)), conds)
        ax.grid(axis="y", visible=False)
    big = ref[["logit_ci_lo", "logit_ci_hi"]].abs().max().max() > 4   # compress only if a few effects dwarf the rest
    if big:
        axes[0].set_xscale("symlog", linthresh=1)
        axes[0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axes[0].set_xlabel("intervention - baseline (logits%s), 95%% CI" % (", symlog axis" if big else ""))
    axes[1].set_xlabel("intervention - baseline (proportion of 240 items), 95% bootstrap CI")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=len(MODELS), fontsize=8, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Reference effects on the full 240-item pool", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", y=1.0)
    fig.savefig(FIG / "fig3_reference_effects.png")
    plt.close(fig)


def fig_power(cells):
    """Null rate against reference effect size, by metric: effect-side clustering collapses onto |d|."""
    cols = dict(zip(METRICS, [PAL["yellow"], PAL["orange"], PAL["aqua"], PAL["blue"]]))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, n in zip(axes, NS):
        for met in METRICS:
            d = cells[(cells.n == n) & (cells.metric == met)].sort_values("abs_dz")
            ax.plot(d.abs_dz, d.null_rate, "o", ms=4.5, color=cols[met], label=met, mec=PAL["surface"], mew=0.6)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_title(f"n = {n}")
        ax.set_xlabel("|paired d| of logit shift on full pool")
    axes[0].set_ylabel("null rate")
    axes[-1].legend(title="metric", fontsize=8, loc="upper right")
    fig.suptitle("Null rate vs reference effect size (each point: one model x intervention)", x=0.01,
                 ha="left", fontsize=10, fontweight="bold", y=1.05)
    fig.savefig(FIG / "fig4_null_vs_effect_size.png")
    plt.close(fig)


def fig_composition(cells):
    """Expected nulls per metric, split by reference class (non-placebo interventions)."""
    d = cells[~cells.placebo]
    classes = [("behavioural", "real behavioural effect (missed)", PAL["orange"]),
               ("latent_only", "logit shift only", PAL["yellow"]),
               ("none", "no detectable effect", PAL["blue"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bottom = np.zeros(len(METRICS))
    for cls, lab, col in classes:
        v = np.array([d[(d.metric == m) & (d.ref_class == cls)].null_rate.sum() for m in METRICS])
        v = v / np.array([len(d[d.metric == m]) for m in METRICS])
        ax.bar(METRICS, v, bottom=bottom, color=col, label=lab, width=0.55, edgecolor=PAL["surface"], lw=2)
        bottom += v
    ax.set_ylabel("null rate (non-placebo cells)")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(title="reference class of the intervention", fontsize=8, loc="upper right")
    ax.set_title("What the nulls are made of, by metric", loc="left")
    fig.savefig(FIG / "fig5_null_composition.png")
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    data, info = load()
    n_items = len(data[(MODELS[0], "baseline")]["logit"])

    base = baseline_table(data)
    base.to_csv(OUT / "baseline_sycophancy.csv", index=False)
    ref = reference_effects(data, info)
    ref.to_csv(OUT / "reference_effects.csv", index=False)

    sims = simulate(data, info).merge(ref[["model", "condition", "ref_class", "dz", "abs_dz"]],
                                      on=["model", "condition"])
    sims.to_csv(OUT / "multiverse_cells.csv", index=False)
    cells = sims[sims.alpha == ALPHA].copy()
    assert len(cells) == len(MODELS) * len(info) * len(METRICS) * len(NS)

    marg = pd.concat([marginals(cells, f, rng) for f in ("metric", "n", "granularity", "model")])
    marg_np = pd.concat([marginals(cells[~cells.placebo], f, rng) for f in ("metric", "n", "granularity", "model")])
    marg.to_csv(OUT / "marginal_null_rates.csv", index=False)
    marg_np.to_csv(OUT / "marginal_null_rates_non_placebo.csv", index=False)

    tests = [perm_within(cells, "metric", "n", rng), perm_within(cells, "n", "metric", rng),
             perm_between(cells, "condition", "granularity", rng),
             perm_between(cells[~cells.placebo], "condition", "granularity", rng) | {"note": "non-placebo only"}]
    pd.DataFrame(tests).to_csv(OUT / "permutation_tests.csv", index=False)

    var = variance_shares(cells)
    var.to_csv(OUT / "variance_shares.csv", index=False)

    comp = pd.concat([
        null_composition(cells, rng, ~cells.placebo, "non_placebo"),
        null_composition(cells, rng, None, "all_interventions"),
        null_composition(cells, rng, ~cells.placebo & (cells.n > 20), "non_placebo_n>20"),
        null_composition(sims[sims.alpha == 0.01], rng, ~sims[sims.alpha == 0.01].placebo, "non_placebo_alpha.01"),
    ] + [null_composition(cells, rng, ~cells.placebo & (cells.metric == m), f"non_placebo_{m}") for m in METRICS])
    comp.to_csv(OUT / "null_composition.csv", index=False)

    # metric contrast inside units with a real behavioural effect; placebo false-positive rates
    beh = cells[cells.ref_class == "behavioural"].pivot_table(index="metric", columns="n", values="null_rate")
    plc = cells[cells.ref_class == "none"].pivot_table(index="metric", columns="n", values="null_rate")
    beh.to_csv(OUT / "null_rate_behavioural_units.csv")
    plc.to_csv(OUT / "null_rate_no_effect_units.csv")
    rob = sims.groupby(["alpha", "metric"]).null_rate.mean().unstack()
    rob.to_csv(OUT / "robustness_alpha.csv")

    fig_map(cells, info)
    fig_marginals(marg)
    fig_reference(ref, info)
    fig_power(cells)
    fig_composition(cells)

    summary = {
        "n_items": n_items, "n_cells": int(len(cells)), "studies_per_cell": R, "alpha": ALPHA,
        "overall_null_rate": float(cells.null_rate.mean()),
        "overall_null_rate_non_placebo": float(cells[~cells.placebo].null_rate.mean()),
        "ref_class_counts": ref.ref_class.value_counts().to_dict(),
        "ref_class_counts_placebo": ref[ref.placebo].ref_class.value_counts().to_dict(),
        "seed": SEED, "n_perm": N_PERM, "n_boot": N_BOOT,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    for title, obj in [("BASELINE", base), ("REFERENCE EFFECTS", ref[[
            "model", "condition", "placebo", "logit_delta", "dz", "p_logit_holm", "greedy_base", "greedy_delta",
            "flips_down", "flips_up", "p_greedy_holm", "answer_mass", "ref_class"]]),
            ("MARGINALS (all)", marg), ("MARGINALS (non-placebo)", marg_np), ("PERMUTATION TESTS", pd.DataFrame(tests)),
            ("VARIANCE SHARES", var), ("NULL COMPOSITION", comp), ("NULL RATE, behavioural units", beh),
            ("NULL RATE, no-effect units", plc), ("ROBUSTNESS alpha", rob), ("SUMMARY", summary)]:
        print(f"\n=== {title} ===\n{obj}")


if __name__ == "__main__":
    main()
