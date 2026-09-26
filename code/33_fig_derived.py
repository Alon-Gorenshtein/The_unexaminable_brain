"""Figure 2 (file fig6_derived): the default total of 15 against what the charted components allow.
(A) Observed eye-plus-motor sum among non-assessable lowest-eye-plus-motor examinations: a total of 15 needs 10.
(B) Stays by SOFA CNS category, official first-day derivation versus the Brennan estimate.
(C) Mortality-model AUROC, official derivation versus the Brennan estimate.
Every plotted or annotated value is read from output/revision; nothing numeric is typed here."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from config import FIG, REV

BLUE, GREY, SALMON, LABEL = "#00468B", "#ADB6B6", "#FDAF91", "#2B2B2B"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": "#9CA3AF", "axes.linewidth": 0.8})
F = json.loads((REV / "feasible.json").read_text())
A = pd.read_csv(REV / "audit_by_rule.csv")
AJ = json.loads((REV / "audit.json").read_text())
dist = pd.read_csv(REV / "feasible_em_distribution.csv")
ct = pd.read_csv(REV / "sofa_crosstab.csv", index_col=0)      # per-stay first-day minimum: Brennan estimate (rows) vs official (columns)


def panel(ax, lab):
    ax.text(-0.16, 1.06, lab, transform=ax.transAxes, fontsize=14, fontweight="bold", color=LABEL)


def faint_grid(ax):
    ax.grid(axis="y", color="#F3F4F6", lw=0.6, zorder=0)


fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.5))

# (a) what the charted components allow
ax[0].bar(dist.em, dist.n, color=[GREY if ok else SALMON for ok in dist.default_15_feasible.astype(bool)],
          edgecolor="white", zorder=3)
ax[0].set_xticks(dist.em)
ax[0].set_xlabel("Eye-plus-motor sum (observed)")
ax[0].set_ylabel("Non-assessable stays\n(lowest eye-plus-motor examination)")
pct_no = F["lowest_em_na_impossible_pct"]
thr = int(dist.em[dist.default_15_feasible.astype(bool)].min())     # lowest eye plus motor total that allows a default of 15
ax[0].legend(handles=[Patch(facecolor=SALMON, edgecolor="white", label=f"Below {thr}: total of 15\nnot possible ({pct_no:.1f}%)"),
                      Patch(facecolor=GREY, edgecolor="white", label=f"{thr} or above: total of 15\npossible ({100 - pct_no:.1f}%)")],
             frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.03, 1.0))
panel(ax[0], "A")
faint_grid(ax[0])

# (b) SOFA CNS category under the two derivations
cats = np.arange(5)
est = ct.sum(axis=1).reindex(cats, fill_value=0).values
der = ct.sum(axis=0).rename(lambda c: int(str(c).split("_")[-1])).reindex(cats, fill_value=0).values
x, wd = cats, 0.38
ax[1].bar(x - wd / 2, 100 * est / est.sum(), wd, color=BLUE, edgecolor="white", label="Brennan estimate", zorder=3)
ax[1].bar(x + wd / 2, 100 * der / der.sum(), wd, color=SALMON, edgecolor="white", label="Official first-day GCS", zorder=3)
ax[1].set_xticks(x)
ax[1].set_xticklabels([str(c) for c in cats])
ax[1].set_xlabel("SOFA CNS category")
ax[1].set_ylabel("Stays (%)")
ax[1].legend(frameon=False, fontsize=9)
panel(ax[1], "B")
faint_grid(ax[1])

# (c) mortality-model discrimination
r = A[(A.rule == "lowest_em") & A.slug.isin(["official", "brennan"])].set_index("slug")
order = ["official", "brennan"]
auc = r.loc[order, "auroc"].values
lo = r.loc[order, "lo"].values
hi = r.loc[order, "hi"].values
ax[2].bar([0, 1], auc, color=[SALMON, BLUE], edgecolor="white", width=0.6, zorder=3,
          yerr=[auc - lo, hi - auc], capsize=4, error_kw=dict(ecolor=LABEL, lw=1))
ax[2].set_ylim(0.5, hi.max() + 0.28 * (hi.max() - 0.5))      # 0.5 = chance; headroom for the annotation from the data
ax[2].set_xticks([0, 1])
ax[2].set_xticklabels(["Official\nfirst-day GCS", "Brennan\nestimate"], fontsize=9)
ax[2].set_xlabel("Rule")
ax[2].set_ylabel("Mortality-model AUROC")
ax[2].text(0.5, 0.97, f"Paired difference {AJ['paired_brennan_minus_official_mean']:.3f}\n"
           f"(95% CI {AJ['paired_brennan_minus_official_lo']:.3f} to {AJ['paired_brennan_minus_official_hi']:.3f})",
           transform=ax[2].transAxes, ha="center", va="top", fontsize=9, color=LABEL)
panel(ax[2], "C")
faint_grid(ax[2])

fig.subplots_adjust(wspace=0.42)
for e in ("pdf", "png"):
    fig.savefig(FIG / f"fig6_derived.{e}", dpi=600, bbox_inches="tight", pad_inches=0.15)
plt.close(fig)
print("fig6_derived done")
