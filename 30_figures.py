import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from config import INT, TAB, FIG
from lib_style import set_style, save, PHENO_COLORS, PHENO_ORDER, STRAT_LABELS

set_style()
NEURO = [p for p in PHENO_ORDER if p != "comparator"]

# ---------- Figure 2: assessability curves ----------
g = pd.read_parquet(INT / "assessability_curve.parquet")
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for ph in PHENO_ORDER:
    d = g[(g.phenotype == ph) & (g.hour <= 168)].sort_values("hour")
    d = d[d.n_at_risk >= 20]
    if len(d):
        # 6-hour centered rolling mean to suppress per-bin sampling noise
        ys = (100 * d.frac_assessable).rolling(6, center=True, min_periods=1).mean()
        ax.plot(d.hour, ys, color=PHENO_COLORS[ph], label=ph, lw=1.8, alpha=0.95)
ax.set_xlabel("Hours since ICU admission")
ax.set_ylabel("Verbal exam assessable (%)")
ax.set_xlim(0, 168); ax.set_ylim(0, 100)
ax.axvspan(0, 72, color="#eeeeee", zorder=0)
ax.legend(ncol=2, frameon=False, loc="lower right")
ax.set_title("Recovery of neurological exam assessability by injury phenotype")
save(fig, FIG / "fig2_assessability")
print("fig2 done")

# ---------- Figure 3: concordance bars ----------
ct = pd.read_csv(TAB / "table_concordance.csv", index_col=0).reindex(NEURO)
fig, ax = plt.subplots(figsize=(5.2, 3.2))
x = np.arange(len(ct)); w = 0.26
for i, (col, lab, c) in enumerate([("sedative", "Sedative", "#0072B2"),
                                   ("vent", "Ventilation", "#E69F00"),
                                   ("nmb", "NMB", "#D55E00")]):
    ax.bar(x + (i - 1) * w, ct[col], w, label=lab, color=c)
ax.set_xticks(x); ax.set_xticklabels(ct.index)
ax.set_ylabel("Non-assessable exams with\nconcurrent exposure (%)")
ax.set_ylim(0, 100); ax.legend(frameon=False, ncol=3)
ax.set_title("Drivers of exam non-assessability")
save(fig, FIG / "fig3_concordance")
print("fig3 done")

# ---------- Figure 4: Aim-3 audit ----------
res = pd.read_csv(TAB / "table3_aim3_by_phenotype.csv")
alln = res[res.phenotype == "ALL"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
strord = ["drop", "impute1", "gcs_t", "kramer"]
xs = np.arange(len(strord))
a = alln.set_index("strategy").reindex(strord)
axes[0].bar(xs, a["auroc"], color="#0072B2")
axes[0].set_ylim(0.5, max(0.8, a["auroc"].max() + 0.03))
axes[0].set_xticks(xs); axes[0].set_xticklabels([STRAT_LABELS[s] for s in strord])
axes[0].set_ylabel("AUROC (out-of-fold)"); axes[0].set_title("Discrimination")
axes[1].bar(xs, a["n_excluded"], color="#D55E00")
axes[1].set_xticks(xs); axes[1].set_xticklabels([STRAT_LABELS[s] for s in strord])
axes[1].set_ylabel("Patients excluded"); axes[1].set_title("Cohort lost to handling choice")
fig.suptitle("How the GCS handling choice changes the mortality model", y=1.02)
save(fig, FIG / "fig4_audit")
print("fig4 done")
