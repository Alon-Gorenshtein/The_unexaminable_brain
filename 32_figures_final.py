"""Journal-grade figures (scientific-figures skill standard): Helvetica, colorblind-safe,
600 DPI PDF+PNG, no in-figure titles, panel labels, error bars."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np, pandas as pd
from config import INT, TAB, FIG

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": "#9CA3AF", "axes.linewidth": 0.8,
})
BLUE, GREY, SALMON = "#00468B", "#ADB6B6", "#FDAF91"
LABEL, MUTED = "#2B2B2B", "#6B7280"
# Okabe-Ito colorblind-safe palette for phenotypes
PH = {"TBI": "#E69F00", "ICH": "#D55E00", "SAH": "#CC79A7", "SDH": "#0072B2",
      "AIS": "#009E73", "anoxic": "#56B4E9", "comparator": "#999999"}
PHORD = ["TBI", "ICH", "SAH", "SDH", "AIS", "anoxic", "comparator"]
NEURO = PHORD[:-1]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def panel(ax, lab):
    ax.text(-0.14, 1.06, lab, transform=ax.transAxes, fontsize=14, fontweight="bold", color=LABEL)


# ---------------- Figure 1: study flow ----------------
def fig1():
    cohort = pd.read_parquet(INT / "cohort.parquet")
    first = cohort[cohort.first_stay]; neuro = first[first.phenotype != "comparator"]
    comp = first[first.phenotype == "comparator"]
    nan = (pd.read_parquet(INT / "burden_per_stay.parquet").phenotype != "comparator").sum()
    fig, ax = plt.subplots(figsize=(6.6, 5.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 12)

    def box(x, y, w, h, t, fc="#FFFFFF"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", lw=1.1,
                                    edgecolor=BLUE, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9, color=LABEL)

    def arr(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1))
    box(2.0, 10.4, 6.0, 1.1, f"MIMIC-IV v3.1 ICU stays\n(N = {len(cohort):,})", "#EAF2F8")
    arr(5, 10.4, 5, 9.6)
    box(2.0, 8.5, 6.0, 1.1, f"First ICU stay per hospitalization, age ≥ 18\n(N = {len(first):,})")
    ax.text(8.5, 9.05, "Excluded:\nrepeat ICU stays,\nage < 18", ha="left", va="center", fontsize=7.5, color=MUTED)
    arr(3.4, 8.5, 3.4, 7.7); arr(6.6, 8.5, 6.6, 7.7)
    box(0.3, 6.5, 5.1, 1.2, f"Acute brain injury\n(6 phenotypes; N = {len(neuro):,})", "#FDF0E8")
    box(5.9, 6.5, 3.8, 1.2, f"General-ICU comparator\n(N = {len(comp):,})", "#EEF1F5")
    arr(2.85, 6.5, 2.85, 5.7)
    box(0.3, 4.5, 5.1, 1.2, f"≥ 1 charted GCS-verbal entry\nAnalytic cohort (N = {nan:,})", "#FDF0E8")
    arr(2.85, 4.5, 2.85, 3.7)
    box(0.3, 2.4, 5.1, 1.2, "Aim 3-4 audit subset:\ncomplete worst first-day GCS\n(N = 12,404)", "#FDF0E8")
    save(fig, "fig1_flow")


# ---------------- Figure 2: assessability curves ----------------
def fig2():
    g = pd.read_parquet(INT / "assessability_curve.parquet")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axvspan(0, 72, color="#F3F4F6", zorder=0)
    for ph in PHORD:
        d = g[(g.phenotype == ph) & (g.hour <= 168)].sort_values("hour")
        d = d[d.n_at_risk >= 20]
        if len(d):
            ys = (100 * d.frac_assessable).rolling(6, center=True, min_periods=1).mean()
            ax.plot(d.hour, ys, color=PH[ph], lw=2.0, label=ph if ph != "comparator" else "Comparator")
    ax.set_xlabel("Hours since ICU admission"); ax.set_ylabel("Verbal examination assessable (%)")
    ax.set_xlim(0, 168); ax.set_ylim(0, 100)
    ax.text(36, 6, "First 72 h", ha="center", fontsize=8, color=MUTED, style="italic")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=4, frameon=False, fontsize=9)
    ax.grid(axis="y", color="#F3F4F6", lw=0.6, zorder=0)
    save(fig, "fig2_assessability")


# ---------------- Figure 3: determinant concordance ----------------
def fig3():
    ct = pd.read_csv(TAB / "table_concordance.csv", index_col=0).reindex(NEURO)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(ct)); w = 0.26
    for i, (col, lab, c) in enumerate([("vent", "Mechanical ventilation", BLUE),
                                       ("sedative", "Sedative infusion", SALMON),
                                       ("nmb", "Neuromuscular blockade", GREY)]):
        ax.bar(x + (i - 1) * w, ct[col], w, label=lab, color=c, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(ct.index)
    ax.set_ylabel("Non-assessable examinations with\nconcurrent exposure (%)"); ax.set_ylim(0, 100)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=9)
    ax.grid(axis="y", color="#F3F4F6", lw=0.6, zorder=0)
    save(fig, "fig3_concordance")


# ---------------- Figure 4: handling-strategy audit ----------------
def fig4():
    t3 = pd.read_csv(TAB / "table3_aim3_by_phenotype.csv")
    a = t3[t3.phenotype == "ALL"].set_index("strategy")
    a4 = pd.read_csv(TAB / "table_aim4_audit.csv").set_index("strategy")
    order = ["drop", "impute1", "gcs_t", "kramer", "learned"]
    labels = ["Complete-\ncase", "Impute\nverbal=1", "E+M\n(T)", "Brennan", "Learned"]
    auroc = [a.loc["drop", "auroc"], a.loc["impute1", "auroc"], a.loc["gcs_t", "auroc"],
             a.loc["kramer", "auroc"], a4.loc["Learned (sedation-aware)", "auroc"]]
    lo = [a.loc["drop", "auroc_lo"], a.loc["impute1", "auroc_lo"], a.loc["gcs_t", "auroc_lo"],
          a.loc["kramer", "auroc_lo"], a4.loc["Learned (sedation-aware)", "auroc_lo"]]
    hi = [a.loc["drop", "auroc_hi"], a.loc["impute1", "auroc_hi"], a.loc["gcs_t", "auroc_hi"],
          a.loc["kramer", "auroc_hi"], a4.loc["Learned (sedation-aware)", "auroc_hi"]]
    excl = [a.loc["drop", "n_excluded"], 0, 0, 0, 0]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    xs = np.arange(5)
    err = [np.array(auroc) - np.array(lo), np.array(hi) - np.array(auroc)]
    cols = [GREY, BLUE, BLUE, BLUE, SALMON]
    axes[0].bar(xs, auroc, color=cols, edgecolor="white", yerr=err, capsize=3,
                error_kw=dict(ecolor=LABEL, lw=1))
    axes[0].set_ylim(0.5, 0.83); axes[0].set_xticks(xs); axes[0].set_xticklabels(labels, fontsize=8.5)
    axes[0].set_ylabel("AUROC (out-of-fold)"); panel(axes[0], "a")
    axes[0].grid(axis="y", color="#F3F4F6", lw=0.6, zorder=0)
    axes[1].bar(xs, excl, color=[SALMON if e else "#E5E7EB" for e in excl], edgecolor="white")
    axes[1].set_xticks(xs); axes[1].set_xticklabels(labels, fontsize=8.5)
    axes[1].set_ylabel("Patients excluded"); panel(axes[1], "b")
    axes[1].text(0, excl[0] + 120, f"{int(excl[0]):,}\n(50% of deaths)", ha="center", fontsize=8, color=LABEL)
    axes[1].grid(axis="y", color="#F3F4F6", lw=0.6, zorder=0)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig4_audit")


# ---------------- Figure 5: Aim 4 learned imputation ----------------
def fig5():
    rec = pd.read_csv(TAB / "table_aim4_recovery.csv")
    aud = pd.read_csv(TAB / "table_aim4_audit.csv")
    agr = pd.read_csv(TAB / "table_aim4_agreement.csv", index_col=0)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    # (a) QWK recovery
    meth = ["Learned (sedation-aware)", "Brennan (eye+motor)", "Fixed verbal=1"]
    short = ["Learned", "Brennan", "Fixed=1"]
    q = [rec[rec.method == m].qwk.iloc[0] for m in meth]
    qlo = [rec[rec.method == m].qwk_lo.iloc[0] for m in meth]
    qhi = [rec[rec.method == m].qwk_hi.iloc[0] for m in meth]
    err = [np.array(q) - np.array(qlo), np.array(qhi) - np.array(q)]
    axes[0].bar(range(3), q, color=[BLUE, GREY, SALMON], edgecolor="white",
                yerr=err, capsize=3, error_kw=dict(ecolor=LABEL, lw=1))
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels(short, fontsize=9)
    axes[0].set_ylabel("Quadratic weighted κ\n(verbal-score recovery)"); axes[0].set_ylim(0, 0.7)
    panel(axes[0], "a"); axes[0].grid(axis="y", color="#F3F4F6", lw=0.6, zorder=0)
    # (b) downstream AUROC
    s = ["Learned (sedation-aware)", "Brennan"]
    au = [aud[aud.strategy == m].auroc.iloc[0] for m in s]
    al = [aud[aud.strategy == m].auroc_lo.iloc[0] for m in s]
    ah = [aud[aud.strategy == m].auroc_hi.iloc[0] for m in s]
    e2 = [np.array(au) - np.array(al), np.array(ah) - np.array(au)]
    axes[1].bar(range(2), au, color=[BLUE, GREY], edgecolor="white", yerr=e2, capsize=3,
                error_kw=dict(ecolor=LABEL, lw=1), width=0.6)
    axes[1].set_xticks(range(2)); axes[1].set_xticklabels(["Learned", "Brennan"], fontsize=9)
    axes[1].set_ylabel("Mortality-model AUROC"); axes[1].set_ylim(0.5, 0.83)
    panel(axes[1], "b"); axes[1].grid(axis="y", color="#F3F4F6", lw=0.6, zorder=0)
    # (c) divergence by sedation depth
    agr = agr.reindex(["Deep sedation (RASS<=-3)", "Light/none (RASS>-3)", "RASS missing"])
    lab = ["Deep\nsedation", "Light /\nnone", "RASS\nmissing"]
    axes[2].bar(range(3), agr["mean"], color=SALMON, edgecolor="white")
    axes[2].axhline(0, color="#9CA3AF", lw=0.7)
    axes[2].set_xticks(range(3)); axes[2].set_xticklabels(lab, fontsize=9)
    axes[2].set_ylabel("Learned − Brennan\ntotal GCS (points)")
    panel(axes[2], "c"); axes[2].grid(axis="y", color="#F3F4F6", lw=0.6, zorder=0)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig5_learned")


for f in (fig1, fig2, fig3, fig4, fig5):
    f(); print(f.__name__, "done")
