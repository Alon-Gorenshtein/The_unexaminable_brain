"""Journal-grade figures (scientific-figures skill standard): Helvetica, colorblind-safe,
600 DPI PDF+PNG, no in-figure titles, panel labels, error bars."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np, pandas as pd
from config import INT, TAB, FIG, REV

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


def ama(n):
    """AMA count style, as in the manuscript: a thousands comma only from 10,000."""
    n = int(n)
    return f"{n:,}" if abs(n) >= 10000 else str(n)


def top_of(hi, floor, share=0.08):
    """Upper axis limit from the data: the highest upper bound plus a share of the axis range."""
    hi = float(np.max(hi))
    return hi + share * (hi - floor)


# ---------------- eFigure 1: study flow ----------------
def fig1():
    cohort = pd.read_parquet(INT / "cohort.parquet")
    first = cohort[cohort.first_stay]
    neuro = first[first.phenotype != "comparator"]
    comp = first[first.phenotype == "comparator"]
    # comparator stays with a charted verbal entry: the same source and rule as the analytic cohort (Table 1)
    burden = pd.read_parquet(INT / "burden_per_stay.parquet")
    comp_verbal = int(comp.stay_id.isin(burden.loc[burden.phenotype == "comparator", "stay_id"]).sum())
    comp_no_verbal = len(comp) - comp_verbal
    CA = json.loads((REV / "cohort_accounting.json").read_text())
    assert len(neuro) == CA["neuro_first_stay_n"], "cohort.parquet and cohort_accounting.json disagree"
    n_repeat = len(cohort) - len(first)
    assert (cohort.age < 18).sum() == 0, "the cohort file holds minors: the exclusion note must name age"
    fig, ax = plt.subplots(figsize=(6.6, 5.1))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(2.8, 13)

    def box(x, y, w, h, t, fc="#FFFFFF"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", lw=1.1, edgecolor=BLUE, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9, color=LABEL)

    def arr(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1))

    def note(x, y, t):
        ax.text(x, y, t, ha="left", va="center", fontsize=8.5, color=MUTED)

    def down(x, y_top, y_bottom, text=None):
        """Arrow between two boxes; an exclusion note is centred on the middle of that arrow."""
        arr(x, y_top, x, y_bottom)
        if text:
            note(x + 2.75, (y_top + y_bottom) / 2, text)

    box(2.0, 11.6, 6.0, 1.1, f"MIMIC-IV v3.1 ICU stays\n(N = {ama(len(cohort))})", "#EAF2F8")
    arr(5, 11.6, 5, 10.8)
    box(2.0, 9.7, 6.0, 1.1, f"First ICU stay per hospitalization, age ≥ 18\n(N = {ama(len(first))})")
    note(8.5, 10.25, f"Excluded:\nrepeat ICU stays\n(n = {ama(n_repeat)})")
    arr(3.4, 9.7, 3.4, 8.9); arr(6.6, 9.7, 6.6, 8.9)
    box(0.3, 7.7, 5.1, 1.2, f"Acute brain injury\n(6 phenotypes; N = {ama(len(neuro))})", "#FDF0E8")
    box(5.9, 7.7, 3.8, 1.2, f"General-ICU comparator\n(N = {ama(len(comp))})", "#EEF1F5")
    ax.text(7.15, 7.5, f"Excluded: no verbal GCS\ndocumented (n = {ama(comp_no_verbal)})\n"
                       f"Comparator analyzed\n(N = {ama(comp_verbal)})",
            ha="left", va="top", fontsize=8.5, color=MUTED)
    arr(2.85, 7.7, 2.85, 6.4)
    note(3.05, 7.05, f"Excluded: no verbal GCS\ndocumented (n = {ama(CA['no_verbal_n'])})")
    box(0.3, 5.2, 5.1, 1.2, f"At least one charted GCS verbal entry\nAnalytic cohort (N = {ama(CA['with_verbal_n'])})", "#FDF0E8")
    down(2.85, 5.2, 4.4, f"Excluded: no complete first-day\nexamination (n = {ama(CA['aim3_excluded_n'])})")
    box(0.3, 3.2, 5.1, 1.2, f"Audit subset: complete first-day\neye, motor and verbal\n(N = {ama(CA['aim3_n'])})", "#FDF0E8")
    save(fig, "fig1_flow")


# ---------------- Figure 1: assessability curves ----------------
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


# ---------------- eFigure 2: determinant concordance ----------------
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


# ---------------- Figure 3: handling-strategy audit (file fig4_audit) ----------------
def fig4():
    c = pd.read_csv(REV / "comparability.csv")
    a = c[c.group == "all"].set_index("slug")
    n = c[c.group == "nonassessable"].set_index("slug")
    order = ["official", "reconstructed", "impute1", "gcs_t", "brennan"]
    # rule names as in Table 3 of the manuscript
    labels = ["Official first-day GCS", "Default-to-15", "Verbal imputed as 1", "Eye-plus-motor sum (T)", "Brennan estimate"]
    AJ = json.loads((REV / "audit.json").read_text())
    cols = [SALMON, SALMON, BLUE, BLUE, BLUE]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.1), gridspec_kw={"width_ratios": [1.5, 1.5, 0.8]})
    xs = np.arange(5)
    au = a.loc[order, "auroc"].values
    hi = a.loc[order, "hi"].values
    axes[0].bar(xs, au, color=cols, edgecolor="white", zorder=3,
                yerr=[au - a.loc[order, "lo"].values, hi - au], capsize=3, error_kw=dict(ecolor=LABEL, lw=1))
    axes[0].set_ylim(0.5, top_of(hi, 0.5)); axes[0].set_xticks(xs); axes[0].set_xticklabels(labels, fontsize=8.5, rotation=30, ha="right", rotation_mode="anchor")
    axes[0].set_xlabel("Handling rule")
    axes[0].set_ylabel(f"AUROC, same {ama(a.loc['brennan', 'n'])} stays"); panel(axes[0], "A")
    oe = n.loc[order, "oe_ratio"].values
    axes[1].bar(xs, oe, color=cols, edgecolor="white", zorder=3)
    axes[1].axhline(1, color="#9CA3AF", lw=0.8, zorder=4)
    axes[1].set_ylim(0, 1.12 * oe.max())
    axes[1].set_xticks(xs); axes[1].set_xticklabels(labels, fontsize=8.5, rotation=30, ha="right", rotation_mode="anchor")
    axes[1].set_xlabel("Handling rule")
    axes[1].set_ylabel("Observed / expected deaths,\nnon-assessable subgroup"); panel(axes[1], "B")
    total, retained = AJ["brennan_lowest_em_n"], AJ["complete_case_lowest_em_n"]
    n_excl = total - retained
    assert n_excl == AJ["sel_n_excluded"], "complete-case exclusions disagree between audit.json keys"
    axes[2].bar([0], [n_excl], color=GREY, edgecolor="white", width=0.5, zorder=3)
    axes[2].set_xlim(-0.6, 0.6)
    axes[2].set_xticks([0]); axes[2].set_xticklabels(["Complete-case\n(stays excluded)"], fontsize=9)
    axes[2].set_ylabel("Stays excluded"); panel(axes[2], "C")
    axes[2].set_ylim(0, n_excl * 1.5)
    axes[2].text(0, n_excl * 1.03,
                 f"{ama(n_excl)} of {ama(total)} stays\n({AJ['sel_pct_deaths_excluded']:.1f}% of deaths)\n"
                 f"{ama(retained)} stays retained,\nmortality {AJ['complete_case_lowest_em_mortality_pct']:.1f}%",
                 ha="center", va="bottom", fontsize=8.5, color=LABEL)
    for ax_ in axes:
        ax_.grid(axis="y", color="#F3F4F6", lw=0.6, zorder=0)
    fig.subplots_adjust(wspace=0.4)
    save(fig, "fig4_audit")


# ---------------- eFigure 3: learned imputation ----------------
def fig5():
    rec = pd.read_csv(REV / "aim4_recovery.csv").set_index("method")
    AJ = json.loads((REV / "aim4.json").read_text())
    miss = pd.read_csv(REV / "aim4_rass_missing.csv", index_col=0).iloc[:, 0]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6), gridspec_kw={"width_ratios": [1, 0.75, 1.4]})
    meth, short = ["learned", "brennan", "fixed1"], ["Model", "Brennan\nestimate", "Fixed = 1"]
    q, lo, hi = (rec.loc[meth, c].values for c in ("qwk", "qwk_lo", "qwk_hi"))
    axes[0].bar(range(3), q, color=BLUE, edgecolor="white", yerr=[q - lo, hi - q],
                capsize=3, error_kw=dict(ecolor=LABEL, lw=1), zorder=3)
    for i_, (v_, h_) in enumerate(zip(q, hi)):
        axes[0].text(i_, h_ + 0.015 * hi.max(), f"{v_:.2f}", ha="center", va="bottom", fontsize=9, color=LABEL)
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels(short, fontsize=9)
    axes[0].set_xlabel("Verbal-score estimate")
    axes[0].set_ylabel("Quadratic weighted kappa\n(charted verbal score)"); axes[0].set_ylim(0, 1.12 * hi.max())
    panel(axes[0], "A")
    au = np.array([AJ["learned_downstream_auroc"], AJ["brennan_downstream_auroc"]])
    al = np.array([AJ["learned_downstream_lo"], AJ["brennan_downstream_lo"]])
    ah = np.array([AJ["learned_downstream_hi"], AJ["brennan_downstream_hi"]])
    axes[1].bar(range(2), au, color=BLUE, edgecolor="white", width=0.6, yerr=[au - al, ah - au],
                capsize=3, error_kw=dict(ecolor=LABEL, lw=1), zorder=3)
    axes[1].set_xticks(range(2)); axes[1].set_xticklabels(["Model", "Brennan\nestimate"], fontsize=9)
    axes[1].set_xlabel("Verbal-score estimate")
    axes[1].set_ylabel("Mortality-model AUROC"); axes[1].set_ylim(0.5, top_of(ah, 0.5)); panel(axes[1], "B")
    keys = ["nonassess=False", "nonassess=True", "vent=0", "vent=1"]
    axes[2].bar(range(4), miss.reindex(keys).values, color=GREY, edgecolor="white", zorder=3)
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(["Assessable", "Non-\nassessable", "Not\nventilated", "Ventilated"], fontsize=9)
    axes[2].set_xlabel("Examination subgroup")
    axes[2].set_ylabel("Examinations without RASS (%)"); panel(axes[2], "C")
    for a_ in axes:
        a_.grid(axis="y", color="#F3F4F6", lw=0.6, zorder=0)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig5_learned")


for f in (fig1, fig2, fig3, fig4, fig5):
    f(); print(f.__name__, "done")
