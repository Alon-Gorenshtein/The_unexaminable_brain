import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from config import INT, FIG
from lib_style import set_style, save

set_style()
c = pd.read_parquet(INT / "cohort.parquet")
first = c[c.first_stay]
neuro = first[first.phenotype != "comparator"]
comp = first[first.phenotype == "comparator"]
burden = pd.read_parquet(INT / "burden_per_stay.parquet")
n_analytic = (burden.phenotype != "comparator").sum()

fig, ax = plt.subplots(figsize=(6.4, 5.4))
ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 12)


def box(x, y, w, h, text, fc="#FFFFFF", ec="#1F2D5C"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                linewidth=1.0, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.4)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0))


box(2.0, 10.4, 6.0, 1.1, f"MIMIC-IV v3.1 ICU stays\n(N = {len(c):,})", fc="#EAF2F2")
arrow(5.0, 10.4, 5.0, 9.6)
box(2.0, 8.5, 6.0, 1.1, f"First ICU stay per hospitalization, age ≥ 18\n(N = {len(first):,})")
# exclusion side note
ax.text(8.4, 9.0, "Excluded:\nrepeat ICU stays,\nage < 18", ha="left", va="center",
        fontsize=6.4, color="#555555")
arrow(3.4, 8.5, 3.4, 7.7)
arrow(6.6, 8.5, 6.6, 7.7)
box(0.4, 6.5, 5.0, 1.2, f"Acute brain injury\n(6 phenotypes; N = {len(neuro):,})", fc="#FBEEE6")
box(5.8, 6.5, 3.8, 1.2, f"General-ICU comparator\n(N = {len(comp):,})", fc="#EEF0F5")
arrow(2.9, 6.5, 2.9, 5.7)
box(0.4, 4.5, 5.0, 1.2, f"≥ 1 charted GCS-verbal entry\nAnalytic cohort (N = {n_analytic:,})", fc="#FBEEE6")
arrow(2.9, 4.5, 2.9, 3.7)
box(0.4, 2.4, 5.0, 1.2, "Aim 3 audit subset:\ncomplete worst first-day GCS\n(N = 12,404)", fc="#FBEEE6")

save(fig, FIG / "fig1_flow")
print("fig1 flow done:", len(c), len(first), len(neuro), len(comp), n_analytic)
