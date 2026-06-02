"""Figure 3 (new centerpiece): the MIMIC-derived GCS normalizes intubated patients.
(a) SOFA CNS category distribution under MIMIC-derived vs component-aware GCS.
(b) Where component-aware-severe patients land under the derived rule (hidden normalization).
(c) Mortality-model discrimination: MIMIC-derived vs component-aware."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from config import TAB, FIG
BLUE, GREY, SALMON, LABEL = "#00468B", "#ADB6B6", "#FDAF91", "#2B2B2B"
plt.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": "#9CA3AF", "axes.linewidth": 0.8})
def save(fig, path):
    for e in ("pdf","png"): fig.savefig(f"{path}.{e}", dpi=600, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
def panel(ax, l): ax.text(-0.16,1.06,l,transform=ax.transAxes,fontsize=14,fontweight="bold",color=LABEL)
def faint_grid(ax): ax.grid(axis="y", color="#F3F4F6", lw=0.6, zorder=0)
ct = pd.read_csv(TAB / "table_sofa_crosstab.csv", index_col=0)   # rows=component-aware, cols=derived
ct.columns = [int(c) for c in ct.columns]; ct.index = [int(i) for i in ct.index]
N = ct.values.sum()

fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.5))
cats = [0, 1, 2, 3, 4]
# (a) category distribution derived vs component-aware
comp = ct.sum(axis=1).reindex(cats, fill_value=0).values
der = ct.sum(axis=0).reindex(cats, fill_value=0).values
x = np.arange(5); w = 0.38
ax[0].bar(x - w/2, 100*comp/N, w, label="Component-aware", color=BLUE, edgecolor="white")
ax[0].bar(x + w/2, 100*der/N, w, label="MIMIC-derived", color=SALMON, edgecolor="white")
ax[0].set_xticks(x); ax[0].set_xticklabels(["0\nnormal", "1", "2", "3", "4\ncoma"])
ax[0].set_xlabel("SOFA CNS category"); ax[0].set_ylabel("Patients (%)")
ax[0].legend(frameon=False, fontsize=8); panel(ax[0], "a"); faint_grid(ax[0])
# (b) hidden normalization: of component-aware severe (cat>=3), how many derived puts in cat 0
sev = ct.loc[[i for i in ct.index if i >= 1]]
to0 = sev[0].reindex([1,2,3,4], fill_value=0).values
ax[1].bar(range(4), to0, color=GREY, edgecolor="white")
ax[1].bar(2, to0[2], color=SALMON, edgecolor="white")   # highlight cat 3
ax[1].bar(3, to0[3], color="#c0392b", edgecolor="white")  # highlight cat 4 (deepest)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["1", "2", "3", "4\ncoma"])
ax[1].set_xlabel("True (component-aware) SOFA CNS")
ax[1].set_ylabel("Patients relabeled\nSOFA CNS 0 by derived GCS")
panel(ax[1], "b"); faint_grid(ax[1])
# (c) mortality AUROC derived vs component-aware
labels = ["MIMIC-\nderived", "Component-\naware"]
auc = [0.746, 0.783]; lo = [0.732, 0.770]; hi = [0.759, 0.797]
err = [np.array(auc)-np.array(lo), np.array(hi)-np.array(auc)]
ax[2].bar([0, 1], auc, color=[SALMON, BLUE], edgecolor="white", yerr=err, capsize=4,
          error_kw=dict(ecolor=LABEL, lw=1), width=0.6)
ax[2].set_ylim(0.5, 0.83); ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(labels, fontsize=9)
ax[2].set_ylabel("Mortality-model AUROC"); panel(ax[2], "c"); faint_grid(ax[2])
fig.subplots_adjust(wspace=0.42)
save(fig, FIG / "fig6_derived")
print("fig6_derived done | derived SOFA0:", int(der[0]), "comp-aware severe->derived0:", int(to0.sum()))
