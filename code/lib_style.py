import matplotlib as mpl
import matplotlib.pyplot as plt

# Colorblind-safe phenotype palette (Okabe-Ito based)
PHENO_COLORS = {
    "TBI": "#E69F00", "ICH": "#D55E00", "SAH": "#CC79A7", "SDH": "#0072B2",
    "AIS": "#009E73", "anoxic": "#56B4E9", "comparator": "#999999",
}
PHENO_ORDER = ["TBI", "ICH", "SAH", "SDH", "AIS", "anoxic", "comparator"]
STRAT_LABELS = {"drop": "Complete-case\n(drop)", "impute1": "Impute\nverbal=1",
                "gcs_t": "E+M only\n(T)", "kramer": "Brennan\nimputation"}


def set_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, path):
    fig.savefig(str(path) + ".png")
    fig.savefig(str(path) + ".pdf")
    plt.close(fig)
