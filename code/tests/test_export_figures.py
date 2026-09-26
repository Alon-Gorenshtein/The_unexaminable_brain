"""Portal figure export (600-dpi RGB LZW TIFF) and a static check that the figure scripts type no result."""
import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

import export_figures

CODE = Path(__file__).resolve().parents[1]
# What counts as a typed result: an AUROC-like number (0.5 up to 0.99..., any number of decimals, as a whole
# token) or a count of four or five digits (plain or with thousands commas), in code or inside a label string.
AUROC_LIKE = re.compile(r"(?<![\w.])0?\.[5-9]\d*")
COUNT_LIKE = re.compile(r"(?<![\w.,])[1-9]\d{3,4}(?![\w.])|(?<![\w.,])\d{1,3}(?:,\d{3})+(?![\w,])")
# Where a literal is layout, not a result: the arguments of an axis-limit call, keyword arguments that only
# size or place things, the two position arguments of ax.text, and rcParams entries for widths and sizes.
AXIS_CALLS = {"set_ylim", "set_xlim"}
LAYOUT_KW = {"figsize", "dpi", "fontsize", "lw", "linewidth", "width", "width_ratios", "height_ratios",
             "gridspec_kw", "wspace", "hspace", "capsize", "pad_inches", "alpha", "zorder", "rotation",
             "bbox_to_anchor", "size", "pad", "s"}


def _synthetic_png(path, size=(120, 80)):
    im = Image.new("RGBA", size, (255, 255, 255, 255))
    for x in range(0, size[0], 7):
        im.putpixel((x, x % size[1]), (0, 70, 139, 255))
    im.save(path)


def test_export_writes_600_dpi_rgb_lzw_tiff(tmp_path):
    src, dst = tmp_path / "png", tmp_path / "tif"
    src.mkdir()
    _synthetic_png(src / "alpha.png")
    _synthetic_png(src / "beta.png", (64, 64))
    written = export_figures.export(src, dst, {"alpha": "Figure1", "beta": "eFigure2"})
    assert sorted(p.name for p in written) == ["Figure1.tif", "eFigure2.tif"]
    for p in written:
        with Image.open(p) as im:
            assert im.format == "TIFF"
            assert im.mode == "RGB"
            assert tuple(round(v) for v in im.info["dpi"]) == (600, 600)
            assert im.info["compression"] == "tiff_lzw"
    with Image.open(dst / "Figure1.tif") as im:
        assert im.size == (120, 80)


def test_export_fails_loudly_when_a_source_is_missing_and_writes_nothing(tmp_path):
    src, dst = tmp_path / "png", tmp_path / "tif"
    src.mkdir()
    _synthetic_png(src / "alpha.png")
    with pytest.raises(FileNotFoundError, match="beta"):
        export_figures.export(src, dst, {"alpha": "Figure1", "beta": "Figure2"})
    assert not dst.exists() or not list(dst.glob("*.tif"))


def test_export_main_exits_non_zero_on_a_missing_source(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    r = subprocess.run([sys.executable, str(CODE / "tools" / "export_figures.py"), "--src", str(empty),
                        "--dst", str(tmp_path / "out")], capture_output=True, text=True)
    assert r.returncode != 0
    assert "missing" in (r.stderr + r.stdout).lower()


def test_export_maps_the_six_journal_figures():
    assert sorted(export_figures.MAP.values()) == ["Figure1", "Figure2", "Figure3",
                                                   "eFigure1", "eFigure2", "eFigure3"]
    # Journal numbers follow first citation in the main text: the feasible-range figure (Results, handling
    # rules) is cited before the handling-rule audit figure.
    assert export_figures.MAP["fig6_derived"] == "Figure2"
    assert export_figures.MAP["fig4_audit"] == "Figure3"


# ---- static check: no typed AUROC, interval or count in the figure scripts ----
def _typed_literal(node):
    """True when this constant is an AUROC-like number or a four/five-digit count."""
    v = node.value
    if isinstance(v, bool):
        return False
    if isinstance(v, float):
        return 0.5 <= v < 1
    if isinstance(v, int):
        return 1000 <= v <= 99999
    if isinstance(v, str):
        return bool(AUROC_LIKE.search(v) or COUNT_LIKE.search(v))
    return False


def _walk(node, exempt, hits):
    if isinstance(node, ast.Call):
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        _walk(f, exempt, hits)
        for i, a in enumerate(node.args):
            _walk(a, exempt or name in AXIS_CALLS or (name == "text" and i < 2), hits)
        for kw in node.keywords:
            _walk(kw.value, exempt or name in AXIS_CALLS or kw.arg in LAYOUT_KW, hits)
        return
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if k is not None:
                _walk(k, exempt, hits)
            layout_key = isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value.split(".")[-1] in LAYOUT_KW
            _walk(v, exempt or layout_key, hits)
        return
    if isinstance(node, ast.Constant):
        if not exempt and _typed_literal(node):
            hits.append((node.lineno, repr(node.value)))
        return
    for child in ast.iter_child_nodes(node):
        _walk(child, exempt, hits)


def find_typed_literals(source, only_functions=None):
    """(line, literal) for every typed AUROC-like number or 4-5 digit count in the code or its label strings.

    Comments are invisible to the parser. Exempt: the arguments of set_ylim/set_xlim, layout keyword arguments
    (figsize, dpi, fontsize, lw, width, ...), the two position arguments of ax.text, and rcParams width/size
    entries. only_functions restricts the scan to those top-level functions (the ones this revision changed)
    in a script that also holds unchanged figure code.
    """
    tree = ast.parse(source)
    roots = [tree] if only_functions is None else [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in only_functions]
    hits = []
    for root in roots:
        _walk(root, False, hits)
    return hits


def test_detector_scans_only_the_named_functions():
    src = ("def fig_old():\n    ax.text(0, 0, 'AUROC 0.783')\n\n\n"
           "def fig_new():\n    ax.set_ylim(0.5, 0.830)\n    ax.text(0, 0, f'{x:.3f}')  # 0.777 in a comment\n"
           "    ax.text(1, 1, 'CI 0.770 to 0.797')\n")
    assert [h[0] for h in find_typed_literals(src, only_functions={"fig_new"})] == [8]
    assert [h[0] for h in find_typed_literals(src, only_functions={"fig_old"})] == [2]
    assert find_typed_literals(src, only_functions=set()) == []


def test_fig_derived_types_no_auroc_or_interval():
    src = (CODE / "33_fig_derived.py").read_text()
    assert find_typed_literals(src) == []


def test_changed_figure_functions_type_no_auroc_or_interval():
    src = (CODE / "32_figures_final.py").read_text()
    assert find_typed_literals(src, only_functions={"fig1", "fig4", "fig5", "top_of"}) == []


# ---- injected literals: what the detector must and must not flag ----
def test_flags_a_typed_auroc_even_when_set_ylim_is_on_the_same_line():
    hits = find_typed_literals('ax.set_ylim(0.5, 0.83); ax.text(0, 0, "0.783")\n')
    assert len(hits) == 1 and "0.783" in hits[0][1]


def test_flags_a_typed_count():
    assert find_typed_literals("n_excluded = 5509\n")
    assert find_typed_literals('ax.text(0, 0, f"{5509:,} excluded")\n')
    assert find_typed_literals('ax.set_ylabel("Excluded (5509)")\n')
    assert find_typed_literals('ax.set_ylabel("Excluded (5,509)")\n')
    assert find_typed_literals("total = 14192\n")


def test_flags_a_two_digit_auroc_in_a_label():
    assert find_typed_literals('ax.set_ylabel("AUROC 0.75")\n')
    assert find_typed_literals('ax.text(0, 0, "CI 0.75 to 0.83")\n')
    assert find_typed_literals("auc = [0.75, 0.82]\n")


def test_does_not_flag_layout_literals_comments_or_longer_numbers():
    ok = (
        "ax.set_ylim(0.5, top_of(hi, 0.5))\n"
        "ax.set_xlim(-0.6, 0.6)\n"
        "ax.bar(x, y, width=0.6, lw=0.8, capsize=3)\n"
        "fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.6), gridspec_kw={'width_ratios': [1.5, 0.75, 1.4]})\n"
        "fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=0.15)\n"
        "ax.text(0.5, 0.97, 'Paired difference', fontsize=9)\n"
        "plt.rcParams.update({'axes.linewidth': 0.8, 'font.size': 10})\n"
        "colour = '#00468B'  # 0.783 in a comment is ignored\n"
        "x = 10.75 + 1.05 + 20.5\n"
        "y = 100 * 0.05 + 999\n"
    )
    assert find_typed_literals(ok) == []


def test_injected_literal_in_the_real_figure_code_is_caught():
    fig_derived = (CODE / "33_fig_derived.py").read_text()
    assert "Paired difference {AJ" in fig_derived
    injected = fig_derived.replace("Paired difference {AJ", "AUROC 0.783; Paired difference {AJ", 1)
    assert len(find_typed_literals(injected)) == 1
    final = (CODE / "32_figures_final.py").read_text()
    anchor = '    fig.subplots_adjust(wspace=0.4)\n    save(fig, "fig4_audit")'
    assert anchor in final
    injected = final.replace(anchor, '    axes[0].text(0, 0, "5509 excluded")\n' + anchor, 1)
    assert find_typed_literals(injected, only_functions={"fig4"})


# ---- figure text uses the manuscript's words: stays, and the rule names of Table 3 ----
FIGURE_SCRIPTS = (("33_fig_derived.py", None), ("32_figures_final.py", {"fig1", "fig4", "fig5"}))
OLD_WORDS = re.compile(r"patient|first_day_gcs|Impute verbal|\(reconstructed\)|Eye\+motor|MIMIC first", re.I)
TABLE_3_NAMES = ["Official first-day GCS", "Default-to-15", "Verbal imputed as 1", "Eye-plus-motor sum (T)",
                 "Brennan estimate"]


def _label_strings(source, only_functions=None):
    """Every string constant of the scanned code with its line, docstrings and comments excluded."""
    tree = ast.parse(source)
    roots = [tree] if only_functions is None else [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in only_functions]
    out = []
    for root in roots:
        docstrings = {id(n.body[0].value) for n in ast.walk(root)
                      if isinstance(n, (ast.Module, ast.FunctionDef)) and n.body and isinstance(n.body[0], ast.Expr)
                      and isinstance(n.body[0].value, ast.Constant)}
        out += [(n.lineno, n.value) for n in ast.walk(root)
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]
    return out


def test_figure_labels_say_stays_and_use_the_table_3_names():
    for name, only in FIGURE_SCRIPTS:
        for line, text in _label_strings((CODE / name).read_text(), only):
            assert not OLD_WORDS.search(text), f"{name}:{line}: old wording in figure text: {text!r}"
    labels = " ".join(t for _, t in _label_strings((CODE / "32_figures_final.py").read_text(), {"fig4"}))
    for expected in TABLE_3_NAMES:
        assert expected in labels, f"Figure 3 label missing: {expected}"
    assert "Stays excluded" in labels and "stays" in labels
    derived = " ".join(t for _, t in _label_strings((CODE / "33_fig_derived.py").read_text()))
    assert "Official first-day GCS" in derived and "Brennan estimate" in derived and "Stays (%)" in derived


def test_label_check_catches_the_old_wording():
    old = 'ax.set_ylabel("Patients excluded")\nlabels = ["MIMIC first_day_gcs", "Eye+motor (T)"]\n'
    assert all(OLD_WORDS.search(t) for _, t in _label_strings(old))
