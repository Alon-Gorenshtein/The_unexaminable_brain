"""Portal figures: 600-dpi RGB TIFF (LZW) from the figure PNGs, named as journal figures.

    python3 code/tools/export_figures.py [--src DIR] [--dst DIR]

Exit code 0 = all six figures written; 1 = a source PNG is missing (nothing is written in that case).
"""
import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIG, SUB = ROOT / "output" / "figures", ROOT / "output" / "submission_figures"
MAP = {"fig2_assessability": "Figure1", "fig6_derived": "Figure2", "fig4_audit": "Figure3",
       "fig1_flow": "eFigure1", "fig3_concordance": "eFigure2", "fig5_learned": "eFigure3"}
DPI = 600


def export(src_dir, dst_dir, mapping):
    """Write dst_dir/<journal name>.tif for every source stem in `mapping`; return the paths written.

    Raises FileNotFoundError (naming every absent source) before writing anything.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    sources = {stem: src_dir / f"{stem}.png" for stem in mapping}
    missing = [str(p) for p in sources.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing source PNG(s): " + ", ".join(missing))
    dst_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, name in mapping.items():
        out = dst_dir / f"{name}.tif"
        with Image.open(sources[stem]) as im:
            im = im.convert("RGB")
            im.save(out, dpi=(DPI, DPI), compression="tiff_lzw")
            size = im.size
        print(f"{name}: {size[0]} x {size[1]} px, {out.stat().st_size / 1e6:.2f} MB")
        written.append(out)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=str(FIG))
    ap.add_argument("--dst", default=str(SUB))
    args = ap.parse_args(argv)
    try:
        export(args.src, args.dst, MAP)
    except FileNotFoundError as exc:
        print(f"export_figures: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
