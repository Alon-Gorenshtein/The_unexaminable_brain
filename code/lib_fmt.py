KINDS = ("pct1", "pct1n", "auc3", "dec2", "dec4", "int")


def fmt(v, kind):
    if kind == "pct1":
        return f"{v:.1f}%"
    if kind == "pct1n":
        return f"{v:.1f}"
    if kind == "auc3":
        return f"{v:.3f}"
    if kind == "dec2":
        return f"{v:.2f}"
    if kind == "dec4":
        return f"{v:.4f}"
    if kind == "int":
        n = int(round(v))
        return f"{n:,}" if abs(n) >= 10000 else str(n)
    raise ValueError(kind)
