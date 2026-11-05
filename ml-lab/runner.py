"""Exécuteur de cellules façon Jupyter, chargé dans Pyodide.
Un seul espace de noms partagé entre les cellules ; seule la dernière expression est affichée."""
import ast
import base64
import contextlib
import io
import json
import re
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn import set_config

    set_config(display="text")
except Exception:
    pass

import os

os.environ.setdefault("SEABORN_DATA", os.path.join(os.getcwd(), "seaborn-data"))

_ns = {"__name__": "__main__"}


def reset():
    global _ns
    plt.close("all")
    _ns = {"__name__": "__main__"}
    return "ok"


def _result_to_out(val, out):
    try:
        import pandas as pd

        if isinstance(val, pd.DataFrame):
            html = val._repr_html_()
            if html:
                out["result_html"] = re.sub(r"<style scoped>.*?</style>", "", html, flags=re.S)
                return
    except Exception:
        pass
    try:
        import numpy as np

        if isinstance(val, np.generic):
            val = val.item()  # np.float64(29.7) -> 29.7
    except Exception:
        pass
    out["result"] = repr(val)


def _collect_figures():
    images = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
        images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    plt.close("all")
    return images


def _format_error(code):
    """Erreur lisible : seulement les lignes de TON code, puis le message (pas les rouages internes de pandas)."""
    etype, evalue, tb = sys.exc_info()
    if issubclass(etype, SyntaxError):
        return "".join(traceback.format_exception_only(etype, evalue)).strip()
    lines = code.splitlines()
    out = []
    for fs in traceback.extract_tb(tb):
        if fs.filename == "<cell>":
            src = lines[fs.lineno - 1].strip() if 0 < fs.lineno <= len(lines) else ""
            out.append(f"Ligne {fs.lineno} : {src}")
    out.append("".join(traceback.format_exception_only(etype, evalue)).strip())
    return "\n".join(out)


def run_cell(code):
    out = {"stdout": "", "result": None, "result_html": None, "images": [], "error": None}
    buf = io.StringIO()
    try:
        tree = ast.parse(code, "<cell>")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(tree, "<cell>", "exec"), _ns)
            val = eval(compile(last, "<cell>", "eval"), _ns) if last is not None else None
        if val is not None:
            _result_to_out(val, out)
    except BaseException:
        out["error"] = _format_error(code)
    out["stdout"] = buf.getvalue()
    try:
        out["images"] = _collect_figures()
    except Exception:
        pass
    return json.dumps(out)
