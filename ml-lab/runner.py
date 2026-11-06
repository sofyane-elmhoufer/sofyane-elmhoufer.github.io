"""Exécuteur de cellules façon Jupyter, chargé dans Pyodide.
Un seul espace de noms partagé entre les cellules ; seule la dernière expression est affichée."""
import ast
import base64
import contextlib
import io
import json
import os
import re
import sys
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

os.environ.setdefault("SEABORN_DATA", os.path.join(os.getcwd(), "seaborn-data"))

_ns = {"__name__": "__main__"}
_last = {}  # résultat brut de la dernière cellule exécutée (utilisé par checker.py)


def reset():
    global _ns, _last
    plt.close("all")
    _ns = {"__name__": "__main__"}
    _last = {}
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


def _fig_sig(fig):
    """Signature d'un graphique : structure (axes, éléments, titres, légende) ET données tracées."""
    import numpy as np

    def _num(v):
        try:
            a = np.asarray(v, dtype=float)
            return a if a.size else None
        except Exception:
            return None

    sig = {"axes": 0, "lines": 0, "patches": 0, "collections": 0, "texts": 0, "images": 0,
           "labels": [], "titles": [], "legend": False, "xy": [], "heights": [], "widths": [],
           "points": [], "mesh": [], "markers": 0, "per_ax": [],
           "size": tuple(float(x) for x in fig.get_size_inches())}
    labels, titles = set(), []
    for ax in fig.get_axes():
        sig["axes"] += 1
        pa = {"title": ax.get_title().strip().lower(), "bars": [], "pts": [], "lines": []}
        sig["per_ax"].append(pa)
        sig["lines"] += len(ax.lines)
        sig["patches"] += len(ax.patches)
        sig["collections"] += len(ax.collections)
        sig["texts"] += len(ax.texts)
        sig["images"] += len(ax.images)
        for lab in (ax.get_xlabel(), ax.get_ylabel()):
            if lab:
                labels.add(lab.strip().lower())
        t = ax.get_title()
        if t:
            titles.append(t.strip().lower())
        if ax.get_legend() is not None:
            sig["legend"] = True
        for ln in ax.lines:
            y = _num(ln.get_ydata())
            if y is not None:
                sig["xy"].append((_num(ln.get_xdata()), y))
                pa["lines"].append(y)
            if ln.get_marker() not in (None, "None", "", " ") and ln.get_visible():
                sig["markers"] += 1
        for pt in ax.patches:
            try:
                sig["heights"].append(float(pt.get_height()))
                sig["widths"].append(float(pt.get_width()))
                pa["bars"].append((round(float(pt.get_x()), 6), round(float(pt.get_y()), 6),
                                   float(pt.get_width()), float(pt.get_height())))
            except Exception:
                pass
        for c in ax.collections:
            try:
                arr = c.get_array()
                if arr is not None and np.size(arr) > 0 and type(c).__name__ == "QuadMesh":
                    sig["mesh"].append(np.asarray(arr, dtype=float).ravel())
                    continue
            except Exception:
                pass
            try:
                off = np.asarray(c.get_offsets(), dtype=float)
                if off.ndim == 2 and off.shape[0] > 0 and not np.isnan(off).all():
                    sig["points"].append(off)
                    pa["pts"].append(off)
            except Exception:
                pass
    for pa in sig["per_ax"]:
        pa["bars"] = [(w, h) for _x, _y, w, h in sorted(pa["bars"])]
    sig["labels"] = sorted(labels)
    sig["titles"] = sorted(titles)
    return sig


def _collect_figures(want_images):
    sigs, images = [], []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        sigs.append(_fig_sig(fig))
        if want_images:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
            images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    plt.close("all")
    return sigs, images


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


def _friendly(exc):
    """Explication en français, sans donner la solution."""
    m = str(exc)
    if isinstance(exc, IndentationError):
        return "Erreur d'indentation : les lignes d'un même bloc doivent commencer à la même colonne."
    if isinstance(exc, SyntaxError):
        return "Erreur de syntaxe : il manque peut-être une parenthèse, un guillemet ou deux-points, ou une valeur après un « = ». Regarde la ligne indiquée."
    if isinstance(exc, KeyError):
        return f"Colonne ou clé introuvable : {m}. Vérifie l'orthographe (majuscules, accents) avec df.columns."
    if isinstance(exc, NameError):
        return f"Nom inconnu : {m}. As-tu exécuté la cellule de préparation et les cellules précédentes ? Le nom est-il bien écrit ?"
    if isinstance(exc, AttributeError):
        return "Cet objet n'a pas cette méthode ou cet attribut. Vérifie l'orthographe et les parenthèses (df.shape sans parenthèses, df.head() avec)."
    if isinstance(exc, FileNotFoundError):
        return "Fichier introuvable : vérifie le nom exact du fichier (extension .csv comprise)."
    if isinstance(exc, TypeError):
        return "Type incorrect : une fonction a reçu un argument inattendu, ou tu appelles quelque chose qui n'est pas une fonction."
    if isinstance(exc, ValueError):
        return "Valeur incorrecte : l'argument a le bon type mais sa valeur ne convient pas (forme des données, NaN, nom de colonne…)."
    if isinstance(exc, ImportError):
        return "Module introuvable : vérifie le nom de l'import."
    if isinstance(exc, ZeroDivisionError):
        return "Division par zéro."
    return "Il y a une erreur dans ton code. Lis la dernière ligne du message ci-dessous."


def _exec_in(ns, code, want_images=False):
    """Exécute `code` dans `ns` façon Jupyter et renvoie tout (stdout, valeur, figures, exception)."""
    res = {"stdout": "", "value": None, "has_value": False, "exc": None, "err_text": None,
           "empty": False, "figs": [], "images": []}
    buf = io.StringIO()
    try:
        tree = ast.parse(code, "<cell>")
        res["empty"] = not tree.body
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(tree, "<cell>", "exec"), ns)
            if last is not None:
                val = eval(compile(last, "<cell>", "eval"), ns)
                if val is not None:
                    res["value"] = val
                    res["has_value"] = True
    except BaseException as e:  # noqa: BLE001
        res["exc"] = e
        res["err_text"] = _format_error(code)
    res["stdout"] = buf.getvalue()
    try:
        res["figs"], res["images"] = _collect_figures(want_images)
    except Exception:
        plt.close("all")
    return res


def run_cell(code):
    global _last
    res = _exec_in(_ns, code, want_images=True)
    _last = res
    out = {"stdout": res["stdout"], "result": None, "result_html": None, "images": res["images"],
           "error": res["err_text"], "friendly": _friendly(res["exc"]) if res["exc"] is not None else None}
    if res["has_value"]:
        _result_to_out(res["value"], out)
    return json.dumps(out)
