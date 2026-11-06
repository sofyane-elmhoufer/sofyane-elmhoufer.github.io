"""Correction automatique.

Principe : la solution de chaque cellule est exécutée dans un noyau de RÉFÉRENCE (séparé de celui de l'élève),
puis on compare ce que l'élève a obtenu (affichage, variables modifiées, graphique) avec ce que produit la solution.
Ce fichier est exécuté dans les mêmes globals que runner.py."""
import ast
import contextlib
import copy
import io
import json
import re
import types

import numpy as np
import pandas as pd

_refs = {}
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
RTOL = 1e-7  # tolérance serrée : un résultat « pas arrondi » ne doit pas passer pour un résultat arrondi


# ------------------------------------------------------------------ outils
def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def _kind(v):
    if isinstance(v, pd.DataFrame):
        return "un DataFrame (tableau)"
    if isinstance(v, pd.Series):
        return "une Series (colonne)"
    if isinstance(v, pd.Index):
        return "un Index"
    if isinstance(v, np.ndarray):
        return "un tableau numpy"
    if isinstance(v, (bool, np.bool_)):
        return "un booléen"
    if isinstance(v, (int, float, np.number)):
        return "un nombre"
    if isinstance(v, str):
        return "du texte"
    if isinstance(v, tuple):
        return "un tuple"
    if isinstance(v, list):
        return "une liste"
    if isinstance(v, dict):
        return "un dictionnaire"
    return "un objet " + type(v).__name__


def _is_scalar(v):
    return isinstance(v, (str, bool, int, float, np.generic))


def _vtext(v):
    if isinstance(v, (pd.DataFrame, pd.Series)):
        return v.to_string()
    if isinstance(v, np.generic):
        v = v.item()
    return repr(v)


def _copy(v):
    if isinstance(v, (types.ModuleType, types.FunctionType, type)):
        return v
    try:
        return copy.deepcopy(v)
    except Exception:
        return v


def _clone(ns):
    return {k: _copy(v) for k, v in ns.items() if k != "__builtins__"}


def _nums(text):
    return [float(x) for x in _NUM.findall(text)]


def _decimals(txt):
    """Nombre de décimales écrites dans la représentation d'un nombre."""
    t = str(txt).lower()
    if "e" in t:
        return 15
    return len(t.split(".")[1]) if "." in t else 0


def _num_text_ok(us, es):
    """us (élève) et es (attendu) : nombres sous forme de texte.
    Accepté si égaux, ou si l'élève a arrondi PLUS grossièrement que l'attendu de façon cohérente.
    Refusé si l'élève montre plus de décimales que l'attendu sans être égal (= il n'a pas arrondi)."""
    try:
        u, e = float(us), float(es)
    except Exception:
        return False
    if np.isclose(u, e, rtol=RTOL, atol=1e-9, equal_nan=True):
        return True
    du, de = _decimals(us), _decimals(es)
    return du < de and abs(u - e) <= 0.5 * 10 ** (-du) + 1e-9


def _num_close(x, y):
    """Comparaison de deux nombres Python/NumPy (même règle que _num_text_ok)."""
    try:
        if isinstance(x, (int, np.integer)) and isinstance(y, (int, np.integer)):
            return int(x) == int(y)
        return _num_text_ok(repr(float(x)), repr(float(y)))
    except Exception:
        return False


# ------------------------------------------------------------------ comparaison de valeurs
def _dfnorm(d):
    d = d.copy()
    for c in d.columns:
        if str(d[c].dtype) == "category":
            d[c] = d[c].astype(object)
    d.columns = [str(c) for c in d.columns]
    return d


def _eq_frame(a, b):
    try:
        pd.testing.assert_frame_equal(_dfnorm(a), _dfnorm(b), check_dtype=False, check_index_type=False,
                                      check_names=False, check_categorical=False, check_column_type=False, rtol=RTOL)
        return True
    except Exception:
        return False


def _eq_series(a, b):
    try:
        a = a.astype(object) if str(a.dtype) == "category" else a
        b = b.astype(object) if str(b.dtype) == "category" else b
        pd.testing.assert_series_equal(a, b, check_dtype=False, check_index_type=False, check_names=False,
                                       check_categorical=False, rtol=RTOL)
        return True
    except Exception:
        return False


def _sorted_idx(x):
    try:
        return x.sort_index()
    except Exception:
        return x


def _est_sig(e):
    """Signature d'un estimateur scikit-learn : types et paramètres simples, SANS les noms d'étapes."""
    if isinstance(e, str) or e is None:
        return repr(e)
    if not hasattr(e, "get_params"):
        return type(e).__name__
    params = e.get_params(deep=False)
    sig = [type(e).__name__]
    for k in sorted(params):
        v = params[k]
        if isinstance(v, (list, tuple)) and v and all(isinstance(x, tuple) and len(x) >= 2 for x in v):
            items = []
            for t in v:
                cols = None
                if len(t) > 2:
                    c = t[2]
                    cols = [str(x) for x in c] if hasattr(c, "__iter__") and not isinstance(c, str) else c
                items.append((_est_sig(t[1]), repr(cols)))
            sig.append((k, tuple(items)))
        elif hasattr(v, "get_params"):
            sig.append((k, _est_sig(v)))
        elif isinstance(v, (str, int, float, bool, type(None))):
            sig.append((k, v))
        else:
            sig.append((k, type(v).__name__))
    return tuple(sig)


def _fitted(e):
    try:
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(e)
        return True
    except Exception:
        return False


def _same(a, b, strict=False):
    """(True, None) si a ≈ b, sinon (False, explication sans révéler la réponse)."""
    if isinstance(a, np.generic):
        a = a.item()
    if isinstance(b, np.generic):
        b = b.item()

    if isinstance(b, pd.DataFrame):
        if not isinstance(a, pd.DataFrame):
            return False, f"Tu obtiens {_kind(a)}, alors qu'on attend {_kind(b)}."
        if a.shape != b.shape:
            return False, f"Les dimensions ne correspondent pas (le tien : {a.shape[0]} lignes × {a.shape[1]} colonnes)."
        ca, cb = [str(c) for c in a.columns], [str(c) for c in b.columns]
        if ca != cb:
            if sorted(ca) == sorted(cb):
                return False, "Ce sont les bonnes colonnes, mais pas dans le même ordre."
            return False, "Les colonnes ne sont pas les bonnes."
        if _eq_frame(a, b):
            return True, None
        if not isinstance(b.index, pd.RangeIndex) and _eq_frame(_sorted_idx(a), _sorted_idx(b)):
            if strict:
                return False, "Bonnes valeurs, mais pas dans l'ordre demandé (pense au tri)."
            return True, None
        for c in cb:
            if not _eq_series(a[c] if c in a.columns else a.iloc[:, cb.index(c)], b.iloc[:, cb.index(c)]):
                return False, f"La colonne « {c} » ne contient pas les valeurs attendues."
        return False, "Les valeurs (ou l'index) ne correspondent pas à ce qui est attendu."

    if isinstance(b, pd.Series):
        if not isinstance(a, pd.Series):
            return False, f"Tu obtiens {_kind(a)}, alors qu'on attend {_kind(b)}."
        if len(a) != len(b):
            return False, f"Ton résultat a {len(a)} éléments, ce n'est pas le bon nombre."
        if _eq_series(a, b):
            return True, None
        if not isinstance(b.index, pd.RangeIndex) and _eq_series(_sorted_idx(a), _sorted_idx(b)):
            if strict:
                return False, "Bonnes valeurs, mais pas dans l'ordre demandé (pense au tri)."
            return True, None
        return False, "Les valeurs ne correspondent pas à ce qui est attendu."

    if isinstance(b, pd.Index):
        try:
            la, lb = [str(x) for x in list(a)], [str(x) for x in list(b)]
        except Exception:
            return False, f"Tu obtiens {_kind(a)}, alors qu'on attend {_kind(b)}."
        return (la == lb, None if la == lb else "Les noms ne correspondent pas.")

    if isinstance(b, np.ndarray):
        try:
            aa = np.asarray(a)
        except Exception:
            return False, f"Tu obtiens {_kind(a)}, alors qu'on attend {_kind(b)}."
        if aa.shape != b.shape:
            return False, f"La forme du tableau ne correspond pas (la tienne : {aa.shape})."
        try:
            ok = bool(np.allclose(aa.astype(float), b.astype(float), rtol=RTOL, atol=1e-6, equal_nan=True))
        except Exception:
            ok = bool(np.array_equal(aa, b))
        return (ok, None if ok else "Les valeurs du tableau ne correspondent pas.")

    if isinstance(b, (bool, np.bool_)):
        ok = isinstance(a, (bool, np.bool_)) and bool(a) == bool(b)
        return (ok, None if ok else "Ce n'est pas la bonne valeur (vrai/faux).")
    if isinstance(b, int):
        if isinstance(a, (int, float)) and not isinstance(a, bool) and a == b:
            return True, None
        return False, "Ce n'est pas le bon nombre." if isinstance(a, (int, float)) else f"Tu obtiens {_kind(a)}, alors qu'on attend un nombre."
    if isinstance(b, float):
        if isinstance(a, (int, float)) and not isinstance(a, bool):
            ok = _num_close(a, b)
            return (ok, None if ok else f"Ce n'est pas la bonne valeur (tu as {a:.6g}).")
        return False, f"Tu obtiens {_kind(a)}, alors qu'on attend un nombre."
    if isinstance(b, str):
        ok = isinstance(a, str) and a == b
        return (ok, None if ok else "Ce n'est pas le bon texte.")
    if b is None:
        return (a is None, None if a is None else "Ce n'est pas la valeur attendue.")

    if isinstance(b, (tuple, list)):
        if isinstance(a, np.ndarray):
            a = a.tolist()
        if not isinstance(a, (tuple, list)):
            return False, f"Tu obtiens {_kind(a)}, alors qu'on attend {_kind(b)}."
        if len(a) != len(b):
            return False, f"Il y a {len(a)} éléments dans ton résultat, ce n'est pas le bon nombre."
        for x, y in zip(a, b):
            ok, why = _same(x, y, strict)
            if not ok:
                return False, why
        return True, None
    if isinstance(b, dict):
        if not isinstance(a, dict) or set(a) != set(b):
            return False, "Les clés du dictionnaire ne correspondent pas."
        for k in b:
            ok, why = _same(a[k], b[k], strict)
            if not ok:
                return False, why
        return True, None
    if isinstance(b, (set, frozenset)):
        ok = isinstance(a, (set, frozenset, list, tuple)) and set(a) == set(b)
        return (ok, None if ok else "Les éléments ne correspondent pas.")

    if hasattr(b, "get_params"):
        if not hasattr(a, "get_params"):
            return False, f"Tu obtiens {_kind(a)}, alors qu'on attend un objet scikit-learn."
        if type(a) is not type(b):
            return False, f"Ce n'est pas le bon type d'objet (tu as un {type(a).__name__})."
        if _est_sig(a) != _est_sig(b):
            return False, "La configuration de l'objet ne correspond pas (étapes ou paramètres)."
        if _fitted(b) and not _fitted(a):
            return False, "L'objet n'a pas été entraîné : pense à fit / fit_transform."
        return True, None
    if callable(b):
        return (callable(a), None if callable(a) else "Ce n'est pas une fonction.")
    try:
        ok = bool(a == b)
    except Exception:
        ok = False
    if not ok and type(a) is type(b) and type(b).__eq__ is object.__eq__:
        ok = True  # objet sans notion d'égalité (ex. générateur aléatoire) : on juge sur les résultats affichés
    return (ok, None if ok else "Ce n'est pas la valeur attendue.")


# ------------------------------------------------------------------ extraction des variables modifiées par une solution
def _collect_target(t, out):
    if isinstance(t, ast.Name):
        out.append((t.id, None))
    elif isinstance(t, (ast.Tuple, ast.List)):
        for x in t.elts:
            _collect_target(x, out)
    elif isinstance(t, ast.Subscript):
        base = t.value
        while isinstance(base, (ast.Subscript, ast.Attribute)):
            base = base.value
        if isinstance(base, ast.Name):
            key = None
            sl = t.slice
            if isinstance(sl, ast.Constant):
                key = sl.value
            elif isinstance(sl, (ast.List, ast.Tuple)) and all(isinstance(x, ast.Constant) for x in sl.elts):
                key = [x.value for x in sl.elts]
            out.append((base.id, key))
    elif isinstance(t, ast.Attribute):
        base = t.value
        while isinstance(base, (ast.Subscript, ast.Attribute)):
            base = base.value
        if isinstance(base, ast.Name):
            out.append((base.id, None))


def _targets(code):
    out = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                _collect_target(t, out)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            _collect_target(node.target, out)
        elif isinstance(node, ast.FunctionDef):
            out.append((node.name, None))
    seen, uniq = set(), []
    for name, key in out:
        k = (name, repr(key))
        if k not in seen and not name.startswith("_"):
            seen.add(k)
            uniq.append((name, key))
    return uniq


def _grab(ns, name, key):
    v = ns[name]
    return v[key] if key is not None else v


# ------------------------------------------------------------------ noyau de référence
def _short(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, pd.DataFrame):
        return f"un DataFrame de forme {v.shape}"
    if isinstance(v, pd.Series):
        return "la Series " + repr(v.tolist()[:6])[:60]
    r = repr(v)
    return r if len(r) <= 70 else r[:67] + "…"


def _eval_tests(ns, tests):
    out = []
    for expr in tests:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                out.append((expr, True, _copy(eval(compile(expr, "<test>", "eval"), ns))))
        except BaseException as e:  # noqa: BLE001
            out.append((expr, False, f"{type(e).__name__}: {e}"))
    return out


def _is_mpl(v):
    items = list(v.ravel()[:50]) if isinstance(v, np.ndarray) and v.dtype == object else [v]
    return any(type(x).__module__.startswith("matplotlib") for x in items)


def _compute_exp(ns, code, tests=()):
    res = _exec_in(ns, code)
    states = []
    for name, key in _targets(code):
        try:
            v = _grab(ns, name, key)
            if _is_mpl(v):
                continue  # figure / axes : comparés via leur contenu (figs), pas comme variables
            states.append((name, key, _copy(v)))
        except Exception:
            pass
    return {
        "stdout": res["stdout"],
        "value": _copy(res["value"]) if res["has_value"] else None,
        "has_value": res["has_value"],
        "figs": res["figs"],
        "states": states,
        "err": res["err_text"] if res["exc"] is not None else None,
        "tests": _eval_tests(ns, tests) if tests and res["exc"] is None else [],
    }


def _ensure_exp(r, k, sols, alts_all, tests_all):
    """Calcule (une seule fois) l'attendu de la cellule k, après avoir avancé le noyau de référence jusqu'à k."""
    while r["pos"] <= k:
        i = r["pos"]
        alts = alts_all[i] if i < len(alts_all) else []
        pre = _clone(r["ns"]) if alts else None
        tests = tests_all[i] if i < len(tests_all) else []
        main = _compute_exp(r["ns"], sols[i], tests)
        alt_exps = []
        for code in alts:
            alt_exps.append(_compute_exp(_clone(pre), code, tests))
        r["exp"][i] = (main, alt_exps)
        r["pos"] = i + 1
    return r["exp"][k]


# ------------------------------------------------------------------ comparaison élève / attendu
def _contains_scalar(u, s):
    vals = []
    v = u["value"] if u["has_value"] else None
    if v is not None:
        if isinstance(v, (pd.Series, pd.Index)) and len(v) <= 20:
            vals = list(v.values)
        elif isinstance(v, pd.DataFrame) and v.size <= 40:
            vals = list(v.to_numpy().ravel())
        elif isinstance(v, np.ndarray) and v.size <= 20:
            vals = list(v.ravel())
        elif isinstance(v, (list, tuple, set)) and len(v) <= 20:
            vals = list(v)
        else:
            vals = [v]
    for x in vals:
        if isinstance(x, np.generic):
            x = x.item()
        if isinstance(s, str):
            if isinstance(x, str) and x == s:
                return True
        elif isinstance(x, (int, float)) and not isinstance(x, bool) and not isinstance(s, bool) and _num_close(x, s):
            return True
    text = u["stdout"]
    if isinstance(s, str):
        return bool(s) and s in text
    nums = _NUM.findall(text)
    es = repr(float(s)) if isinstance(s, float) else str(s)
    return 0 < len(nums) <= 3 and any(_num_text_ok(x, es) for x in nums)


_TOK = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|[^\W\d_]+")


def _is_numtok(t):
    return bool(t) and (t[0].isdigit() or (t[0] == "-" and len(t) > 1 and t[1].isdigit()))


def _tok_match(ut, et):
    """Même suite de mots et de nombres (mise en forme libre : parenthèses, virgules, espaces)."""
    a, b = _TOK.findall(ut), _TOK.findall(et)
    if not b or len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if _is_numtok(x) and _is_numtok(y):
            if not _num_text_ok(x, y):
                return False
        elif x != y:
            return False
    return True


def _out_match(u, e, strict):
    u_text = _norm(u["stdout"] + "\n" + (_vtext(u["value"]) if u["has_value"] else ""))
    e_text = _norm(e["stdout"] + "\n" + (_vtext(e["value"]) if e["has_value"] else ""))
    if u_text == e_text:
        return True, None
    reason = None
    if u["has_value"] and e["has_value"]:
        ok, why = _same(u["value"], e["value"], strict)
        if ok and (not e["stdout"].strip() or _norm(e["stdout"]) in _norm(u["stdout"])):
            return True, None
        reason = why
    if e["has_value"] and _is_scalar(e["value"]) and not e["stdout"].strip():
        if _contains_scalar(u, e["value"]):
            return True, None
    big = isinstance(e["value"], (pd.DataFrame, pd.Series, np.ndarray)) if e["has_value"] else False
    if not big and len(e_text) < 300 and _tok_match(u_text, e_text):
        return True, None
    if len(e_text) >= 8 and e_text in u_text:
        return True, None
    if not u["stdout"].strip() and not u["has_value"]:
        return False, "Ta cellule n'affiche rien : termine par ce que tu veux voir (dernière ligne) ou utilise print()."
    return False, reason or "Le résultat affiché ne correspond pas à ce qui est attendu."


def _arr_close(a, b):
    try:
        return a.shape == b.shape and bool(np.allclose(a, b, rtol=RTOL, atol=1e-6, equal_nan=True))
    except Exception:
        return False


def _figs_match(user_figs, exp_figs):
    if len(user_figs) < len(exp_figs):
        return False, "Il manque un graphique : pense à tracer puis à appeler plt.show()."
    for u, e in zip(user_figs, exp_figs):
        if u["axes"] != e["axes"]:
            return False, "Le nombre de zones de dessin (sous-graphiques) ne correspond pas à la consigne."
        for k in ("lines", "patches", "collections", "texts"):
            if (u[k] > 0) != (e[k] > 0):
                return False, "Le graphique n'a pas la forme attendue : relis la consigne (type de graphique, options demandées)."
        if not set(e["labels"]) <= set(u["labels"]):
            return False, "Les noms des axes ne correspondent pas à ceux demandés (xlabel / ylabel, ou mauvaises variables)."
        if not set(e["titles"]) <= set(u["titles"]):
            return False, "Le titre du graphique ne correspond pas à celui demandé."
        if e["legend"] and not u["legend"]:
            return False, "Il manque la légende (plt.legend())."
        if not np.allclose(e["size"], (6.4, 4.8), atol=0.05) and not np.allclose(u["size"], e["size"], atol=0.05):
            return False, "La taille de la figure (figsize) ne correspond pas à la consigne."
        # courbes : x et y
        rest = list(u["xy"])
        for ex, ey in e["xy"]:
            hit = next((i for i, (ux, uy) in enumerate(rest)
                        if _arr_close(uy, ey) and (ex is None or (ux is not None and _arr_close(ux, ex)))), None)
            if hit is None:
                return False, "Les valeurs tracées (courbe ou ligne) ne correspondent pas à ce qui est attendu."
            rest.pop(hit)
        if e.get("markers", 0) > 0 and u.get("markers", 0) == 0:
            return False, "Il manque les marqueurs (points) sur la courbe : option marker=\"o\"."
        # barres / histogramme (hauteurs ET largeurs : couvre les barres horizontales), graphique par graphique
        if e["heights"]:
            for ua, ea in zip(u["per_ax"], e["per_ax"]):
                ub = np.array(ua["bars"], dtype=float).reshape(-1, 2)
                eb = np.array(ea["bars"], dtype=float).reshape(-1, 2)
                if ub.shape != eb.shape or (eb.size and not _arr_close(ub, eb)):
                    return False, "Les barres ne correspondent pas à ce qui est attendu (valeurs, ordre, nombre de barres ou de classes)."
        # plusieurs zones de dessin : chacune doit contenir le bon contenu, dans l'ordre
        if e["axes"] > 1:
            for k, (ua, ea) in enumerate(zip(u["per_ax"], e["per_ax"]), 1):
                if ea["title"] and ua["title"] != ea["title"]:
                    return False, f"Le titre du graphique n°{k} ne correspond pas à celui demandé (bon graphique au bon endroit ?)."
                if len(ua["pts"]) != len(ea["pts"]) or len(ua["lines"]) != len(ea["lines"]):
                    return False, f"Le graphique n°{k} ne contient pas ce qui est demandé (bon type de graphique au bon endroit ?)."
                for ul, el in zip(ua["lines"], ea["lines"]):
                    if not _arr_close(ul, el):
                        return False, f"Les courbes du graphique n°{k} ne correspondent pas."
                for up_, ep_ in zip(ua["pts"], ea["pts"]):
                    if up_.shape != ep_.shape or not _arr_close(up_, ep_):
                        return False, f"Les points du graphique n°{k} ne correspondent pas à ce qui est attendu."
        # cartes de chaleur
        restm = list(u["mesh"])
        for m in e["mesh"]:
            hit = next((i for i, z in enumerate(restm) if _arr_close(z, m)), None)
            if hit is None:
                return False, "Les valeurs affichées dans la carte de chaleur ne correspondent pas."
            restm.pop(hit)
        # nuage de points
        if e["points"]:
            ep = np.vstack(e["points"])
            up = np.vstack(u["points"]) if u["points"] else np.empty((0, 2))
            if ep.shape != up.shape:
                return False, "Le nombre de points tracés ne correspond pas."
            ep = ep[np.lexsort((ep[:, 1], ep[:, 0]))]
            up = up[np.lexsort((up[:, 1], up[:, 0]))]
            if not _arr_close(up, ep):
                return False, "Les points tracés ne correspondent pas à ce qui est attendu (mauvaises colonnes ou axes inversés ?)."
    return True, None


def _kind(dt):
    k = getattr(dt, "kind", "O")
    s = str(dt)
    if s == "category" or s.startswith("string"):
        return "O"
    return {"i": "i", "u": "i", "f": "f", "b": "b", "M": "M", "m": "m"}.get(k, "O")


def _dtype_diff(uv, refv):
    """Nom d'une colonne dont le type (entier / décimal / texte / date…) diffère, sinon None."""
    try:
        if isinstance(uv, pd.DataFrame) and isinstance(refv, pd.DataFrame):
            for c in refv.columns:
                if c in uv.columns and _kind(uv[c].dtype) != _kind(refv[c].dtype):
                    return f"{c} : {uv[c].dtype} au lieu de {refv[c].dtype}"
        elif isinstance(uv, pd.Series) and isinstance(refv, pd.Series):
            if _kind(uv.dtype) != _kind(refv.dtype):
                return f"{uv.dtype} au lieu de {refv.dtype}"
    except Exception:
        pass
    return None


def _state_match(ns, e, strict):
    for name, key, refv in e["states"]:
        label = f"`{name}`" if key is None else f"`{name}[{key!r}]`"
        if name not in ns:
            return False, f"La variable `{name}` n'existe pas : as-tu fait l'affectation avec « = » ?"
        try:
            uv = _grab(ns, name, key)
        except Exception:
            return False, f"{label} est introuvable."
        ok, why = _same(uv, refv, strict)
        if not ok:
            return False, f"{label} n'a pas la valeur attendue. {why}"
        dd = _dtype_diff(uv, refv)
        if dd:
            return False, f"{label} a les bonnes valeurs mais pas le bon type de données ({dd})."
    return True, None


def _comparable(v):
    """Vrai si on peut réellement comparer cette valeur (pas un objet opaque comme un générateur aléatoire)."""
    if isinstance(v, np.ndarray) and v.dtype == object:
        return all(_comparable(x) for x in v.ravel()[:50]) and not any(
            type(x).__module__.startswith("matplotlib") for x in v.ravel()[:50])
    return isinstance(v, (pd.DataFrame, pd.Series, pd.Index, np.ndarray, str, bool, int, float, list, tuple,
                          dict, set, frozenset, np.generic)) or hasattr(v, "get_params")


def _compare(u, e, strict, ns):
    e_out = bool(e["stdout"].strip()) or e["has_value"]
    u_out = bool(u["stdout"].strip()) or u["has_value"]
    note = ""
    state_ok, state_why = (True, None)
    if e["states"]:
        state_ok, state_why = _state_match(ns, e, strict)
    informative = any(_comparable(refv) for _n, _k, refv in e["states"])
    if e_out and (u_out or not informative):
        ok, why = _out_match(u, e, strict)
        if not ok:
            return {"ok": False, "msg": why}
        if e["states"] and not state_ok:
            note = "Ton affichage est bon, mais la variable n'a pas été modifiée comme prévu : les cellules suivantes en dépendent. " + (state_why or "")
    elif e["states"]:
        if not state_ok:
            return {"ok": False, "msg": state_why}
    if e["figs"]:
        ok, why = _figs_match(u["figs"], e["figs"])
        if not ok:
            return {"ok": False, "msg": why}
    for expr, ok_ref, refval in e.get("tests", []):
        if not ok_ref:
            continue
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                got = eval(compile(expr, "<test>", "eval"), ns)
        except BaseException as ex:  # noqa: BLE001
            return {"ok": False, "msg": f"Ton code plante quand on l'essaie avec `{expr}` : {type(ex).__name__}: {ex}"}
        same, _why = _same(got, refval, True)
        if not same:
            return {"ok": False, "msg": f"Test `{expr}` : on attend {_short(refval)}, ton code donne {_short(got)}."}
    return {"ok": True, "note": note}


def check_cell(owner, setup, sols_json, alts_json, j, strict_json, tests_json="[]"):
    sols, alts_all, strict = json.loads(sols_json), json.loads(alts_json), json.loads(strict_json)
    tests_all = json.loads(tests_json)
    u = _last
    if not u:
        return json.dumps({"status": "unchecked", "msg": "Rien à vérifier."})
    if u.get("empty"):
        return json.dumps({"status": "empty", "msg": "Écris ton code dans la cellule, puis exécute-la."})
    if u.get("exc") is not None:
        return json.dumps({"status": "error", "msg": _friendly(u["exc"])})

    r = _refs.get(owner)
    if r is None:
        r = {"ns": {"__name__": "__main__"}, "pos": 0, "exp": {}, "setup_err": None}
        _refs[owner] = r
        res = _exec_in(r["ns"], setup)
        if res["exc"] is not None:
            r["setup_err"] = res["err_text"]
    if r["setup_err"]:
        return json.dumps({"status": "unchecked", "msg": "Correction automatique indisponible (préparation de référence échouée)."})

    main, alt_exps = _ensure_exp(r, j, sols, alts_all, tests_all)
    first_fail = None
    for e in [main] + alt_exps:
        if e["err"]:
            continue
        v = _compare(u, e, bool(strict[j]), _ns)
        if v["ok"]:
            return json.dumps({"status": "ok", "note": v.get("note", "")})
        first_fail = first_fail or v
    if first_fail is None:
        return json.dumps({"status": "unchecked", "msg": "Correction automatique indisponible pour cette cellule (ton code s'est exécuté sans erreur)."})
    return json.dumps({"status": "wrong", "msg": first_fail["msg"]})
