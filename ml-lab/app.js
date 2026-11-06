import { decryptPayload } from "./crypto.js";

const PYODIDE_BASE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
const SEABORN_WHEEL = "wheels/seaborn-0.13.2-py3-none-any.whl";
const CHAT_URL_RE = /^https:\/\/claude\.ai\/(chat|share)\/[0-9a-fA-F-]{20,}([/?#].*)?$/;
const isShareUrl = (u) => /^https:\/\/claude\.ai\/share\//.test(u);

/* ------------------------------------------------------------ utilitaires */
const $ = (s, r = document) => r.querySelector(s);
function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === false || v == null) continue;
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return e;
}
const store = {
  get(k, d = null) { try { const v = localStorage.getItem("mllab:" + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("mllab:" + k, JSON.stringify(v)); } catch { /* stockage indisponible */ } },
};
function toast(msg) {
  const t = el("div", { class: "toast" }, msg);
  document.body.append(t);
  setTimeout(() => t.remove(), 3200);
}
const htmlToText = (h) => { const d = document.createElement("div"); d.innerHTML = h; return d.textContent.replace(/\s+/g, " ").trim(); };
const clip = (s, n) => (s.length > n ? s.slice(0, n) + "\n…(tronqué)" : s);
const textOf = (n) => (n ? (n.innerText ?? n.textContent ?? "") : "");
const isBlankCode = (code) => !code.replace(/#.*$/gm, "").trim();

/* ------------------------------------------------------------ état */
let DATA = null; // { lessons, datasets, seabornNames, claudeChatUrl? }
let py = null;
let pyPromise = null;
let execCount = 0;
let kernelOwner = null;
let activeCell = null;
let currentLesson = null;
let deferredInstall = null;
let replaying = false;

/* ------------------------------------------------------------ Python (Pyodide) */
function loadScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = src; s.onload = res; s.onerror = () => rej(new Error("Impossible de charger " + src));
    document.head.append(s);
  });
}
function setPyStatus(kind, text) {
  const st = $("#pyStatus"), bn = $("#pyBanner");
  st.className = "py-status " + kind;
  st.textContent = kind === "ready" ? "🐍 prêt" : kind === "error" ? "⚠️ Python" : "⏳ Python";
  if (kind === "ready") { bn.hidden = true; return; }
  bn.hidden = false;
  bn.className = "py-banner" + (kind === "error" ? " error" : "");
  bn.replaceChildren(el("span", {}, text));
  if (kind === "error") bn.append(el("button", { class: "btn small", onclick: () => startPython() }, "Réessayer"));
}
function startPython() {
  if (pyPromise) return pyPromise;
  setPyStatus("loading", "Python se charge… La première fois : ~100 Mo (Wi-Fi conseillé). Ensuite, tout reste en cache.");
  pyPromise = (async () => {
    await loadScript(PYODIDE_BASE + "pyodide.js");
    py = await globalThis.loadPyodide({ indexURL: PYODIDE_BASE });
    setPyStatus("loading", "Chargement de pandas, matplotlib, scikit-learn…");
    await py.loadPackage(["numpy", "pandas", "matplotlib", "scikit-learn", "micropip"]);
    setPyStatus("loading", "Installation de seaborn…");
    const whl = new URL(SEABORN_WHEEL, location.href).href;
    await py.runPythonAsync(`import micropip\nawait micropip.install("${whl}")`);
    const cwd = py.FS.cwd();
    py.FS.mkdirTree(cwd + "/seaborn-data");
    for (const [name, text] of Object.entries(DATA.datasets)) {
      py.FS.writeFile(cwd + "/" + name, text);
      if (DATA.seabornNames.includes(name)) py.FS.writeFile(cwd + "/seaborn-data/" + name, text);
    }
    for (const f of ["runner.py", "checker.py"]) py.runPython(await (await fetch(f)).text());
    setPyStatus("ready", "");
  })().catch((e) => {
    pyPromise = null;
    setPyStatus("error", "Python n'a pas pu se charger (" + (e.message || e) + "). Vérifie ta connexion.");
    throw e;
  });
  return pyPromise;
}
let queue = Promise.resolve();
/** Exécute une cellule ; si `check` est fourni, vérifie ensuite la réponse (même tâche : pas d'interférence). */
function runCode(code, check) {
  const job = queue.then(async () => {
    await startPython();
    await new Promise((r) => setTimeout(r, 25)); // laisse l'interface se redessiner
    const t0 = performance.now();
    const res = JSON.parse(await py.globals.get("run_cell")(code));
    res.ms = performance.now() - t0;
    if (check) {
      try {
        res.verdict = JSON.parse(await py.globals.get("check_cell")(check.owner, check.setup, check.sols, check.alts, check.j, check.strict, check.tests));
      } catch (e) {
        res.verdict = { status: "unchecked", msg: "Vérification indisponible : " + (e.message || e) };
      }
    }
    return res;
  });
  queue = job.catch(() => {});
  return job;
}
async function resetKernel() {
  await startPython();
  await py.globals.get("reset")();
  kernelOwner = null;
}

/* ------------------------------------------------------------ cellules */
function renderResult(out, res) {
  out.replaceChildren();
  out.append(el("div", { class: "exec-meta" }, `exécuté en ${res.ms < 1000 ? Math.round(res.ms) + " ms" : (res.ms / 1000).toFixed(1) + " s"}`));
  if (res.stdout) out.append(el("pre", { class: "out-text" }, res.stdout.replace(/\n$/, "")));
  if (res.result_html) out.append(el("div", { class: "df-wrap", html: res.result_html }));
  else if (res.result != null) out.append(el("pre", { class: "out-text" }, res.result));
  for (const b64 of res.images || []) out.append(el("img", { class: "out-img", alt: "graphique", src: "data:image/png;base64," + b64 }));
  if (res.error) out.append(el("pre", { class: "err" }, res.error));
}

/**
 * Cellule de code façon Jupyter.
 * o = { key, ctx, q, starter, hint, solution, tag, check, beforeRun, trackDone, onProgress, onSolved, onDelete }
 */
function cellView(o) {
  const root = el("div", { class: "cell" });
  const ta = el("textarea", { rows: "1", spellcheck: "false", autocapitalize: "none", autocorrect: "off", "aria-label": "Code Python" });
  ta.value = store.get("code:" + o.key, o.starter);
  const prompt = el("div", { class: "prompt" }, "[ ]");
  const out = el("div", { class: "out" });
  const verdict = el("div", { class: "verdict", hidden: true, role: "status" });
  const runBtn = el("button", { class: "btn small primary" }, "▶ Exécuter");
  let tries = 0, chk = null;

  const autosize = () => { ta.style.height = "auto"; ta.style.height = ta.scrollHeight + 2 + "px"; };
  const activate = () => {
    if (activeCell?.root) activeCell.root.classList.remove("active");
    activeCell = { ...o.ctx, ta, out, root };
    root.classList.add("active");
  };
  const isDone = () => !!store.get("done:" + o.key, false);
  const setDone = (v) => {
    store.set("done:" + o.key, v);
    root.classList.toggle("done", v);
    if (chk) chk.checked = v;
  };

  let saveTimer;
  ta.addEventListener("input", () => {
    autosize(); activate();
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => store.set("code:" + o.key, ta.value), 400);
  });
  ta.addEventListener("focus", activate);
  ta.addEventListener("keydown", (ev) => {
    if (ev.key === "Tab") {
      ev.preventDefault();
      ta.setRangeText("    ", ta.selectionStart, ta.selectionEnd, "end");
      ta.dispatchEvent(new Event("input"));
    } else if (ev.key === "Enter" && (ev.shiftKey || ev.ctrlKey || ev.metaKey)) {
      ev.preventDefault(); run();
    }
  });

  function showVerdict(kind, title, detail, extra) {
    verdict.hidden = false;
    verdict.className = "verdict " + kind;
    verdict.replaceChildren(el("div", { class: "v-title" }, title));
    if (detail) verdict.append(el("div", { class: "v-detail" }, detail));
    if (extra) verdict.append(el("div", { class: "v-extra" }, extra));
  }

  /** Affiche le verdict d'une cellule d'exercice et valide / fait avancer si c'est correct. */
  function handleVerdict(res) {
    if (res.error) {
      tries += 1;
      showVerdict("error", "⚠️ Il y a une erreur dans ton code", res.friendly, tries >= 3 ? helpLine() : "Lis le message rouge ci-dessous : la dernière ligne dit quel est le problème.");
      return "error";
    }
    const v = res.verdict;
    if (!v) return null;
    if (v.status === "ok") {
      const wasDone = isDone();
      tries = 0;
      showVerdict("ok", wasDone ? "✅ Correct (déjà validé)" : "✅ Correct ! Question validée", v.note ? "⚠️ " + v.note : null);
      if (!wasDone) { setDone(true); o.onProgress?.(); o.onSolved?.(); }
      return "ok";
    }
    if (v.status === "wrong") {
      tries += 1;
      showVerdict("wrong", "❌ Pas encore ça", v.msg, tries >= 3 ? helpLine() : "Relis la consigne, ou ouvre l'indice 💡, puis modifie ton code et ré-exécute.");
      return "wrong";
    }
    if (v.status === "empty") { showVerdict("info", "✍️ Cellule vide", v.msg); return "empty"; }
    if (v.status === "error") {
      tries += 1;
      showVerdict("error", "⚠️ Il y a une erreur dans ton code", v.msg, tries >= 3 ? helpLine() : null);
      return "error";
    }
    showVerdict("info", "ℹ️ Vérification automatique", v.msg);
    return "unchecked";
  }
  const helpLine = () => "Bloqué ? Ouvre l'indice 💡, regarde la solution ✅, ou clique sur « Discuter avec Claude » : ton code et ce résultat sont envoyés automatiquement.";

  async function run() {
    activate();
    runBtn.disabled = true;
    prompt.textContent = "[*]";
    verdict.hidden = true;
    out.replaceChildren(el("div", { class: "exec-meta" }, "Exécution… (au premier lancement, Python se charge)"));
    let status = null;
    try {
      if (o.beforeRun) await o.beforeRun();
      activate(); // la préparation automatique a pu activer une autre cellule
      const res = await runCode(ta.value, o.check);
      execCount += 1;
      prompt.textContent = `[${execCount}]`;
      renderResult(out, res);
      if (o.check) status = handleVerdict(res);
      else if (res.error) showVerdict("error", "⚠️ Il y a une erreur dans ton code", res.friendly);
    } catch (e) {
      prompt.textContent = "[!]";
      out.replaceChildren(el("pre", { class: "err" }, "Erreur : " + (e.message || e)));
    } finally {
      runBtn.disabled = false;
    }
    return status;
  }
  runBtn.addEventListener("click", run);

  const tools = el("div", { class: "tools" }, runBtn);
  let hintBox = null, solBox = null;
  if (o.hint) {
    tools.append(el("button", { class: "btn small", onclick: () => {
      if (hintBox) { hintBox.remove(); hintBox = null; return; }
      hintBox = el("div", { class: "hint", html: "💡 " + o.hint });
      tools.after(hintBox);
    } }, "💡 Indice"));
  }
  if (o.solution) {
    tools.append(el("button", { class: "btn small", onclick: () => {
      if (solBox) { solBox.remove(); solBox = null; return; }
      solBox = el("div", { class: "solution" },
        el("div", { class: "muted", style: "margin-bottom:6px" }, "✅ Solution (essaie d'abord seul !)"),
        el("pre", {}, o.solution),
        el("button", { class: "btn small", onclick: () => { ta.value = o.solution; ta.dispatchEvent(new Event("input")); toast("Solution insérée dans la cellule"); } }, "Insérer dans la cellule"));
      (hintBox || tools).after(solBox);
    } }, "✅ Solution"));
  }
  tools.append(el("button", { class: "btn small", title: "Remettre le code de départ", onclick: () => { ta.value = o.starter; ta.dispatchEvent(new Event("input")); store.set("code:" + o.key, o.starter); } }, "↺"));
  tools.append(el("span", { class: "spacer" }));
  if (o.trackDone) {
    chk = el("input", { type: "checkbox" });
    chk.checked = isDone();
    root.classList.toggle("done", chk.checked);
    chk.addEventListener("change", () => { setDone(chk.checked); o.onProgress?.(); });
    tools.append(el("label", {}, chk, "Validé"));
  }
  if (o.onDelete) tools.append(el("button", { class: "btn small", onclick: o.onDelete }, "🗑"));

  if (o.q) root.append(el("div", { class: "q" }, el("span", { class: "tag" }, o.tag || "Question"), el("span", { html: o.q })));
  root.append(el("div", { class: "row" }, prompt, ta), tools, verdict, out);
  requestAnimationFrame(autosize);
  return { root, run, ta, out, isDone };
}

/* noyau : réinitialise l'espace de noms quand on change d'exercice */
const ownerReset = (owner) => async () => {
  if (kernelOwner === owner) return;
  await resetKernel();
  kernelOwner = owner;
};

/* ------------------------------------------------------------ vues */
const view = () => $("#view");
function setTop(title, back) {
  $("#topTitle").textContent = title;
  $("#backBtn").hidden = !back;
}
const spaceOf = (l) => l.space || "ml";
const SPACES = [
  { id: "ml", emoji: "🧪", name: "Workflow ML", title: "Le cycle de vie du machine learning, pas à pas",
    sub: "Exploration, nettoyage, préparation : le workflow avant l'entraînement." },
  { id: "py", emoji: "🐍", name: "Python & librairies", title: "Maîtriser Python, NumPy, Pandas et Matplotlib",
    sub: "Les bases du langage et des librairies de traitement de données, avec beaucoup de pratique." },
];
const GROUP_INFO = {
  Python: "Les bases du langage", NumPy: "Calcul sur tableaux", Pandas: "Manipuler des tableaux de données",
  Matplotlib: "Tracer des graphiques", Seaborn: "Graphiques statistiques",
};
function lessonProgress(lesson) {
  let total = 0, done = 0;
  lesson.exercises.forEach((ex, i) => ex.cells.forEach((_, j) => { total++; if (store.get(`done:${lesson.id}:${i}:${j}`, false)) done++; }));
  return { total, done };
}

function renderHome(spaceId) {
  currentLesson = null; activeCell = null;
  setTop("ML Lab", false);
  $("#chatFab").hidden = false;
  const sid = spaceId || store.get("space", "ml");
  const space = SPACES.find((x) => x.id === sid) || SPACES[0];
  store.set("space", space.id);
  const lessons = DATA.lessons.filter((l) => spaceOf(l) === space.id);
  const stat = (list) => list.reduce((a, l) => { const p = lessonProgress(l); return { T: a.T + p.total, D: a.D + p.done }; }, { T: 0, D: 0 });

  const tabs = el("div", { class: "tabs", role: "tablist" });
  SPACES.forEach((sp) => {
    const st = stat(DATA.lessons.filter((l) => spaceOf(l) === sp.id));
    tabs.append(el("button", { class: "tab" + (sp.id === space.id ? " on" : ""), role: "tab", "aria-selected": sp.id === space.id ? "true" : "false",
      onclick: () => renderHome(sp.id) },
      el("span", { class: "t-emoji" }, sp.emoji), el("span", { class: "t-name" }, sp.name),
      el("span", { class: "t-count" }, `${st.D}/${st.T}`)));
  });

  const { T, D } = stat(lessons);
  const hero = el("div", { class: "hero" },
    el("h2", {}, space.title),
    el("p", { class: "muted" }, `${space.sub} Lis la règle, écris le code, exécute : l'app te dit si c'est juste. ${D} / ${T} questions validées.`),
    el("div", { class: "global-progress" }, el("i", { style: `width:${T ? (100 * D) / T : 0}%` })));

  const card = (l) => {
    const p = lessonProgress(l);
    const complete = p.total > 0 && p.done === p.total;
    return el("button", { class: "card" + (complete ? " complete" : ""), onclick: () => renderLesson(l) },
      el("div", { class: "emoji" }, complete ? "✅" : l.emoji),
      el("div", { class: "meta" },
        el("div", { class: "name" }, `${l.id}. ${l.title}`),
        el("div", { class: "sub" }, `${l.exercises.length} exercices · ${p.done}/${p.total} questions validées`),
        el("div", { class: "bar" }, el("i", { style: `width:${p.total ? (100 * p.done) / p.total : 0}%` }))));
  };
  const list = el("div", {});
  if (space.id === "py") {
    const groups = [...new Set(lessons.map((l) => l.group))];
    groups.forEach((g) => {
      const gl = lessons.filter((l) => l.group === g);
      const st = stat(gl);
      list.append(el("div", { class: "group-head" }, el("h3", {}, g), el("span", { class: "muted" }, `${GROUP_INFO[g] || ""} · ${st.D}/${st.T}`)));
      list.append(el("div", { class: "cards" }, gl.map(card)));
    });
  } else {
    list.append(el("div", { class: "cards" }, lessons.map(card)));
  }
  const sandbox = el("div", { class: "cards" }, el("button", { class: "card", onclick: renderSandbox },
    el("div", { class: "emoji" }, "🧰"),
    el("div", { class: "meta" }, el("div", { class: "name" }, "Bac à sable"), el("div", { class: "sub" }, "Code libre sur tous les datasets, autant de cellules que tu veux"))));
  view().replaceChildren(tabs, hero, installCard(), list, sandbox);
  window.scrollTo(0, 0);
}

function installCard() {
  const standalone = matchMedia("(display-mode: standalone)").matches || navigator.standalone;
  if (standalone) return el("div");
  const box = el("div", { class: "panel install" });
  if (deferredInstall) {
    box.append(el("p", {}, "📲 Installe l'app sur ton téléphone pour l'ouvrir comme une application."),
      el("button", { class: "btn primary", onclick: async () => { deferredInstall.prompt(); await deferredInstall.userChoice; deferredInstall = null; renderHome(); } }, "Installer l'application"));
  } else {
    box.append(el("p", { class: "muted" }, "📲 Pour installer : sur Android (Chrome) menu ⋮ → « Installer l'application » ; sur iPhone (Safari) Partager → « Sur l'écran d'accueil »."));
  }
  return box;
}

function renderLesson(lesson) {
  currentLesson = lesson; activeCell = null;
  setTop(`${lesson.emoji} ${lesson.title}`, true);
  $("#chatFab").hidden = false;
  const v = view();
  v.replaceChildren();

  /* progression de la leçon + revalidation */
  const progLabel = el("span", {});
  const progBar = el("i", {});
  const exos = []; // { details, upd, replay, cells, count }
  const footer = el("div", { class: "panel lesson-done", hidden: true });
  const updProg = () => {
    const p = lessonProgress(lesson);
    progLabel.textContent = `${p.done} / ${p.total} questions validées`;
    progBar.style.width = (p.total ? (100 * p.done) / p.total : 0) + "%";
    const complete = p.total > 0 && p.done === p.total;
    footer.hidden = !complete;
    if (complete) {
      const sameSpace = DATA.lessons.filter((l) => spaceOf(l) === spaceOf(lesson));
      const next = sameSpace[sameSpace.indexOf(lesson) + 1];
      footer.replaceChildren(el("h3", {}, "🎉 Leçon terminée !"),
        el("p", {}, "Toutes les questions de cette leçon sont validées."),
        next ? el("button", { class: "btn primary", onclick: () => renderLesson(next) }, `Leçon suivante : ${next.title} →`)
          : el("button", { class: "btn primary", onclick: () => renderHome(spaceOf(lesson)) }, "Retour à l'accueil"));
    }
  };
  const revalidate = async () => {
    if (replaying) return;
    replaying = true;
    const before = lessonProgress(lesson).done;
    revalBtn.disabled = true;
    try {
      for (const x of exos) await x.replay();
    } finally { replaying = false; revalBtn.disabled = false; }
    exos.forEach((x) => x.upd()); updProg();
    const after = lessonProgress(lesson);
    toast(`Revalidation terminée : ${after.done}/${after.total} questions validées` + (after.done > before ? ` (+${after.done - before})` : ""));
  };
  const revalBtn = el("button", { class: "btn small", title: "Ré-exécute tout le code que tu as déjà écrit dans cette leçon et valide ce qui est juste", onclick: revalidate }, "🔄 Revalider tout mon code");
  v.append(el("div", { class: "lesson-progress" },
    el("div", { class: "lp-row" }, progLabel, revalBtn),
    el("div", { class: "global-progress" }, progBar)));

  v.append(el("div", { class: "panel" }, el("h3", {}, "La règle"), el("div", { html: lesson.rule })));

  const ex0 = cellView({
    key: `${lesson.id}:example`, tag: "Exemple", q: "Exemple à exécuter et à modifier :", starter: lesson.example,
    ctx: { lesson, exTitle: "Exemple de la leçon", q: "Exemple de la leçon" },
    beforeRun: ownerReset(`example:${lesson.id}`),
  });
  v.append(el("div", { class: "panel" }, el("h3", {}, "Le code"), ex0.root));

  lesson.exercises.forEach((ex, i) => {
    const owner = `ex:${lesson.id}:${i}`;
    const body = el("div", { class: "exo-body" });
    const count = el("span", { class: "count" });
    const banner = el("div", { class: "exo-done", hidden: true }, "🎉 Exercice terminé !");
    const sols = JSON.stringify(ex.cells.map((c) => c.solution));
    const alts = JSON.stringify(ex.cells.map((c) => c.alts || []));
    const strict = JSON.stringify(ex.cells.map((c) => !!c.strict));
    const tests = JSON.stringify(ex.cells.map((c) => c.tests || []));
    let cells = [];
    const upd = () => {
      const n = cells.filter((c) => c.isDone()).length;
      count.textContent = `${n}/${ex.cells.length}`;
      banner.hidden = n !== ex.cells.length;
      d.classList.toggle("finished", n === ex.cells.length);
    };
    if (ex.intro) body.append(el("p", { html: ex.intro }));

    const setup = cellView({
      key: `${lesson.id}:${i}:setup`, tag: "Préparation", q: "Exécute d'abord cette cellule (elle charge les données) :", starter: ex.setup,
      ctx: { lesson, exTitle: ex.title, q: "Préparation (chargement des données)" },
      beforeRun: ownerReset(owner),
    });
    // la préparation redémarre toujours le noyau pour cet exercice
    const setupRun = async () => { kernelOwner = null; await setup.run(); };

    /* après une bonne réponse : on avance vers la question suivante */
    const onSolved = (j) => {
      upd(); updProg();
      if (replaying) return;
      const next = cells.slice(j + 1).find((c) => !c.isDone());
      if (next) { setTimeout(() => next.root.scrollIntoView({ behavior: "smooth", block: "center" }), 650); return; }
      if (cells.every((c) => c.isDone())) {
        const nextExo = exos[i + 1];
        toast("🎉 Exercice terminé !");
        if (nextExo) setTimeout(() => { nextExo.details.open = true; nextExo.details.scrollIntoView({ behavior: "smooth", block: "start" }); }, 900);
        else setTimeout(() => footer.scrollIntoView({ behavior: "smooth", block: "center" }), 900);
      }
    };
    cells = ex.cells.map((c, j) => cellView({
      key: `${lesson.id}:${i}:${j}`, tag: `Question ${j + 1}`, q: c.q, starter: c.starter, hint: c.hint, solution: c.solution,
      ctx: { lesson, exTitle: ex.title, q: htmlToText(c.q) }, trackDone: true,
      onProgress: () => { upd(); updProg(); },
      onSolved: () => onSolved(j),
      check: { owner, setup: ex.setup, sols, alts, strict, tests, j },
      beforeRun: async () => { if (kernelOwner !== owner) await setupRun(); },
    }));
    const replay = async () => {
      await setupRun();
      for (const c of cells) if (!isBlankCode(c.ta.value)) await c.run();
    };
    body.append(el("div", { class: "exo-actions" },
      el("button", { class: "btn small", title: "Ré-exécute la préparation puis toutes tes cellules dans l'ordre", onclick: async () => {
        replaying = true;
        try { await replay(); } finally { replaying = false; }
        upd(); updProg();
      } }, "⏩ Tout rejouer et vérifier")));
    body.append(banner, setup.root, ...cells.map((c) => c.root));
    const d = el("details", { class: "exo" }, el("summary", {}, `Exercice ${i + 1} · ${ex.title}`, count), body);
    exos.push({ details: d, upd, replay });
    upd();
    v.append(d);
  });
  // premier exercice non terminé ouvert
  const firstOpen = exos.find((x) => x.details.querySelector(".count").textContent.split("/")[0] !== x.details.querySelector(".count").textContent.split("/")[1]) || exos[0];
  firstOpen.details.open = true;
  v.append(footer);
  updProg();
  window.scrollTo(0, 0);
}

function renderSandbox() {
  currentLesson = { id: "sandbox", title: "Bac à sable", rule: "Code libre." }; activeCell = null;
  setTop("🧰 Bac à sable", true);
  $("#chatFab").hidden = false;
  const names = DATA.seabornNames.map((n) => n.replace(".csv", "")).join(", ");
  const files = Object.keys(DATA.datasets).join(", ");
  const v = view();
  v.replaceChildren(el("div", { class: "panel" },
    el("h3", {}, "Code libre"),
    el("p", { html: `Écris n'importe quel code Python. Les cellules partagent leurs variables, comme dans Jupyter (<code>Maj+Entrée</code> pour exécuter).` }),
    el("p", { html: `Datasets seaborn : <code>sns.load_dataset("…")</code> avec : ${names}.` }),
    el("p", { class: "muted" }, `Fichiers CSV disponibles avec pd.read_csv : ${files}`)));
  const list = el("div");
  let cellsCodes = store.get("sandbox:ids", [1]);
  const save = () => store.set("sandbox:ids", cellsCodes);
  const beforeRun = ownerReset("sandbox");
  const add = (id) => {
    const c = cellView({
      key: "sandbox:" + id, tag: "Cellule", starter: id === 1 ? 'import pandas as pd\nimport seaborn as sns\n\ndf = sns.load_dataset("titanic")\ndf.head()' : "# ton code ici\n",
      ctx: { lesson: currentLesson, exTitle: "Bac à sable", q: "Code libre" }, beforeRun,
      onDelete: () => { c.root.remove(); cellsCodes = cellsCodes.filter((x) => x !== id); save(); },
    });
    list.append(c.root);
  };
  cellsCodes.forEach(add);
  v.append(list, el("button", { class: "btn", onclick: () => { const id = Math.max(0, ...cellsCodes) + 1; cellsCodes.push(id); save(); add(id); } }, "+ Ajouter une cellule"));
  window.scrollTo(0, 0);
}

/* ------------------------------------------------------------ discussion avec Claude (conversation « Request ») */
function getChatUrl() {
  const local = store.get("chatUrl", "");
  if (local && CHAT_URL_RE.test(local)) return local;
  return DATA?.claudeChatUrl && CHAT_URL_RE.test(DATA.claudeChatUrl) ? DATA.claudeChatUrl : "";
}

function buildPrompt(question, guided) {
  const c = activeCell;
  const lesson = c?.lesson || currentLesson;
  const sp = (c?.lesson || currentLesson)?.space;
  const parts = [sp === "py"
    ? "[ML Lab] J'apprends Python pour la data science et le machine learning (NumPy, Pandas, Matplotlib, Seaborn) avec mon app d'exercices."
    : "[ML Lab] Je m'entraîne au workflow de préparation des données pour le machine learning (pandas, seaborn, scikit-learn) avec mon app d'exercices."];
  if (lesson) {
    parts.push(`\nLeçon : ${lesson.title}`);
    if (lesson.rule && lesson.id !== "sandbox") parts.push(`Règle de la leçon : ${clip(htmlToText(lesson.rule), 500)}`);
  } else {
    parts.push("\nJe suis sur l'écran d'accueil de l'app.");
  }
  if (c) {
    parts.push(`Exercice : ${c.exTitle}`);
    parts.push(`Question : ${c.q}`);
    parts.push("\nMon code :\n```python\n" + clip(c.ta.value.trim(), 1500) + "\n```");
    const verdictText = textOf(c.root?.querySelector(".verdict:not([hidden])")).trim();
    if (verdictText) parts.push("Verdict de l'app : " + clip(verdictText.replace(/\s+/g, " "), 400));
    const outText = textOf(c.out).replace(/^exécuté en .*\n?/i, "").replace(/^Exécution….*$/m, "").trim();
    parts.push("Résultat affiché dans l'app :\n```\n" + (outText ? clip(outText, 1200) : "(pas encore exécuté)") + "\n```");
  }
  parts.push("\nMa question : " + (question.trim() || "Peux-tu vérifier mon code et m'expliquer ce que je dois comprendre ici ?"));
  if (guided) parts.push("\nGuide-moi avec des explications et des indices progressifs, sans me donner directement la solution complète, sauf si je te le demande.");
  return parts.join("\n");
}

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch { /* repli */ }
  try {
    const t = el("textarea", { style: "position:fixed;opacity:0" }); t.value = text; document.body.append(t); t.select();
    const ok = document.execCommand("copy"); t.remove(); return ok;
  } catch { return false; }
}

function openChatSheet() {
  const sheet = $("#sheet");
  const close = () => { sheet.hidden = true; sheet.replaceChildren(); };
  const q = el("textarea", { placeholder: "Pose ta question sur ce que tu vois à l'écran (laisse vide pour demander une vérification)…" });
  const guided = el("input", { type: "checkbox" }); guided.checked = true;
  const preview = el("textarea", { class: "preview", readonly: true });
  const refresh = () => { preview.value = buildPrompt(q.value, guided.checked); };
  q.addEventListener("input", refresh); guided.addEventListener("change", refresh); refresh();
  const where = activeCell ? `${activeCell.lesson?.title || ""} › ${activeCell.exTitle} › ${activeCell.q}` : (currentLesson ? currentLesson.title : "Accueil");

  /* destination : la conversation « Request » */
  const dest = el("div", { class: "dest" });
  const drawDest = (editing) => {
    const url = getChatUrl();
    dest.replaceChildren();
    if (url && !editing) {
      dest.append(el("div", {}, "📨 Envoi vers ta conversation « Request »"),
        isShareUrl(url) ? el("div", { class: "muted", style: "font-size:12px" }, "ℹ️ C'est un lien de partage : il ouvre une copie en lecture de la conversation. Pour écrire dans la conversation d'origine, remplace-le par son adresse https://claude.ai/chat/… (ouvre-la dans ton compte et copie l'adresse).") : null,
        el("button", { class: "btn small", onclick: () => drawDest(true) }, "Changer le lien"));
    } else {
      const inp = el("input", { type: "url", placeholder: "https://claude.ai/chat/…  ou  https://claude.ai/share/…", value: url || "" });
      const msg = el("div", { class: "muted", style: "font-size:12px" }, "Ouvre ta conversation « Request » dans Claude, copie son adresse, colle-la ici une seule fois.");
      dest.append(el("div", {}, "📨 Conversation « Request »"), inp, msg,
        el("button", { class: "btn small primary", onclick: () => {
          const val = inp.value.trim();
          if (!CHAT_URL_RE.test(val)) { msg.textContent = "Ce lien n'est pas valide : il doit ressembler à https://claude.ai/chat/xxxxxxxx-… ou https://claude.ai/share/xxxxxxxx-…"; msg.style.color = "var(--err)"; return; }
          store.set("chatUrl", val); toast("Lien enregistré sur cet appareil"); drawDest(false);
        } }, "Enregistrer le lien"));
    }
  };
  drawDest(false);

  const open = async () => {
    const text = buildPrompt(q.value, guided.checked);
    const copied = await copyText(text);
    const chat = getChatUrl();
    if (chat) {
      window.open(chat, "_blank", "noopener");
      toast(copied ? "Message copié : colle-le dans « Request »." : "Copie le texte ci-dessous puis colle-le dans « Request ».");
    } else {
      const url = "https://claude.ai/new?q=" + encodeURIComponent(text);
      window.open(url.length <= 6000 ? url : "https://claude.ai/new", "_blank", "noopener");
      toast("Lien « Request » non enregistré : nouvelle conversation ouverte. Enregistre le lien ci-dessus pour tout regrouper.");
    }
  };
  sheet.replaceChildren(el("div", { class: "sheet-card" },
    el("h3", {}, "💬 Discuter avec Claude"),
    el("div", { class: "ctx" }, "Contexte envoyé : ", where, ". Ton code, le verdict et le résultat sont inclus."),
    dest,
    q,
    el("label", { class: "chk" }, guided, "Guide-moi sans me donner la solution directement"),
    el("details", {}, el("summary", { class: "muted" }, "Voir le message complet"), preview),
    el("div", { class: "btns" },
      el("button", { class: "btn primary", onclick: open }, "Copier et ouvrir « Request »"),
      el("button", { class: "btn", onclick: async () => toast((await copyText(preview.value)) ? "Message copié" : "Copie impossible") }, "Copier seulement"),
      el("button", { class: "btn", onclick: close }, "Fermer"))));
  sheet.hidden = false;
  sheet.onclick = (e) => { if (e.target === sheet) close(); };
}

/* ------------------------------------------------------------ démarrage / verrouillage */
async function onUnlock(ev) {
  ev.preventDefault();
  const pw = $("#pw").value, msg = $("#lockMsg"), btn = $("#unlockBtn");
  msg.textContent = ""; btn.disabled = true; btn.textContent = "Déchiffrement…";
  try {
    const env = await (await fetch("content.enc", { cache: "no-cache" })).json();
    DATA = await decryptPayload(env, pw);
  } catch (e) {
    msg.textContent = e && e.name === "OperationError" ? "Mot de passe incorrect." : "Impossible de lire le contenu (connexion ?).";
    btn.disabled = false; btn.textContent = "Déverrouiller"; $("#pw").select();
    return;
  }
  $("#pw").value = "";
  $("#lock").hidden = true;
  $("#app").hidden = false;
  renderHome();
  startPython().catch(() => {});
}

window.addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); deferredInstall = e; if (DATA && !currentLesson) renderHome(); });
$("#lockForm").addEventListener("submit", onUnlock);
$("#lockBtn").addEventListener("click", () => location.reload());
$("#backBtn").addEventListener("click", () => renderHome(currentLesson && currentLesson.space ? currentLesson.space : undefined));
$("#chatFab").addEventListener("click", openChatSheet);
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
