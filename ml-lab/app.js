import { decryptPayload } from "./crypto.js";

const PYODIDE_BASE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
const SEABORN_WHEEL = "wheels/seaborn-0.13.2-py3-none-any.whl";

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
  setTimeout(() => t.remove(), 2600);
}
const htmlToText = (h) => { const d = document.createElement("div"); d.innerHTML = h; return d.textContent.replace(/\s+/g, " ").trim(); };
const clip = (s, n) => (s.length > n ? s.slice(0, n) + "\n…(tronqué)" : s);

/* ------------------------------------------------------------ état */
let DATA = null; // { lessons, datasets, seabornNames }
let py = null;
let pyPromise = null;
let execCount = 0;
let kernelOwner = null;
let activeCell = null;
let currentLesson = null;
let deferredInstall = null;

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
    const runner = await (await fetch("runner.py")).text();
    py.runPython(runner);
    setPyStatus("ready", "");
  })().catch((e) => {
    pyPromise = null;
    setPyStatus("error", "Python n'a pas pu se charger (" + (e.message || e) + "). Vérifie ta connexion.");
    throw e;
  });
  return pyPromise;
}
let queue = Promise.resolve();
function runCode(code) {
  const job = queue.then(async () => {
    await startPython();
    await new Promise((r) => setTimeout(r, 25)); // laisse l'interface se redessiner
    const t0 = performance.now();
    const res = JSON.parse(py.globals.get("run_cell")(code));
    res.ms = performance.now() - t0;
    return res;
  });
  queue = job.catch(() => {});
  return job;
}
async function resetKernel() {
  await startPython();
  py.globals.get("reset")();
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
 * Crée une cellule de code façon Jupyter.
 * o = { key, ctx:{lesson, exTitle, q}, q (html), starter, hint, solution, tag, beforeRun, trackDone, onProgress, onDelete }
 */
function cellView(o) {
  const root = el("div", { class: "cell" });
  const ta = el("textarea", { rows: "1", spellcheck: "false", autocapitalize: "none", autocorrect: "off", "aria-label": "Code Python" });
  ta.value = store.get("code:" + o.key, o.starter);
  const prompt = el("div", { class: "prompt" }, "[ ]");
  const out = el("div", { class: "out" });
  const runBtn = el("button", { class: "btn small primary" }, "▶ Exécuter");

  const autosize = () => { ta.style.height = "auto"; ta.style.height = ta.scrollHeight + 2 + "px"; };
  const ctx = () => ({ ...o.ctx, ta, out });
  const activate = () => {
    if (activeCell?.root) activeCell.root.classList.remove("active");
    activeCell = { ...ctx(), root };
    root.classList.add("active");
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
      const s = ta.selectionStart, e2 = ta.selectionEnd;
      ta.setRangeText("    ", s, e2, "end");
      ta.dispatchEvent(new Event("input"));
    } else if (ev.key === "Enter" && (ev.shiftKey || ev.ctrlKey || ev.metaKey)) {
      ev.preventDefault(); run();
    }
  });

  async function run() {
    activate();
    runBtn.disabled = true;
    prompt.textContent = "[*]";
    out.replaceChildren(el("div", { class: "exec-meta" }, "Exécution… (au premier lancement, Python se charge)"));
    try {
      if (o.beforeRun) await o.beforeRun();
      activate(); // la préparation automatique a pu activer une autre cellule
      const res = await runCode(ta.value);
      execCount += 1;
      prompt.textContent = `[${execCount}]`;
      renderResult(out, res);
    } catch (e) {
      prompt.textContent = "[!]";
      out.replaceChildren(el("pre", { class: "err" }, "Erreur : " + (e.message || e)));
    } finally {
      runBtn.disabled = false;
    }
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
    const chk = el("input", { type: "checkbox" });
    chk.checked = !!store.get("done:" + o.key, false);
    root.classList.toggle("done", chk.checked);
    chk.addEventListener("change", () => {
      store.set("done:" + o.key, chk.checked);
      root.classList.toggle("done", chk.checked);
      o.onProgress?.();
    });
    tools.append(el("label", {}, chk, "Fait"));
  }
  if (o.onDelete) tools.append(el("button", { class: "btn small", onclick: o.onDelete }, "🗑"));

  if (o.q) root.append(el("div", { class: "q" }, el("span", { class: "tag" }, o.tag || "Question"), el("span", { html: o.q })));
  root.append(el("div", { class: "row" }, prompt, ta), tools, out);
  requestAnimationFrame(autosize);
  return { root, run, ta, out };
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
function lessonProgress(lesson) {
  let total = 0, done = 0;
  lesson.exercises.forEach((ex, i) => ex.cells.forEach((_, j) => { total++; if (store.get(`done:${lesson.id}:${i}:${j}`, false)) done++; }));
  return { total, done };
}

function renderHome() {
  currentLesson = null; activeCell = null;
  setTop("ML Lab", false);
  $("#chatFab").hidden = false;
  let T = 0, D = 0;
  DATA.lessons.forEach((l) => { const p = lessonProgress(l); T += p.total; D += p.done; });
  const hero = el("div", { class: "hero" },
    el("h2", {}, "Le cycle de vie du machine learning, pas à pas"),
    el("p", { class: "muted" }, "Lis la règle, écris le code, regarde le résultat. " + D + " / " + T + " exercices terminés."),
    el("div", { class: "global-progress" }, el("i", { style: `width:${T ? (100 * D) / T : 0}%` })));
  const cards = el("div", { class: "cards" });
  DATA.lessons.forEach((l) => {
    const p = lessonProgress(l);
    cards.append(el("button", { class: "card", onclick: () => renderLesson(l) },
      el("div", { class: "emoji" }, l.emoji),
      el("div", { class: "meta" },
        el("div", { class: "name" }, `${l.id}. ${l.title}`),
        el("div", { class: "sub" }, `${l.exercises.length} exercices · ${p.done}/${p.total} cellules faites`),
        el("div", { class: "bar" }, el("i", { style: `width:${p.total ? (100 * p.done) / p.total : 0}%` })))));
  });
  cards.append(el("button", { class: "card", onclick: renderSandbox },
    el("div", { class: "emoji" }, "🧰"),
    el("div", { class: "meta" }, el("div", { class: "name" }, "Bac à sable"), el("div", { class: "sub" }, "Code libre sur tous les datasets, autant de cellules que tu veux"))));
  view().replaceChildren(hero, installCard(), cards);
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
    const upd = () => { count.textContent = `${ex.cells.filter((_, j) => store.get(`done:${lesson.id}:${i}:${j}`, false)).length}/${ex.cells.length}`; };
    if (ex.intro) body.append(el("p", { html: ex.intro }));

    const setup = cellView({
      key: `${lesson.id}:${i}:setup`, tag: "Préparation", q: "Exécute d'abord cette cellule (elle charge les données) :", starter: ex.setup,
      ctx: { lesson, exTitle: ex.title, q: "Préparation (chargement des données)" },
      beforeRun: ownerReset(owner),
    });
    // la préparation redémarre toujours le noyau pour cet exercice
    const setupRun = async () => { kernelOwner = null; await setup.run(); };
    const cells = ex.cells.map((c, j) => cellView({
      key: `${lesson.id}:${i}:${j}`, tag: `Question ${j + 1}`, q: c.q, starter: c.starter, hint: c.hint, solution: c.solution,
      ctx: { lesson, exTitle: ex.title, q: htmlToText(c.q) }, trackDone: true, onProgress: upd,
      beforeRun: async () => { if (kernelOwner !== owner) await setupRun(); },
    }));
    body.append(el("div", { class: "exo-actions" },
      el("button", { class: "btn small", onclick: async () => { await setupRun(); for (const c of cells) if (c.ta.value.replace(/#.*$/gm, "").trim()) await c.run(); } }, "⏩ Tout rejouer")));
    body.append(setup.root, ...cells.map((c) => c.root));
    const d = el("details", { class: "exo" }, el("summary", {}, `Exercice ${i + 1} · ${ex.title}`, count), body);
    if (i === 0) d.open = true;
    upd();
    v.append(d);
  });
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

/* ------------------------------------------------------------ discussion avec Claude */
function buildPrompt(question, guided) {
  const c = activeCell;
  const lesson = c?.lesson || currentLesson;
  const parts = ["Je m'entraîne au workflow de préparation des données pour le machine learning (pandas, seaborn, scikit-learn) avec une app d'exercices."];
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
    const outText = (c.out.innerText || "").replace(/^exécuté en .*\n?/i, "").replace(/^Exécution….*$/m, "").trim();
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
  const q = el("textarea", { placeholder: "Pose ta question sur ce que tu vois à l'écran (laisse vide pour demander une vérification)…" });
  const guided = el("input", { type: "checkbox" }); guided.checked = true;
  const preview = el("textarea", { class: "preview", readonly: true });
  const refresh = () => { preview.value = buildPrompt(q.value, guided.checked); };
  q.addEventListener("input", refresh); guided.addEventListener("change", refresh); refresh();
  const where = activeCell ? `${activeCell.lesson?.title || ""} › ${activeCell.exTitle} › ${activeCell.q}` : (currentLesson ? currentLesson.title : "Accueil");
  const open = async () => {
    const text = buildPrompt(q.value, guided.checked);
    const copied = await copyText(text);
    const url = "https://claude.ai/new?q=" + encodeURIComponent(text);
    window.open(url.length <= 6000 ? url : "https://claude.ai/new", "_blank", "noopener");
    toast(copied ? "Contexte copié. Colle-le dans Claude s'il n'apparaît pas." : "Copie le texte ci-dessous puis colle-le dans Claude.");
  };
  sheet.replaceChildren(el("div", { class: "sheet-card" },
    el("h3", {}, "💬 Discuter avec Claude"),
    el("div", { class: "ctx" }, "Contexte envoyé : ", where, ". Ton code et son résultat sont inclus."),
    q,
    el("label", { class: "chk" }, guided, "Guide-moi sans me donner la solution directement"),
    el("details", {}, el("summary", { class: "muted" }, "Voir le message complet"), preview),
    el("div", { class: "btns" },
      el("button", { class: "btn primary", onclick: open }, "Copier et ouvrir Claude"),
      el("button", { class: "btn", onclick: async () => toast((await copyText(preview.value)) ? "Message copié" : "Copie impossible") }, "Copier seulement"),
      el("button", { class: "btn", onclick: () => { sheet.hidden = true; sheet.replaceChildren(); } }, "Fermer"))));
  sheet.hidden = false;
  sheet.onclick = (e) => { if (e.target === sheet) { sheet.hidden = true; sheet.replaceChildren(); } };
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
$("#backBtn").addEventListener("click", renderHome);
$("#chatFab").addEventListener("click", openChatSheet);
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
