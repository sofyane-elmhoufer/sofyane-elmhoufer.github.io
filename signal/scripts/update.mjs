// Daily edition updater. Runs in GitHub Actions (Node 22), working directory = signal/.
//   APP_PASSWORD      (required)  decrypts and re-encrypts data.enc.json. Never written anywhere.
//   ANTHROPIC_API_KEY (optional)  writes plain-language explanations and micro-lessons.
//   ANTHROPIC_MODEL   (optional)  default: claude-haiku-5-5
// Every network step is best-effort: if a source fails, the edition is built from what worked.
import fs from 'node:fs';
import { encryptJSON, decryptJSON, AAD_DATA } from '../lib/crypto.mjs';

const PW = process.env.APP_PASSWORD;
if (!PW) { console.error('APP_PASSWORD is not set'); process.exit(1); }
const FILE = new URL('../data.enc.json', import.meta.url);
const TODAY = new Date().toISOString().slice(0, 10);
const KEEP_DAYS = 75, MAX_STORIES = 140, MAX_NEW = 8, MAX_FRESH_NEW = 2;

const env = JSON.parse(fs.readFileSync(FILE, 'utf8'));
const data = await decryptJSON(env, PW, AAD_DATA);
data.stories ||= []; data.fresh ||= []; data.glossary ||= []; data.radar ||= [];

async function get(url, ms = 15000, type = 'json') {
  const c = new AbortController(), t = setTimeout(() => c.abort(), ms);
  try {
    const r = await fetch(url, { signal: c.signal, headers: { 'user-agent': 'signal-daily/1.0 (+personal study app)', accept: '*/*' } });
    if (!r.ok) throw new Error(`${r.status} ${url}`);
    return type === 'json' ? await r.json() : await r.text();
  } finally { clearTimeout(t); }
}
const safe = async (name, fn) => { try { const r = await fn(); console.log(`ok   ${name}: ${r.length}`); return r; } catch (e) { console.warn(`skip ${name}: ${e.message}`); return []; } };

const strip = (s = '') => s.replace(/<!\[CDATA\[|\]\]>/g, '').replace(/<[^>]+>/g, ' ').replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;|&#x27;/g, "'").replace(/\s+/g, ' ').trim();
const tag = (x, t) => (x.match(new RegExp(`<${t}[^>]*>([\\s\\S]*?)</${t}>`, 'i')) || [])[1] || '';
function parseFeed(xml, source) {
  const items = xml.match(/<item[\s>][\s\S]*?<\/item>|<entry[\s>][\s\S]*?<\/entry>/gi) || [];
  return items.slice(0, 12).map((x) => {
    const link = (x.match(/<link[^>]*href="([^"]+)"/i) || [])[1] || strip(tag(x, 'link'));
    const d = new Date(strip(tag(x, 'pubDate') || tag(x, 'published') || tag(x, 'updated')));
    return { title: strip(tag(x, 'title')), url: link, source, date: isNaN(d) ? TODAY : d.toISOString().slice(0, 10), desc: strip(tag(x, 'description') || tag(x, 'summary') || tag(x, 'content')).slice(0, 600) };
  }).filter((i) => i.title && /^https?:/.test(i.url));
}

const AI_Q = 'AI OR LLM OR "language model" OR GPU OR database OR PostgreSQL OR "vector database"';
const sources = [
  ['Hacker News', async () => (await get(`https://hn.algolia.com/api/v1/search_by_date?tags=story&numericFilters=points%3E100&hitsPerPage=40&query=${encodeURIComponent(AI_Q)}`)).hits.filter((h) => h.url && h.title).map((h) => ({ title: h.title, url: h.url, source: 'Hacker News', date: h.created_at.slice(0, 10), desc: '', points: h.points }))],
  ['Hugging Face Papers', async () => (await get('https://huggingface.co/api/daily_papers')).slice(0, 10).map((p) => ({ title: p.paper.title, url: 'https://huggingface.co/papers/' + p.paper.id, source: 'Hugging Face Papers', date: (p.publishedAt || TODAY).slice(0, 10), desc: (p.paper.summary || '').replace(/\s+/g, ' ').slice(0, 600), cat: 'research' }))],
  ['OpenAI', async () => parseFeed(await get('https://openai.com/news/rss.xml', 15000, 'text'), 'OpenAI')],
  ['Google DeepMind', async () => parseFeed(await get('https://deepmind.google/blog/rss.xml', 15000, 'text'), 'Google DeepMind')],
  ['Hugging Face blog', async () => parseFeed(await get('https://huggingface.co/blog/feed.xml', 15000, 'text'), 'Hugging Face blog')],
  ['MIT Technology Review AI', async () => parseFeed(await get('https://www.technologyreview.com/topic/artificial-intelligence/feed', 15000, 'text'), 'MIT Technology Review')],
  ['NVIDIA blog', async () => parseFeed(await get('https://blogs.nvidia.com/feed/', 15000, 'text'), 'NVIDIA blog')],
  ['Ars Technica AI', async () => parseFeed(await get('https://arstechnica.com/ai/feed/', 15000, 'text'), 'Ars Technica')]
];

const KW = { models: /\b(llm|gpt|claude|gemini|llama|mistral|qwen|deepseek|model|transformer|diffusion|reasoning|agent)s?\b/i, research: /\b(paper|arxiv|benchmark|study|researchers?)\b/i, hardware: /\b(gpu|nvidia|chip|tpu|hardware|silicon|data ?cent(er|re)|memory|hbm)\b/i, data: /\b(database|postgres|sql|data ?lake|pipeline|vector|duckdb|spark|etl|iceberg)\b/i, policy: /\b(regulat|law|act|ban|policy|copyright|safety|court|ai act)\b/i, tools: /\b(open[- ]source|sdk|framework|library|release|mcp|cli|api)\b/i, industry: /\b(raises|funding|acquires|acquisition|launch|startup|billion)\b/i };
const classify = (t) => Object.keys(KW).find((k) => KW[k].test(t)) || 'industry';
const relevant = (i) => /\b(ai|a\.i\.|llm|gpt|claude|gemini|model|gpu|nvidia|chip|data|database|sql|postgres|agent|machine learning|neural|openai|anthropic|deepmind|robot|cuda|inference|training)\b/i.test(i.title + ' ' + i.desc);
const norm = (s) => s.toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').replace(/\s+/g, ' ').trim();
const slug = (s) => norm(s).split(' ').slice(0, 6).join('-').slice(0, 48);

// ---- glossary matching: which terms and lessons does an item touch? ----
const termHits = (text) => {
  const t = ' ' + norm(text) + ' ', out = [];
  for (const g of data.glossary) if (g.id !== 'ai' && [g.t, ...(g.aka || [])].some((n) => t.includes(' ' + norm(n) + ' '))) out.push(g);
  return out;
};
const lessonsFor = (hits) => { const c = {}; hits.forEach((g) => g.l && (c[g.l] = (c[g.l] || 0) + 1)); return Object.keys(c).sort((a, b) => c[b] - c[a]).slice(0, 2); };

// ---- optional: Anthropic API explanations ----
const KEY = process.env.ANTHROPIC_API_KEY, MODEL = process.env.ANTHROPIC_MODEL || 'claude-haiku-5-5';
async function ask(prompt, max = 1800) {
  const r = await fetch('https://api.anthropic.com/v1/messages', { method: 'POST', headers: { 'content-type': 'application/json', 'x-api-key': KEY, 'anthropic-version': '2023-06-01' }, body: JSON.stringify({ model: MODEL, max_tokens: max, messages: [{ role: 'user', content: prompt }] }) });
  if (!r.ok) throw new Error('anthropic ' + r.status);
  const j = await r.json(); const t = j.content?.map((c) => c.text || '').join('') || '';
  const m = t.match(/\{[\s\S]*\}/); if (!m) throw new Error('no json'); return JSON.parse(m[0]);
}
const lessonIndex = 'hw-1..hw-6 (hardware), os-1..os-3 (OS/PC), net-1..net-2 (network), data-1..data-6 (data/databases), ai-1..ai-6 (AI)';
async function explain(item) {
  const p = `You write for a first-year data-science and AI engineering student (French speaker, strong English). Using ONLY the facts below, return strict JSON with keys: summary (2 sentences, plain English, use **bold** for key terms), eli5 (1-2 sentences, an everyday analogy), why (2-3 sentences: why it matters for a student/engineer; hedge claims you cannot verify), level ("Beginner"|"Intermediate"|"Advanced"), learn (array of up to 2 Academy lesson ids from: ${lessonIndex}). Do not invent numbers, names or dates that are not in the facts.\n\nTitle: ${item.title}\nSource: ${item.source}\nText: ${item.desc || '(headline only)'}`;
  return ask(p, 700);
}
async function microLesson(item) {
  const p = `Write a 2-minute micro-lesson for a first-year data-science and AI engineering student about the key concept behind this news item. Use ONLY well-established facts. Return strict JSON: {"title":"<=60 chars","hook":"one sentence","body":"3-5 short sentences, **bold** key terms","points":["3 bullets"],"quiz":[{"q":"","o":["4 options"],"a":<index>,"why":""}]}.\n\nNews: ${item.title} (${item.source}). ${item.desc}`;
  return ask(p, 1100);
}

// ---- gather ----
const results = (await Promise.all(sources.map(([n, f]) => safe(n, f)))).flat();
const known = new Set(data.stories.map((s) => s.url)), knownT = new Set(data.stories.map((s) => norm(s.title)));
const seen = new Set();
let fresh = results.filter((i) => {
  const key = i.url.replace(/[?#].*$/, '');
  if (known.has(i.url) || seen.has(key) || knownT.has(norm(i.title)) || !relevant(i)) return false;
  const age = (Date.now() - new Date(i.date)) / 864e5; if (age > 4 || age < -1) return false;
  seen.add(key); return true;
});
fresh.sort((a, b) => (b.points || 0) - (a.points || 0) || b.date.localeCompare(a.date));
fresh = fresh.slice(0, MAX_NEW);
console.log(`new candidates: ${fresh.length}`);

const added = [];
let rank = 20, freshLessons = 0;
for (const it of fresh) {
  const hits = termHits(it.title + ' ' + it.desc);
  const story = { id: `${it.date}-${slug(it.title)}`, rank: rank++, date: it.date, cat: it.cat || classify(it.title + ' ' + it.desc), level: 'Intermediate', title: it.title, source: it.source, url: it.url, summary: it.desc ? it.desc.slice(0, 320) + (it.desc.length > 320 ? '…' : '') : '', learn: lessonsFor(hits), auto: true };
  if (KEY) {
    try {
      const x = await explain(it);
      Object.assign(story, { summary: String(x.summary || story.summary), eli5: String(x.eli5 || ''), why: String(x.why || ''), level: ['Beginner', 'Intermediate', 'Advanced'].includes(x.level) ? x.level : 'Intermediate' });
      const valid = (x.learn || []).filter((id) => /^(hw|os|net|data|ai)-\d$/.test(id)); if (valid.length) story.learn = valid.slice(0, 2);
      story.note = 'Explanation drafted automatically by Claude from the source text. Check the original for details.';
    } catch (e) { console.warn('explain failed:', e.message); }
    if (freshLessons < MAX_FRESH_NEW && hits.length === 0 && story.eli5) {
      try {
        const m = await microLesson(it);
        if (m.title && m.body && Array.isArray(m.quiz) && m.quiz.every((q) => Array.isArray(q.o) && q.o.length >= 2 && Number.isInteger(q.a) && q.a >= 0 && q.a < q.o.length)) {
          data.fresh.push({ id: 'fresh-' + slug(m.title), date: it.date, title: m.title, story: story.id, hook: m.hook || '', body: m.body, points: m.points || [], quiz: m.quiz.slice(0, 2) }); freshLessons++;
        }
      } catch (e) { console.warn('micro-lesson failed:', e.message); }
    }
  } else story.note = 'Headline pulled automatically from a public feed. A fuller explanation appears when an AI key is configured.';
  added.push(story);
}

// ---- merge, prune, write ----
const cutoff = new Date(Date.now() - KEEP_DAYS * 864e5).toISOString().slice(0, 10);
data.stories = [...added, ...data.stories].filter((s) => !s.auto || (s.date || '') >= cutoff).slice(0, MAX_STORIES); // hand-written seed stories are never pruned
const fids = new Set(); data.fresh = data.fresh.filter((f) => !fids.has(f.id) && fids.add(f.id)).filter((f) => f.date >= cutoff).slice(-30);
data.meta = { ...(data.meta || {}), generated: new Date().toISOString(), edition: TODAY, added: added.length };

// Always re-written, so 'edition built' moves forward and the repo shows activity (GitHub pauses scheduled workflows after 60 days without repo activity).
const out = await encryptJSON(data, PW, AAD_DATA, env.salt, env.iter);
fs.writeFileSync(FILE, JSON.stringify(out));
console.log(`wrote data.enc.json: +${added.length} stories, ${data.stories.length} total, ${data.fresh.length} micro-lessons`);
