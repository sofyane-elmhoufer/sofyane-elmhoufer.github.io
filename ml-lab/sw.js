// Service worker : permet l'utilisation hors-ligne après le premier chargement.
const SHELL = "ml-lab-shell-v1";
const CDN = "ml-lab-cdn-v1";
const SHELL_FILES = [
  "./", "index.html", "style.css", "app.js", "crypto.js", "runner.py", "content.enc",
  "manifest.webmanifest", "icons/icon-192.png", "icons/icon-512.png",
  "wheels/seaborn-0.13.2-py3-none-any.whl",
];
const CDN_HOSTS = ["cdn.jsdelivr.net"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(SHELL_FILES)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(
    // Les caches sont partagés par tout le site : on ne supprime que les anciens caches de CETTE app.
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k.startsWith("ml-lab-") && k !== SHELL && k !== CDN).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

async function networkFirst(req) {
  const cache = await caches.open(SHELL);
  try {
    const res = await fetch(req, { cache: "no-cache" });
    if (res.ok) cache.put(req, res.clone());
    return res;
  } catch (err) {
    const hit = (await cache.match(req, { ignoreSearch: true })) || (req.mode === "navigate" ? await cache.match("index.html") : null);
    if (hit) return hit;
    throw err;
  }
}
async function cacheFirst(req) {
  const cache = await caches.open(CDN);
  const hit = await cache.match(req);
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok || res.type === "opaque") cache.put(req, res.clone());
  return res;
}

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin === self.location.origin) e.respondWith(networkFirst(req));
  else if (CDN_HOSTS.includes(url.hostname)) e.respondWith(cacheFirst(req));
});
