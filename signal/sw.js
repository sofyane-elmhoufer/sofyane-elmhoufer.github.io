/* The Signal: offline shell. Everything cached here is already encrypted. Version 756ef75e */
const CACHE = 'signal-__VER__';
const ASSETS = ['./', 'index.html', 'app.enc.json', 'data.enc.json', 'manifest.webmanifest', 'icons/icon-192.png', 'icons/icon-512.png', 'icons/apple-touch-icon.png'];
self.addEventListener('install', (e) => { e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting())); });
self.addEventListener('activate', (e) => { e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener('fetch', (e) => {
  const r = e.request;
  if (r.method !== 'GET') return;
  const u = new URL(r.url);
  if (u.origin !== location.origin) {
    // fonts only; API calls for live headlines go straight to the network
    if (!/fonts\.(googleapis|gstatic)\.com$/.test(u.host)) return;
    e.respondWith(caches.open(CACHE).then(async (c) => {
      const m = await c.match(r);
      const f = fetch(r).then((res) => { if (res.ok || res.type === 'opaque') c.put(r, res.clone()); return res; }).catch(() => m);
      return m || f;
    }));
    return;
  }
  // network first so the daily data file is always fresh; cache is the offline fallback
  e.respondWith(fetch(r).then((res) => { const cp = res.clone(); caches.open(CACHE).then((c) => c.put(r, cp)); return res; })
    .catch(() => caches.match(r, { ignoreSearch: true }).then((m) => m || caches.match('index.html'))));
});
