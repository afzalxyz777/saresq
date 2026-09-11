/* Service worker for the installed app.
 *
 * Served from / (not /static/) because a worker can only control URLs at or
 * below its own path, and this one has to control the whole console.
 *
 * The caching policy has one rule that matters more than the rest: LIVE DATA
 * IS NEVER SERVED FROM CACHE. Everything under /api/ is network-only. A shell
 * that loads offline is useful; a two-hour-old aircraft position presented as
 * current would be worse than a blank screen, because the operator cannot tell
 * the difference. When the ground station is unreachable the pages say so.
 *
 * What IS cached is the shell: the HTML, the CSS, the renderer, and the 97 KB
 * of baked-in OSM geometry and SRTM terrain -- which is exactly the payload
 * that makes this thing work in a field tent with no internet.
 */
const VERSION = "saresq-v3";
const SHELL = [
  "/", "/radar", "/map", "/evidence", "/review", "/analytics",
  "/static/console.css",
  "/static/basemap.js",
  "/static/mapcore.js",
  "/static/mapview.js",
  "/static/radar.js",
  "/static/manifest.webmanifest",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png"
];

self.addEventListener("install", (e) => {
  e.waitUntil((async () => {
    const cache = await caches.open(VERSION);
    // addAll() rejects the whole batch if any single request fails, which
    // would leave the app permanently uninstallable because of one 404.
    await Promise.all(SHELL.map((u) =>
      cache.add(new Request(u, { cache: "reload" })).catch(() => {})));
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  // Live data: never cached, never faked.
  if (url.pathname.startsWith("/api/")) return;

  // Evidence blobs are content-addressed and can be large. Cache what has
  // already been fetched so a reviewed clip stays viewable off-network, but
  // never go to the network twice for the same immutable blob.
  if (url.pathname.startsWith("/media/") || url.pathname.startsWith("/thumbs/")) {
    e.respondWith((async () => {
      const cache = await caches.open(VERSION + "-media");
      const hit = await cache.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) cache.put(req, res.clone());
      return res;
    })());
    return;
  }

  // Pages: network first, so a running ground station always wins, with the
  // cached shell as the offline fallback.
  if (req.mode === "navigate") {
    e.respondWith((async () => {
      try {
        const res = await fetch(req);
        const cache = await caches.open(VERSION);
        cache.put(req, res.clone());
        return res;
      } catch (err) {
        return (await caches.match(req)) || (await caches.match("/radar"))
            || new Response("Offline and this page was never cached.",
                            { status: 503, headers: { "Content-Type": "text/plain" } });
      }
    })());
    return;
  }

  // Static assets: cache first. basemap.js is 97 KB and changes about never.
  e.respondWith((async () => {
    const hit = await caches.match(req);
    if (hit) return hit;
    try {
      const res = await fetch(req);
      if (res.ok) (await caches.open(VERSION)).put(req, res.clone());
      return res;
    } catch (err) {
      return new Response("", { status: 504 });
    }
  })());
});
